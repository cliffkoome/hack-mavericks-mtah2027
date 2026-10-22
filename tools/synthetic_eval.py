"""Score pipeline variants on several synthetic matches: a correctness and robustness check.

    python tools/synthetic_eval.py                       # 5 full-size seeds, default variants
    python tools/synthetic_eval.py --size small --seeds 0 1 2

Read the output for what it can say -- does a change work as intended, is it stable across seeds,
how sensitive is it to a parameter -- and not for what it cannot: the synthetic matches come from
tools/make_synthetic.py, whose assumptions overlap with our own models, so a gain here is an upper
bound on the real gain. Never pick a value because it wins here (EXPERIMENTS.md).
"""

import argparse
import copy
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
from config import PARAMS  # noqa: E402
from possession import atd, pipeline, spec as spec_module  # noqa: E402
import make_synthetic  # noqa: E402
import possession_metrics as pm  # noqa: E402

EVAL_DIR = os.path.join(ROOT, ".synthetic", "eval")


def variants():
    """name -> params. The defaults, the decoder, its evidence weight either side, and E003."""
    out = {"vote": dict(copy.deepcopy(PARAMS), model="vote"),
           "hmm": dict(copy.deepcopy(PARAMS), model="hmm")}
    for weight in (0.1, 0.4):
        params = dict(copy.deepcopy(PARAMS), model="hmm")
        params["evidence"]["weight"] = weight
        out["hmm w={}".format(weight)] = params
    no_gap = dict(copy.deepcopy(PARAMS), model="hmm")
    no_gap["evidence"]["gap_acc_strong"] = None
    out["hmm no-gap (E003)"] = no_gap
    gap_dead = dict(copy.deepcopy(PARAMS), model="hmm")
    gap_dead["evidence"]["gap_dead_alive"] = 0.02
    out["hmm gap-dead (E010c)"] = gap_dead
    return out


def bundle(size, seed):
    """A cached bundle, regenerated when it was written by another generator version."""
    out = os.path.join(EVAL_DIR, "{}-seed{}".format(size, seed))
    path = os.path.join(out, spec_module.SYNTHETIC_SPEC)
    version = None
    if os.path.isfile(path):
        with open(path) as handle:
            version = json.load(handle).get("generator_version")
    if version != make_synthetic.GENERATOR_VERSION:
        make_synthetic.generate(out, seed=seed, size=size)
    return out


def evaluate(size, seeds):
    """{variant: [(spellF1, macroF1, n_pred / n_gt possessions) per seed]}."""
    results = {name: [] for name in variants()}
    for seed in seeds:
        data_dir = bundle(size, seed)
        spec = spec_module.load(data_dir)
        tracking = atd.load(spec, data_dir)
        gt = pm.label_codes(pm.read_label_csv(
            os.path.join(data_dir, "synthetic_possession_gt.csv")))
        for name, params in variants().items():
            result = pm.evaluate(gt, pipeline.label(tracking, spec, params))
            results[name].append((result["spell_f1_mean"], result["macro_f1"],
                                  result["n_pred_spells"] / result["n_gt_spells"]))
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--size", default="full", choices=sorted(make_synthetic.LAYOUTS))
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    args = parser.parse_args()
    results = evaluate(args.size, args.seeds)
    print("{} synthetic, seeds {}".format(args.size, args.seeds))
    print("{:<12} {:>15} {:>15} {:>12}".format("variant", "spellF1", "macroF1", "poss ratio"))
    for name, rows in results.items():
        rows = np.array(rows)
        print("{:<12} {:>7.3f} +/-{:.3f} {:>7.3f} +/-{:.3f} {:>12.2f}".format(
            name, rows[:, 0].mean(), rows[:, 0].std(), rows[:, 1].mean(), rows[:, 1].std(),
            rows[:, 2].mean()))
