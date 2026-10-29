"""E015 in one run: measure the confident-alive cue's rates, then run E010d's criterion with them.

    python tools/gap_alive_run.py --gt evaluation/gt/BAR-ATH_possession_gt_public.csv
    python tools/gap_alive_run.py --recipe e010 --gt …      # E010d again: a check of this tool
    python tools/gap_alive_run.py --data-dir .synthetic/eval/full-seed0 \
        --gt .synthetic/eval/full-seed0/synthetic_possession_gt.csv

Step 1, the value rule (EXPERIMENTS.md, E010d/E015): on decoded frames with the ball untracked,
P(p_dead <= 0.3 | alive) and P(p_dead <= 0.3 | dead) against the pass-1 labels the classifier
trains on, rounded to 2 decimals. Step 2: the pipeline with those values switched on, scored by
the pre-set criterion: masked harness (5 seeds), full-match shares, public-truth D floor.

Everything printed is a rate, a share or a count over many frames. Safe to paste back.
"""

import argparse
import copy
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
from config import PARAMS  # noqa: E402
from possession import D, X, atd, dead_model, features, masking, pipeline, rules  # noqa: E402
from possession import submission, spec as spec_module  # noqa: E402
import masked_harness  # noqa: E402
import possession_metrics as pm  # noqa: E402

# Params
CONFIDENT = 0.3        # p_dead at or below this is a confident "alive" call (E010d)
RATE_LIMITS = (0.01, 0.99)  # a rate of 0 or 1 would make the cue an infinite veto (log 0)
TO_DEAD_MAX = 0.111    # E005c 0.141 - 0.03
RECALL_MIN = 0.264     # E005c 0.284 - 0.02
D_SHARE = (0.31, 0.45)
POSSESSIONS = (250, 400)
MEAN_S = (10.0, 20.0)
TRUTH_D_MIN = 0.643    # E005c 0.743 - 0.10


def measure(tracking, spec, params, recipe):
    """(P(sure alive | alive), P(sure alive | dead), n alive, n dead) on untracked decoded frames,
    against the pass-1 labels, with the classifier the pipeline would train on them."""
    pass1 = rules.finalize(pipeline.first_pass(tracking, spec, params), spec)
    p_dead, _ = dead_model.cross_half(tracking, spec, pass1, recipe=recipe)
    where = masking.decoded_mask(spec) & ~features.ball_visible(tracking) & ~np.isnan(p_dead)
    sure = p_dead <= CONFIDENT
    alive, dead = where & (pass1 != D) & (pass1 != X), where & (pass1 == D)
    return float(sure[alive].mean()), float(sure[dead].mean()), int(alive.sum()), int(dead.sum())


def switched_on(params, recipe, rate_alive, rate_dead):
    """params with the confident-alive cue on, at the measured rates rounded to 2 decimals and
    kept inside RATE_LIMITS."""
    def value(rate):
        return float(np.clip(round(rate, 2), *RATE_LIMITS))

    out = copy.deepcopy(params)
    out["evidence"].update(gap_alive_alive=value(rate_alive), gap_alive_dead=value(rate_dead),
                           gap_dead_recipe=recipe, gap_dead_teacher="pass1")
    return out


def score(tracking, spec, params, gt):
    """The criterion's aggregates for one config: masked harness (5 seeds), full-match shares and
    the public-truth D floor. gt: label codes of the public (or synthetic) ground truth."""
    _, rows, _ = masked_harness.run(tracking, spec, params, None, masked_harness.SEEDS)
    stats = masked_harness.summarise(rows)
    codes = pipeline.label(tracking, spec, params)
    summary = submission.summary(codes, spec)
    graded = summary["graded"]
    truth_dead = (gt == D) & ~features.ball_visible(tracking)
    return {
        "to_dead": stats["to_dead"], "change_recall": stats["change_recall"],
        "wrong_side": stats["wrong_side"][0], "agreement": stats["agreement"][0],
        "d_share": graded["D"] / sum(graded.values()),
        "n_possessions": summary["n_possessions"], "mean_s": summary["mean_possession_s"],
        "home_share": summary["home_share"],
        "truth_dead_called": float(np.mean(codes[truth_dead] == D)) if truth_dead.any()
        else float("nan"),
        "n_truth_dead": int(truth_dead.sum()),
    }


def print_score(s):
    print("  masked to_dead {:.3f} ± {:.3f}   change_recall {:.3f} ± {:.3f}".format(
        *s["to_dead"], *s["change_recall"]))
    print("  masked wrong_side {:.3f}   agreement {:.3f}".format(s["wrong_side"], s["agreement"]))
    print("  D share {:.3f}   possessions {}   mean {:.1f} s   home share {:.3f}".format(
        s["d_share"], s["n_possessions"], s["mean_s"], s["home_share"]))
    print("  truth D frames, ball untracked: called D {:.3f}  n {}".format(
        s["truth_dead_called"], s["n_truth_dead"]))


def criterion(s, to_dead_max, recall_min, d_share):
    """[(name, ok)]: the pre-set checks, with the bounds each experiment fixed."""
    return [
        ("1. to_dead <= {} and change_recall >= {}".format(to_dead_max, recall_min),
         s["to_dead"][0] <= to_dead_max and s["change_recall"][0] >= recall_min),
        ("2. D share {}-{}, possessions {}-{}, mean {}-{} s".format(*d_share, *POSSESSIONS,
                                                                    *MEAN_S),
         d_share[0] <= s["d_share"] <= d_share[1]
         and POSSESSIONS[0] <= s["n_possessions"] <= POSSESSIONS[1]
         and MEAN_S[0] <= s["mean_s"] <= MEAN_S[1]),
        ("3. truth untracked D called D >= {}".format(TRUTH_D_MIN),
         s["truth_dead_called"] >= TRUTH_D_MIN),
    ]


def report(checks):
    for name, ok in checks:
        print(check(name, ok))
    print("criterion: {}".format("KEEP" if all(ok for _, ok in checks) else "DROP"))


def check(name, ok):
    return "  {:<58} {}".format(name, "PASS" if ok else "FAIL")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir")
    parser.add_argument("--gt", required=True, help="public (or synthetic) ground truth")
    parser.add_argument("--recipe", choices=dead_model.RECIPES, default="lang")
    args = parser.parse_args()

    spec = spec_module.load(args.data_dir)
    tracking = atd.load(spec, args.data_dir)
    rate_alive, rate_dead, n_alive, n_dead = measure(tracking, spec, PARAMS, args.recipe)
    params = switched_on(PARAMS, args.recipe, rate_alive, rate_dead)
    ev = params["evidence"]
    print("gap confident-alive run, recipe {}".format(args.recipe))
    print("1. value rule (untracked decoded frames vs pass-1 labels)")
    print("  P(sure alive | alive) {:.3f}  n {}   -> gap_alive_alive {}".format(
        rate_alive, n_alive, ev["gap_alive_alive"]))
    print("  P(sure alive | dead)  {:.3f}  n {}   -> gap_alive_dead {}".format(
        rate_dead, n_dead, ev["gap_alive_dead"]))

    gt = pm.label_codes(pm.read_label_csv(args.gt))
    print("2. criterion (E005c: to_dead 0.141, change_recall 0.284, D 35.4%, 308 poss, 13.3 s)")
    s = score(tracking, spec, params, gt)
    print_score(s)
    report(criterion(s, TO_DEAD_MAX, RECALL_MIN, D_SHARE))
    return 0


if __name__ == "__main__":
    sys.exit(main())
