"""Dead/alive-from-movement classifier check (E010a gate): aggregates only.

    python tools/dead_model_check.py --gt evaluation/gt/BAR-ATH_possession_gt_public.csv
    python tools/dead_model_check.py --data-dir .synthetic/eval/full-seed0 \
        --gt .synthetic/eval/full-seed0/synthetic_possession_gt.csv

    python tools/dead_model_check.py --recipe lang --gt …   # E013a: Lang et al.'s recipe

Every decoded frame is predicted by a model fitted on the other half (possession/dead_model.py).
Rows (each: balanced accuracy, P(says dead | alive), P(says dead | dead), n):

  (1) E005a's hidden frames (5 seeds pooled; all reference-alive): the masked decoder's dead rate
      there vs the classifier's -- gate 1
  (2) ball-seen frames vs the reference labels -- gate 2; these rates feed E010b's cue
  (3) ball-untracked frames vs the reference labels
  (4) --gt only: graded truth frames with the ball untracked, the classifier and the decoded
      reference vs the truth -- gate 3; and, for reading only (E012's lesson), on the frames
      where the decoded reference is wrong dead/alive, the share the classifier gets right

The slow-players cue alone is shown beside the classifier as the thing it would replace.
Everything printed is a rate or a count over many frames. Safe to paste back.
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from config import PARAMS  # noqa: E402
from possession import D, X, atd, dead_model, features, masking, pipeline  # noqa: E402
from possession import spec as spec_module  # noqa: E402
import possession_metrics as pm  # noqa: E402

# Params
SEEDS = (0, 1, 2, 3, 4)
GATE_2 = 0.80          # balanced accuracy on ball-seen frames
GATE_3_MARGIN = 0.02   # classifier may trail the decoded labels by at most this on truth gaps


def rates(says_dead, is_dead, where):
    """(balanced accuracy, P(dead | alive), P(dead | dead), n) over `where`."""
    where = where & ~np.isnan(says_dead.astype(float))
    says, truth = says_dead[where].astype(bool), is_dead[where]
    fa = float(says[~truth].mean()) if (~truth).any() else float("nan")
    hit = float(says[truth].mean()) if truth.any() else float("nan")
    balanced = np.nanmean([1 - fa, hit]) if not (np.isnan(fa) and np.isnan(hit)) else np.nan
    return float(balanced), fa, hit, int(where.sum())


def line(name, values):
    print("  {:<44} bal {:.3f}  dead|alive {:.3f}  dead|dead {:.3f}  n {}".format(name, *values))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir")
    parser.add_argument("--gt")
    parser.add_argument("--recipe", choices=dead_model.RECIPES,
                        default=PARAMS["evidence"]["gap_dead_recipe"])
    args = parser.parse_args()

    spec = spec_module.load(args.data_dir)
    tracking = atd.load(spec, args.data_dir)
    ev = PARAMS["evidence"]
    reference = pipeline.label(tracking, spec, PARAMS)
    p_dead, _ = dead_model.cross_half(tracking, spec, reference, recipe=args.recipe)
    says = np.where(np.isnan(p_dead), np.nan, p_dead >= 0.5)
    walking = features.players_median_speed(tracking, spec.fps)
    slow = np.where(np.isnan(walking), np.nan, walking < ev["slow_speed"])
    decoded = masking.decoded_mask(spec)
    visible = features.ball_visible(tracking)
    ref_dead = reference == D
    has = ~np.isnan(p_dead)

    print("dead-model check (E010a gate), model {}, recipe {}".format(PARAMS["model"],
                                                                       args.recipe))
    print("(1) hidden frames, reference alive (5 seeds pooled)")
    masked_dead, model_dead, n = [], [], 0
    for seed in SEEDS:
        masked, hidden = masking.mask_ball(tracking, reference, spec, seed, ev["mid_m"])
        codes = pipeline.label(masked, spec, PARAMS)
        where = hidden & (reference <= 1) & has
        masked_dead.append(float(np.mean(codes[where] == D)))
        model_dead.append(float(np.mean(p_dead[where] >= 0.5)))
        n += int(where.sum())
    gate_1 = np.mean(model_dead) <= 0.5 * np.mean(masked_dead)
    print("  masked decoder to_dead {:.3f}   classifier dead rate {:.3f}   slow cue n/a   n {}"
          .format(np.mean(masked_dead), np.mean(model_dead), n))
    print("  gate 1 (classifier <= half the decoder): {}".format("PASS" if gate_1 else "FAIL"))

    print("(2) ball-seen frames vs reference")
    seen = rates(says, ref_dead, decoded & visible & has)
    line("classifier", seen)
    line("slow cue", rates(slow, ref_dead, decoded & visible & ~np.isnan(slow)))
    gate_2 = seen[0] >= GATE_2
    print("  gate 2 (balanced >= {:.2f}): {}".format(GATE_2, "PASS" if gate_2 else "FAIL"))

    print("(3) ball-untracked frames vs reference")
    line("classifier", rates(says, ref_dead, decoded & ~visible & has))
    line("slow cue", rates(slow, ref_dead, decoded & ~visible & ~np.isnan(slow)))

    gate_3 = True
    if args.gt:
        gt = pm.label_codes(pm.read_label_csv(args.gt))
        truth_dead = gt == D
        where = (gt != X) & ~visible & has
        print("(4) truth graded frames, ball untracked, vs truth")
        model = rates(says, truth_dead, where)
        decoded_rates = rates(ref_dead.astype(float), truth_dead, where)
        line("classifier", model)
        line("decoded reference (E005c)", decoded_rates)
        line("slow cue", rates(slow, truth_dead, where & ~np.isnan(slow)))
        wrong = where & (ref_dead != truth_dead)
        right = np.mean(says[wrong].astype(bool) == truth_dead[wrong]) if wrong.any() else np.nan
        print("  not a gate: classifier right where decoded is wrong {:.3f}  n {}".format(
            right, int(wrong.sum())))
        gate_3 = model[0] >= decoded_rates[0] - GATE_3_MARGIN
        print("  gate 3 (classifier >= decoded - {:.2f}): {}".format(
            GATE_3_MARGIN, "PASS" if gate_3 else "FAIL"))
    print("E010a gate: {}".format("PASS" if gate_1 and gate_2 and gate_3 else "FAIL"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
