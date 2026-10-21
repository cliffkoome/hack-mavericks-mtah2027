"""Team-from-movement classifier check (E005b): half-to-half accuracy, aggregates only.

    python tools/team_model_check.py --gt evaluation/gt/BAR-ATH_possession_gt_public.csv
    python tools/team_model_check.py --data-dir .synthetic/eval/full-seed0 \
        --gt .synthetic/eval/full-seed0/synthetic_possession_gt.csv

Every decoded frame is predicted by a model fitted on the other half (possession/team_model.py),
then scored against the current pipeline's labels (the reference, config.py) and, with --gt, the
ground truth. The classifier never sees the ball, so hiding it changes which frames we score, not
the prediction. Rows:

  (a) training-type frames: reference alive, ball seen, a player within near_m
  (b) every reference-alive frame -- the gate's set: gap frames are not close-control frames
  (c) E005a's hidden frames (5 seeds pooled), by the length of the hidden stretch
  (d) --gt only: truth-alive frames with the ball untracked -- the gate's second set

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
from possession import H, A, atd, features, masking, pipeline, team_model  # noqa: E402
from possession import spec as spec_module  # noqa: E402
import possession_metrics as pm  # noqa: E402

# Params
SEEDS = (0, 1, 2, 3, 4)
GAP_BANDS_S = ((0, 5), (5, 15), (15, None))
CONFIDENT = 0.7          # p >= this or <= 1 - this counts as a confident call
GATE_B, GATE_D = 0.60, 0.5


def score(p_home, label_home, where):
    """(accuracy, n, confident share, confident accuracy) of p_home >= 0.5 against labels."""
    where = where & ~np.isnan(p_home)
    n = int(where.sum())
    if not n:
        return float("nan"), 0, float("nan"), float("nan")
    p, truth = p_home[where], label_home[where]
    right = (p >= 0.5) == truth
    sure = (p >= CONFIDENT) | (p <= 1 - CONFIDENT)
    return (float(right.mean()), n, float(sure.mean()),
            float(right[sure].mean()) if sure.any() else float("nan"))


def rows(tracking, spec, params, gt):
    near_m = params["evidence"]["near_m"]
    reference = pipeline.label(tracking, spec, params)
    p_home, folds = team_model.cross_half(tracking, spec, reference, near_m)
    decoded = masking.decoded_mask(spec)
    alive = decoded & ((reference == H) | (reference == A))
    ref_home = reference == H
    visible = features.ball_visible(tracking)
    team, distance = features.nearest_player(tracking)
    close = alive & visible & (distance < near_m) & (team >= 0)

    out = [("(a) training-type frames", score(p_home, ref_home, close)),
           ("(b) all reference-alive", score(p_home, ref_home, alive))]
    hidden_by_band = {band: np.zeros(spec.n_frames, dtype=bool) for band in GAP_BANDS_S}
    hits = {band: [0, 0, 0, 0] for band in GAP_BANDS_S}
    for seed in SEEDS:
        _, hidden = masking.mask_ball(tracking, reference, spec, seed, params["evidence"]["mid_m"])
        length = features.run_length(hidden) / spec.fps
        for low, high in GAP_BANDS_S:
            band = hidden & alive & (length >= low) & (length < (high or np.inf))
            accuracy, n, sure, sure_acc = score(p_home, ref_home, band)
            if n:
                hits[(low, high)][0] += accuracy * n
                hits[(low, high)][1] += n
                hits[(low, high)][2] += sure * n
                hits[(low, high)][3] += (sure_acc * sure * n) if sure else 0.0
    for (low, high), (right, n, sure, sure_right) in hits.items():
        name = "(c) hidden, {}{} s".format(low, "+" if high is None else "-{}".format(high))
        out.append((name, (right / n if n else float("nan"), n,
                           sure / n if n else float("nan"),
                           sure_right / sure if sure else float("nan"))))
    if gt is not None:
        truth_alive = (gt == H) | (gt == A)
        out.append(("(d) truth-alive, untracked", score(p_home, gt == H, truth_alive & ~visible)))
        out.append(("    truth-alive, all", score(p_home, gt == H, truth_alive & decoded)))
    training = sum(len(train) for _, train in folds)
    return out, training


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir")
    parser.add_argument("--gt", help="ground-truth CSV for row (d) (public window or synthetic)")
    args = parser.parse_args()

    spec = spec_module.load(args.data_dir)
    tracking = atd.load(spec, args.data_dir)
    gt = pm.label_codes(pm.read_label_csv(args.gt)) if args.gt else None
    table, training = rows(tracking, spec, PARAMS, gt)

    print("model: {}, half-to-half, {} training frames over both folds".format(
        PARAMS["model"], training))
    print("{:<28} {:>9} {:>8} {:>10} {:>10}".format("frames", "accuracy", "n", "confident",
                                                   "conf. acc"))
    results = dict(table)
    for name, (accuracy, n, sure, sure_acc) in table:
        print("{:<28} {:>9.3f} {:>8} {:>10.3f} {:>10.3f}".format(name, accuracy, n, sure,
                                                                  sure_acc))
    b = results["(b) all reference-alive"][0]
    d = results.get("(d) truth-alive, untracked", (float("nan"),))[0]
    passed = b >= GATE_B and (np.isnan(d) or d >= GATE_D)
    print("\ngate: (b) {:.3f} >= {:.2f} and (d) {} >= {:.2f} -> {}".format(
        b, GATE_B, "n/a" if np.isnan(d) else "{:.3f}".format(d), GATE_D,
        "PASS" if passed else "FAIL"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
