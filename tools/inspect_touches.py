"""Touch detection and the last-touch side (E006a): label-free rates, plus a calibration check.

    python tools/inspect_touches.py --gt evaluation/gt/BAR-ATH_possession_gt_public.csv
    python tools/inspect_touches.py --data-dir .synthetic/eval/full-seed0 \
        --gt .synthetic/eval/full-seed0/synthetic_possession_gt.csv

1. full match, no labels: the touch threshold tau (free-flight rule, possession/touches.py),
   touches per minute of play, duel share, coverage of the last-touch side on seen alive frames
   (alive and stoppages from the current pipeline, config.py), agreement with the nearest player.
2. --gt: on truth-alive seen frames where the last-touch side is known, accuracy of the
   last-touch side and of the nearest-player side on the same frames, by nearest distance.

Then the pre-set gate (EXPERIMENTS.md, E006). Every number is a rate or count over many frames:
safe to paste back.
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from config import PARAMS  # noqa: E402
from possession import H, A, D, atd, features, masking, pipeline, touches  # noqa: E402
from possession import spec as spec_module  # noqa: E402
import possession_metrics as pm  # noqa: E402

# Params
GATE_MARGIN, GATE_COVERAGE = 0.03, 0.5
BANDS = (("< 2 m", 0.0, 2.0), ("2-5 m", 2.0, 5.0), ("> 5 m", 5.0, np.inf))


def measure(tracking, spec, params, gt):
    reference = pipeline.label(tracking, spec, params)
    decoded = masking.decoded_mask(spec)
    visible = features.ball_visible(tracking)
    team, distance = features.nearest_player(tracking)
    alive = decoded & ((reference == H) | (reference == A))

    dv = touches.velocity_change(tracking.ball, spec.fps)
    tau = touches.free_flight_threshold(dv, distance, decoded & visible)
    frames, sides = touches.detect(tracking, dv, tau)
    side = touches.last_touch(spec.n_frames, frames, sides, visible, reference == D, spec.fps)

    in_play = alive[frames]
    minutes = alive.sum() / spec.fps / 60.0
    seen_alive = alive & visible
    known = seen_alive & (side >= 0)
    match = {
        "tau (m/s)": tau,
        "touches per minute of play": in_play.sum() / minutes,
        "  attributed per minute": np.sum(in_play & (sides >= 0)) / minutes,
        "  duel share": float(np.mean(sides[in_play] < 0)) if in_play.any() else float("nan"),
        "coverage of seen alive frames": known.sum() / max(seen_alive.sum(), 1),
        "agreement with nearest player": float(np.mean(side[known] == team[known])),
        "seen alive frames": int(seen_alive.sum()),
    }
    calibration = None
    if gt is not None:
        truth_alive = (gt == H) | (gt == A)
        base = truth_alive & visible & (side >= 0) & (team >= 0)
        calibration = []
        for name, low, high in (("all",) + (0.0, np.inf),) + BANDS:
            where = base & (distance >= low) & (distance < high)
            n = int(where.sum())
            calibration.append((name, n,
                                float(np.mean(side[where] == gt[where])) if n else float("nan"),
                                float(np.mean(team[where] == gt[where])) if n else float("nan")))
    return match, calibration


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir")
    parser.add_argument("--gt", help="ground-truth CSV for table 2 (public window or synthetic)")
    args = parser.parse_args()

    spec = spec_module.load(args.data_dir)
    tracking = atd.load(spec, args.data_dir)
    gt = pm.label_codes(pm.read_label_csv(args.gt)) if args.gt else None
    match, calibration = measure(tracking, spec, PARAMS, gt)

    print("1. full match, label-free (alive/stoppage from the current pipeline)")
    for key, value in match.items():
        print("  {:<34} {}".format(key, value if isinstance(value, int) else "{:.3f}".format(value)))
    if calibration is None:
        return 0
    print("\n2. truth-alive seen frames with a last-touch side: accuracy against the truth")
    print("  {:<10} {:>7} {:>11} {:>15}".format("nearest", "n", "last touch", "nearest player"))
    for name, n, last, nearest in calibration:
        print("  {:<10} {:>7} {:>11.3f} {:>15.3f}".format(name, n, last, nearest))
    _, _, last, nearest = calibration[0]
    coverage = match["coverage of seen alive frames"]
    passed = last - nearest >= GATE_MARGIN and coverage >= GATE_COVERAGE
    print("\ngate: last touch {:.3f} - nearest {:.3f} = {:+.3f} >= {:.2f}, coverage {:.3f} >= {:.2f}"
          " -> {}".format(last, nearest, last - nearest, GATE_MARGIN, coverage, GATE_COVERAGE,
                          "PASS" if passed else "FAIL"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
