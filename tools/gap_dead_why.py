"""Why the masked decoder calls hidden in-play frames dead (E010 follow-up): aggregates only.

    python tools/gap_dead_why.py
    python tools/gap_dead_why.py --data-dir .synthetic/eval/full-seed0

Masks the ball as E005a does (5 seeds), then splits the hidden, reference-alive frames by what
the masked decoder calls them (D or alive) and reports, for each group, the share where the
slow-players cue fires and how long the untracked run around the frame is. If the D calls sit
mostly in long runs with the players moving, the missing-ball cue is what makes them.
Everything printed is a share or a count over many frames. Safe to paste back.
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from config import PARAMS  # noqa: E402
from possession import D, atd, features, masking, pipeline  # noqa: E402
from possession import spec as spec_module  # noqa: E402

# Params
SEEDS = (0, 1, 2, 3, 4)
RUN_BANDS_S = ((0, 3), (3, 10), (10, None))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir")
    args = parser.parse_args()

    spec = spec_module.load(args.data_dir)
    tracking = atd.load(spec, args.data_dir)
    ev = PARAMS["evidence"]
    reference = pipeline.label(tracking, spec, PARAMS)
    walking = features.players_median_speed(tracking, spec.fps)
    slow = walking < ev["slow_speed"]
    totals = {"D": np.zeros(2 + len(RUN_BANDS_S)), "alive": np.zeros(2 + len(RUN_BANDS_S))}
    for seed in SEEDS:
        masked, hidden = masking.mask_ball(tracking, reference, spec, seed, ev["mid_m"])
        codes = pipeline.label(masked, spec, PARAMS)
        run = features.run_length(~features.ball_visible(masked)) / spec.fps
        base = hidden & (reference <= 1)
        for name, group in (("D", base & (codes == D)), ("alive", base & (codes <= 1))):
            bands = [np.sum(group & (run >= lo) & ((run < hi) if hi else True))
                     for lo, hi in RUN_BANDS_S]
            totals[name] += [group.sum(), np.sum(group & slow)] + bands
    print("hidden, reference-alive frames (5 seeds pooled), by the masked decoder's call")
    print("  {:<8} {:>9} {:>10}".format("call", "frames", "slow fires") + "".join(
        "{:>12}".format("run {}-{} s".format(lo, hi) if hi else "run > {} s".format(lo))
        for lo, hi in RUN_BANDS_S))
    for name, row in totals.items():
        n = max(row[0], 1)
        print("  {:<8} {:>9.0f} {:>10.3f}".format(name, row[0], row[1] / n) + "".join(
            "{:>12.3f}".format(v / n) for v in row[2:]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
