"""Prove our reader is right on the real data: rebuild the published baselines with possession/ and
compare them frame for frame with the committed evaluation/results/baseline_*.csv.

Those four files were written by starter/baseline.py from the real ATD. If our atd.load ->
features.nearest_team -> smoothing.majority_vote reproduces all four exactly, then our ATD parsing,
frame alignment (row = frame_id - 1), slot-team mapping and distance arithmetic are all right.

    python tools/verify_baseline.py              # real data in data/ -- a human runs this
    python tools/verify_baseline.py --data-dir .synthetic/small-seed0 --results <dir>

Output is safe to share: one line per baseline, IDENTICAL or the number of differing frames.
No labels, frames or tracking values are printed.
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from possession import atd, features, smoothing, spec as spec_module  # noqa: E402
import possession_metrics as pm  # noqa: E402

RESULTS_DIR = os.path.join(ROOT, "evaluation", "results")
# The published variants and their half-widths, as in starter/baseline.py.
VARIANTS = (("baseline_raw", 0), ("baseline_smooth25", 25), ("baseline_smooth50", 50),
            ("baseline_smooth125", 125))


def compare(data_dir=None, results_dir=RESULTS_DIR):
    """{name: number of frames where ours differs from the committed file}."""
    spec = spec_module.load(data_dir)
    team = features.nearest_team(atd.load(spec, data_dir))
    out = {}
    for name, window in VARIANTS:
        published = pm.label_codes(pm.read_label_csv(os.path.join(results_dir, name + ".csv")))
        ours = smoothing.majority_vote(team, window)
        out[name] = int(np.sum(ours != published)) if len(ours) == len(published) else -1
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir")
    parser.add_argument("--results", default=RESULTS_DIR)
    args = parser.parse_args()
    differences = compare(args.data_dir, args.results)
    for name, count in differences.items():
        print("{:<20} {}".format(name, "IDENTICAL" if count == 0 else
                                 "length mismatch" if count < 0 else
                                 "{} frames differ".format(count)))
    sys.exit(0 if all(count == 0 for count in differences.values()) else 1)
