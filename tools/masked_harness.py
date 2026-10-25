"""Masked-ball harness (E005a): what the pipeline loses when the ball goes missing in play.

    python tools/masked_harness.py                                    # real data -- a human runs it
    python tools/masked_harness.py --gt evaluation/gt/BAR-ATH_possession_gt_public.csv
    python tools/masked_harness.py --data-dir .synthetic/eval/full-seed0 \
        --gt .synthetic/eval/full-seed0/synthetic_possession_gt.csv

Runs the pipeline (config.py) once on the bundle as it is -- the reference -- then once per seed
with the ball hidden on seen frames in gap-shaped stretches (possession/masking.py), and compares.
Table 1 scores each masked run against the reference; with --gt, table 2 scores the unmasked and
masked runs against that ground truth, around the same hidden frames.

Everything printed is an aggregate over thousands of frames: rates, counts, means. Safe to paste
back. Read it as "how much worse without the ball", never as "how good" (EXPERIMENTS.md, E005a).
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from config import PARAMS  # noqa: E402
from possession import atd, features, masking, pipeline, spec as spec_module  # noqa: E402
import possession_metrics as pm  # noqa: E402

# Params
SEEDS = (0, 1, 2, 3, 4)
RATES = ("change_recall", "mean_iou", "agreement", "to_dead", "wrong_side", "spell_f1")
COUNTS = ("n_possessions", "n_changes", "n_touched", "n_hidden")
COLUMNS = RATES + COUNTS


def run(tracking, spec, params, gt, seeds, tags=None):
    """(context, table 1 rows, table 2 rows or None); rows are dicts of COLUMNS per run.

    tags: Smart Tagging events, needed when the phase tag cue (E012) is on (pipeline.tags_for).
    """
    tol = spec.fps
    reference = pipeline.label(tracking, spec, params, tags)
    visible = features.ball_visible(tracking)
    decoded = masking.decoded_mask(spec)
    lengths = masking.gap_lengths(visible, reference, decoded)
    vs_reference, vs_truth = [], []
    for seed in seeds:
        masked, hidden = masking.mask_ball(tracking, reference, spec, seed,
                                           params["evidence"]["mid_m"])
        codes = pipeline.label(masked, spec, params, tags)
        vs_reference.append(masking.compare(reference, codes, hidden, spec, tol))
        if gt is not None:
            vs_truth.append((masking.compare(gt, reference, hidden, spec, tol),
                             masking.compare(gt, codes, hidden, spec, tol)))
    alive = decoded & (reference <= 1)
    context = {
        "reference possessions": pm.possession_stats(reference, pm.valid_runs(reference))[
            "n_spells"],
        "real gaps in play": len(lengths),
        "their length p50 / p90 (s)": "{:.1f} / {:.1f}".format(
            *(np.percentile(lengths, (50, 90)) / spec.fps)),
        "untracked share of alive frames": "{:.3f}".format((~visible[alive]).mean()),
        "hidden share of alive frames": "{:.3f}".format(
            np.mean([row["n_hidden"] for row in vs_reference]) / alive.sum()),
    }
    return context, vs_reference, (vs_truth or None)


def summarise(rows):
    """{column: (mean, std)} over seeds."""
    return {key: (float(np.nanmean([row[key] for row in rows])),
                  float(np.nanstd([row[key] for row in rows]))) for key in COLUMNS}


def print_table(title, named_rows):
    """One line per column, one column per named run: narrow enough for any terminal."""
    print(title)
    print("{:<16}".format("") + "".join("{:>17}".format(name) for name, _ in named_rows))
    for key in COLUMNS:
        pattern = "{:.3f} ± {:.3f}" if key in RATES else "{:.0f} ± {:.0f}"
        print("{:<16}".format(key) + "".join(
            "{:>17}".format(pattern.format(*stats[key])) for _, stats in named_rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir")
    parser.add_argument("--gt", help="ground-truth CSV for table 2 (public window or synthetic)")
    args = parser.parse_args()

    spec = spec_module.load(args.data_dir)
    tracking = atd.load(spec, args.data_dir)
    gt = pm.label_codes(pm.read_label_csv(args.gt)) if args.gt else None
    context, vs_reference, vs_truth = run(tracking, spec, PARAMS, gt, SEEDS,
                                          pipeline.tags_for(PARAMS, args.data_dir))

    print("model: {}, seeds: {}".format(PARAMS["model"], list(SEEDS)))
    for key, value in context.items():
        print("  {:<34} {}".format(key, value))
    print()
    print_table("1. masked run vs unmasked reference (mean ± std over seeds)",
                [("masked", summarise(vs_reference))])
    if vs_truth:
        print()
        print_table("2. vs ground truth around the hidden frames (graded truth only)",
                    [("unmasked", summarise([u for u, _ in vs_truth])),
                     ("masked", summarise([m for _, m in vs_truth]))])
    return 0


if __name__ == "__main__":
    sys.exit(main())
