"""Smart Tagging set pieces against the current labels (E007a): aggregates only.

    python tools/inspect_restarts.py --gt evaluation/gt/BAR-ATH_possession_gt_public.csv
    python tools/inspect_restarts.py --data-dir .synthetic/eval/full-seed0 \
        --gt .synthetic/eval/full-seed0/synthetic_possession_gt.csv

A set piece is a restart: the ball was dead just before it, and the tag's Team (E002: the
restarting side for free and goal kicks) has the ball just after. Per set-piece Type, over the
tags whose start falls in a decoded half, against the pipeline's labels (config.py):

  dead_before  share of tags with any D in the WINDOW_S before the tag start
  dead_near    share of tags with any D within WINDOW_S either side of the start
  alive_after  share of tags whose first alive side after the start is known
  own_after    of those, share where that side is the tag's Team (N/A tags: not counted)
  start_to_alive_s  median seconds from tag start to the first alive frame after it

A restart with no D near it in our labels is a stoppage we miss, so two possessions merge or a
side change goes through D unseen. With --gt the same over tags inside the graded truth.
Everything printed is a share, a median or a count over tags. Safe to paste back.
"""

import argparse
import os
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from config import PARAMS  # noqa: E402
from possession import D, X, atd, masking, pipeline, tagging  # noqa: E402
from possession import spec as spec_module  # noqa: E402
import possession_metrics as pm  # noqa: E402

# Params
WINDOW_S = 3.0
AFTER_S = 10.0     # look this far after the start for the first alive frame


def rows(events, codes, where, fps):
    """{type: dict of the columns above} over set-piece tags starting on `where` frames."""
    reach, after = int(WINDOW_S * fps), int(AFTER_S * fps)
    acc = defaultdict(lambda: defaultdict(list))
    for event in events:
        if event["code"] != "SET PIECES":
            continue
        start = event["start_frame"]
        if not (0 <= start < len(codes)) or not where[start]:
            continue
        key = event["type"] or "?"
        before = codes[max(0, start - reach):start]
        near = codes[max(0, start - reach):start + reach + 1]
        ahead = codes[start:start + after]
        alive = np.flatnonzero(ahead <= 1)
        row = acc[key]
        row["dead_before"].append(bool((before == D).any()))
        row["dead_near"].append(bool((near == D).any()))
        row["alive_after"].append(len(alive) > 0)
        if len(alive):
            row["start_to_alive_s"].append(alive[0] / fps)
            if event["side"] is not None:
                row["own_after"].append(bool(ahead[alive[0]] == event["side"]))
    return acc


def print_rows(title, acc):
    print(title)
    print("  {:<22} {:>5} {:>12} {:>10} {:>12} {:>10} {:>8}".format(
        "type", "tags", "dead_before", "dead_near", "alive_after", "own_after", "to_alive"))
    total = defaultdict(list)
    for key in sorted(acc):
        row = acc[key]
        for name, values in row.items():
            total[name] += values
        _print_row(key, row)
    _print_row("ALL", total)


def _print_row(name, row):
    def share(values):
        return "{:.2f}".format(np.mean(values)) if values else "-"
    own = row.get("own_after", [])
    print("  {:<22} {:>5} {:>12} {:>10} {:>12} {:>10} {:>8}".format(
        name[:22], len(row["dead_near"]), share(row["dead_before"]), share(row["dead_near"]),
        share(row["alive_after"]), "{} ({})".format(share(own), len(own)),
        "{:.1f}".format(np.median(row["start_to_alive_s"])) if row.get("start_to_alive_s")
        else "-"))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir")
    parser.add_argument("--gt")
    args = parser.parse_args()

    spec = spec_module.load(args.data_dir)
    tracking = atd.load(spec, args.data_dir)
    events = tagging.load(args.data_dir)
    codes = pipeline.label(tracking, spec, PARAMS)
    print_rows("1. pipeline labels, full match (window {:.0f} s)".format(WINDOW_S),
               rows(events, codes, masking.decoded_mask(spec), spec.fps))
    if args.gt:
        gt = pm.label_codes(pm.read_label_csv(args.gt))
        graded = gt != X
        print_rows("2. truth, tags starting on graded truth frames", rows(events, gt, graded,
                                                                          spec.fps))
        print_rows("3. pipeline labels, the same tags", rows(events, codes, graded, spec.fps))
    return 0


if __name__ == "__main__":
    sys.exit(main())
