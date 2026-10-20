"""Check the decoder's assumptions against the public ground truth: aggregates only.

    python tools/inspect_cues.py                                  # real data -- a human runs it
    python tools/inspect_cues.py --data-dir .synthetic/small-seed0 \
        --gt .synthetic/small-seed0/synthetic_possession_gt.csv

Three tables, every cell a rate or a count over many frames -- nothing frame-level, no positions,
safe to paste back:

  1. cues      how often each evidence cue fires when the truth is alive vs dead, next to the
               value config.py assumes. This is calibration of stated probabilities, not a score:
               a cue whose real rates differ from the assumption by a lot is a bug in the
               assumption, whichever way it moves the public spellF1.
  2. team      how often the nearest player's side is the possessing side, by distance band.
  3. tagging   per Smart Tagging code (and set-piece Type), over tags overlapping the graded
               frames of the GT: what the truth is during the tag, split into the tag's own side,
               the other side, and dead ball. Answers "does Team mean attacker or defender here?"

17 possessions is a small sample: read the n column before any rate.
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
from possession import H, A, D, X, atd, features, spec as spec_module, tagging  # noqa: E402
import possession_metrics as pm  # noqa: E402

PUBLIC_GT = os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")


def rate(flag, where):
    n = int(where.sum())
    return (float(flag[where].mean()) if n else float("nan")), n


def cue_rows(tracking, spec, gt, params):
    graded = gt != X
    alive, dead = graded & (gt <= A), graded & (gt == D)
    visible = features.ball_visible(tracking)
    speed = features.ball_speed(tracking, spec.fps, params["speed_lag"])
    outside = features.ball_outside(tracking, spec.pitch, params["outside_margin_m"])
    still = speed < params["still_speed"]
    known = ~np.isnan(speed)
    # Diagnostics with no config value yet: does lost ball come in long runs in both states
    # (what missing_run_cap assumes), and does a still ball show up over a full second?
    long_gap = features.run_length(~visible) > spec.fps
    slow = features.ball_speed(tracking, spec.fps, spec.fps)
    rows = []
    for name, flag, where, assumed in (
            ("ball not tracked", ~visible, np.ones_like(visible), "missing"),
            ("outside lines | tracked", outside, visible, "outside"),
            ("still | speed known", still, known, "still"),
            ("untracked, run > 1 s", long_gap, np.ones_like(visible), None),
            ("still over 1 s (< 0.5 m/s)", slow < 0.5, ~np.isnan(slow), None)):
        (r_alive, n_alive), (r_dead, n_dead) = rate(flag, alive & where), rate(flag, dead & where)
        rows.append((name, r_alive, params[assumed + "_alive"] if assumed else None, n_alive,
                     r_dead, params[assumed + "_dead"] if assumed else None, n_dead))
    return rows


def team_rows(tracking, gt, params):
    team, distance = features.nearest_player(tracking)
    alive = (gt == H) | (gt == A)
    bands = (("< near_m", distance < params["near_m"], params["team_acc_near"]),
             ("near_m..mid_m", (distance >= params["near_m"]) & (distance < params["mid_m"]),
              params["team_acc_mid"]),
             (">= mid_m", (distance >= params["mid_m"]) & np.isfinite(distance), None),
             ("no ball/players", team < 0, None))
    rows = []
    total = int(alive.sum())
    for name, band, assumed in bands:
        where = alive & band
        accuracy, n = rate(team == gt, where)
        rows.append((name, n / total if total else float("nan"), accuracy, assumed, n))
    return rows


def tagging_rows(events, gt):
    """{(code, type): [n tags, frames, share own side, share other side, share D]}."""
    graded = gt != X
    sums = defaultdict(lambda: np.zeros(5))
    for event in events:
        start, stop = max(event["start_frame"], 0), min(event["end_frame"], len(gt))
        window = gt[start:stop][graded[start:stop]]
        if not len(window):
            continue
        key = (event["code"], event["type"] if event["code"] == "SET PIECES" else None)
        side = event["side"]
        own = np.sum(window == side) if side is not None else 0
        other = np.sum(window == (1 - side)) if side is not None else np.sum(window <= A)
        sums[key] += [1, len(window), own, other, np.sum(window == D)]
    return {key: [int(v[0]), int(v[1])] + list(v[2:] / v[1]) for key, v in sorted(sums.items())}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir")
    parser.add_argument("--gt", default=PUBLIC_GT)
    args = parser.parse_args()

    spec = spec_module.load(args.data_dir)
    tracking = atd.load(spec, args.data_dir)
    gt = pm.label_codes(pm.read_label_csv(args.gt))
    params = PARAMS["evidence"]

    print("1. cues: rate when truth is alive / dead (assumed in config.py)")
    print("{:<26} {:>8} {:>8} {:>7}   {:>8} {:>8} {:>7}".format(
        "cue", "alive", "assumed", "n", "dead", "assumed", "n"))
    for name, ra, aa, na, rd, ad, nd in cue_rows(tracking, spec, gt, params):
        print("{:<26} {:>8.3f} {:>8} {:>7}   {:>8.3f} {:>8} {:>7}".format(
            name, ra, "-" if aa is None else "{:.3f}".format(aa), na,
            rd, "-" if ad is None else "{:.3f}".format(ad), nd))

    print("\n2. team: nearest player's side vs possessing side, alive frames")
    print("{:<18} {:>10} {:>9} {:>8} {:>7}".format("distance band", "share", "accuracy",
                                                   "assumed", "n"))
    for name, share, accuracy, assumed, n in team_rows(tracking, gt, params):
        print("{:<18} {:>10.3f} {:>9.3f} {:>8} {:>7}".format(
            name, share, accuracy, "-" if assumed is None else "{:.3f}".format(assumed), n))

    print("\n3. tagging: truth during tags overlapping graded GT frames")
    print("   own = tag's Team side, other = the other side (for N/A: any side)")
    print("{:<30} {:<24} {:>5} {:>7} {:>6} {:>6} {:>6}".format(
        "code", "type", "tags", "frames", "own", "other", "D"))
    for (code, kind), (n, frames, own, other, dead) in tagging_rows(
            tagging.load(args.data_dir), gt).items():
        print("{:<30} {:<24} {:>5} {:>7} {:>6.2f} {:>6.2f} {:>6.2f}".format(
            code[:30], (kind or "")[:24], n, frames, own, other, dead))


if __name__ == "__main__":
    sys.exit(main())
