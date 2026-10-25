"""E012 gate: is a Smart Tagging phase tag a useful team cue where the ball is missing?

    python tools/tag_gate.py                         # real data -- a human runs it
    python tools/tag_gate.py --data-dir .synthetic/eval/full-seed0

Uses the masked-ball harness (E005a): the unmasked E005c run is the reference, the ball is hidden
in gap-shaped stretches (seeds 0-4), and on the hidden frames alive in the reference each phase
group's side (tagging.phase_cue, that group alone) is compared with the reference and with the
masked E005c run. The pass lines were fixed in EXPERIMENTS.md (E012) before this tool existed.

Everything printed is an aggregate over tags and frames: counts and shares. Safe to paste back.
"""

import argparse
import copy
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from config import PARAMS  # noqa: E402
from possession import H, A, atd, features, masking, pipeline, rules  # noqa: E402
from possession import tagging, team_model  # noqa: E402
from possession import spec as spec_module  # noqa: E402

# Params
SEEDS = (0, 1, 2, 3, 4)        # the masked harness's seeds
GROUPS = ("attacking", "defending")
PASS_AGREE = 0.80              # (2) agreement with the reference, per group
PASS_FIXES = 0.50              # (3) right where the masked E005c run has the wrong side
PASS_TAGS = 50                 # (1) tags covering a hidden in-play frame, mean over seeds
FALLBACK_CONFIDENT = 0.60      # (4) at or below this where the classifier is confident: unsure-only
ROWS = ("tags", "frames", "agree", "fixes", "n_wrong", "agree_confident", "agree_unsure")


def group_params(group):
    """Evidence params with only `group` on; its accuracy is unused here (side only)."""
    ev = copy.deepcopy(PARAMS["evidence"])
    ev.update(tag_acc_attack=1.0 if group == "attacking" else None,
              tag_acc_defend=1.0 if group == "defending" else None)
    return ev


def masked_p_home(masked, spec):
    """The gap classifier's P(home) on the masked run, as pipeline._hmm computes it."""
    labels = rules.finalize(pipeline.first_pass(masked, spec, PARAMS), spec)
    return team_model.cross_half(masked, spec, labels, PARAMS["evidence"]["near_m"])[0]


def _share(hit, where):
    return float(hit[where].mean()) if where.any() else float("nan")


def measure(side, intervals, reference, codes, p_home, frames):
    """One group, one seed: the gate's numbers on `frames` (hidden, alive in the reference)."""
    cue = frames & (side >= 0)
    right = side == reference
    wrong = cue & (codes <= A) & (codes != reference)
    conf = PARAMS["evidence"]["gap_conf"]
    confident = (p_home >= conf) | (p_home <= 1 - conf)
    return {
        "tags": sum(bool(cue[start:stop].any()) for start, stop, _, _ in intervals),
        "frames": int(cue.sum()),
        "agree": _share(right, cue),
        "fixes": _share(right, wrong),
        "n_wrong": int(wrong.sum()),
        "agree_confident": _share(right, cue & confident),
        "agree_unsure": _share(right, cue & ~confident),
    }


def run(tracking, spec, events, seeds):
    """(context, {group: [row per seed]})."""
    if pipeline.tag_cue_on(PARAMS):
        raise ValueError("the gate measures against E005c: switch the phase tag cue off first")
    reference = pipeline.label(tracking, spec, PARAMS)
    decoded = masking.decoded_mask(spec)
    alive = decoded & ((reference == H) | (reference == A))
    cues = {}
    for group in GROUPS:
        ev = group_params(group)
        cues[group] = (tagging.phase_cue(events, spec.n_frames, spec.fps, ev)[0],
                       tagging.phase_intervals(events, spec.n_frames, spec.fps, ev))
    gaps = alive & ~features.ball_visible(tracking)
    context = {"real untracked in-play frames": int(gaps.sum())}
    for group, (side, intervals) in cues.items():
        context["{} tags after trim".format(group)] = len(intervals)
        context["{} share of real gaps covered".format(group)] = "{:.3f}".format(
            _share(side >= 0, gaps))
    rows = {group: [] for group in GROUPS}
    for seed in seeds:
        masked, hidden = masking.mask_ball(tracking, reference, spec, seed,
                                           PARAMS["evidence"]["mid_m"])
        codes = pipeline.label(masked, spec, PARAMS)
        p_home = masked_p_home(masked, spec)
        for group, (side, intervals) in cues.items():
            rows[group].append(measure(side, intervals, reference, codes, p_home,
                                       hidden & alive))
    return context, rows


def verdict(stats):
    """(pass, fallback) for one group from {row: (mean, std)}."""
    passed = (stats["agree"][0] >= PASS_AGREE and stats["fixes"][0] >= PASS_FIXES
              and stats["tags"][0] >= PASS_TAGS)
    return passed, stats["agree_confident"][0] <= FALLBACK_CONFIDENT


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir")
    args = parser.parse_args()

    spec = spec_module.load(args.data_dir)
    tracking = atd.load(spec, args.data_dir)
    context, rows = run(tracking, spec, tagging.load(args.data_dir), SEEDS)

    print("E012 gate, seeds: {}".format(list(SEEDS)))
    for key, value in context.items():
        print("  {:<38} {}".format(key, value))
    stats = {group: {key: (float(np.nanmean([r[key] for r in rows[group]])),
                           float(np.nanstd([r[key] for r in rows[group]]))) for key in ROWS}
             for group in GROUPS}
    print("\nhidden in-play frames (mean ± std over seeds)")
    print("{:<18}".format("") + "".join("{:>17}".format(group) for group in GROUPS))
    for key in ROWS:
        pattern = "{:.0f} ± {:.0f}" if key in ("tags", "frames", "n_wrong") else "{:.3f} ± {:.3f}"
        print("{:<18}".format(key) + "".join(
            "{:>17}".format(pattern.format(*stats[group][key])) for group in GROUPS))
    print("\npass lines: agree >= {}, fixes >= {}, tags >= {}; fallback if agree_confident <= {}"
          .format(PASS_AGREE, PASS_FIXES, PASS_TAGS, FALLBACK_CONFIDENT))
    for group in GROUPS:
        passed, fallback = verdict(stats[group])
        print("  {:<10} {}   unsure-only fallback: {}".format(
            group, "PASS" if passed else "FAIL", "yes" if fallback else "no"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
