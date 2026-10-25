"""Smart Tagging as frame intervals with a side (H/A) instead of a club name.

Reading and the time base are the starter's (starter/load_smart_tagging.py: round(seconds * 25),
verified against both kickoffs). What this adds is the club -> side mapping, done through an
explicit table so a spelling we have not seen is an error rather than a silent "N/A".
"""

import os
import sys

import numpy as np

from possession import ROOT, H, A

sys.path.insert(0, os.path.join(ROOT, "starter"))
import load_smart_tagging  # noqa: E402

# Params
# Lower-cased Team tag -> side. DATA.md documents "Barcelona", "Atletic Club" (sic) and "N/A";
# the correct spelling is listed too in case a re-export fixes the typo.
TEAM_SIDES = {"barcelona": H, "atletic club": A, "athletic club": A, "n/a": None}
# E012: phase codes whose Team is the side with the ball (attacking) or without it (defending),
# as measured in E002. Every other code carries no side for this cue. Fixed before any run.
ATTACKING = frozenset({"BUILD UP", "PROGRESSION", "LONG BALL", "ATTACKING TRANSITION"})
DEFENDING = frozenset({"DEFENDING IN ATTACKING THIRD", "DEFENDING IN MIDDLE THIRD",
                       "DEFENDING IN DEFENSIVE THIRD", "DEFENSIVE TRANSITION"})


def side_of(team_tag):
    """H, A, or None for N/A. Raises on a club name not in TEAM_SIDES."""
    key = (team_tag or "n/a").strip().lower()
    if key not in TEAM_SIDES:
        raise ValueError("unknown Smart Tagging Team value {!r}; add it to TEAM_SIDES"
                         .format(team_tag))
    return TEAM_SIDES[key]


def load(data_dir=None):
    """[{code, type, side, start_frame, end_frame}] with code and type upper-cased, side H/A/None."""
    out = []
    for event in load_smart_tagging.load(data_dir=data_dir):
        out.append({"code": (event["code"] or "").strip().upper(),
                    "type": (event["tags"].get("Type") or "").strip().upper() or None,
                    "side": side_of(event["tags"].get("Team")),
                    "start_frame": event["start_frame"], "end_frame": event["end_frame"]})
    return out


def phase_intervals(events, n_frames, fps, params):
    """[(start, stop, side, accuracy)] half-open: the on-groups' phase tags as cue intervals (E012).

    side is the side with the ball (a defending tag's Team flipped). Each tag loses tag_trim_s at
    both ends, and is skipped if less than tag_trim_s is left. A group whose accuracy is None is
    off, and its tags are ignored entirely.
    """
    groups = ((ATTACKING, params["tag_acc_attack"], False),
              (DEFENDING, params["tag_acc_defend"], True))
    trim = int(round(params["tag_trim_s"] * fps))
    out = []
    for event in events:
        for codes, accuracy, flip in groups:
            if accuracy is None or event["code"] not in codes or event["side"] is None:
                continue
            start = max(0, event["start_frame"] + trim)
            stop = min(n_frames, event["end_frame"] - trim)
            if stop - start >= trim:
                out.append((start, stop, 1 - event["side"] if flip else event["side"], accuracy))
    return out


def phase_cue(events, n_frames, fps, params):
    """(side, accuracy, run) per frame from phase_intervals.

    side: H or A where the covering tags agree, -1 where none covers the frame or they disagree.
    accuracy: the highest trust among the agreeing tags, NaN where side is -1.
    run: an id per run of consecutive frames with one side, so agreeing tags that overlap count
    as one reading under the cap; -1 where side is -1.
    """
    covers = np.zeros((2, n_frames), dtype=bool)
    trust = np.full((2, n_frames), np.nan)
    for start, stop, side, accuracy in phase_intervals(events, n_frames, fps, params):
        covers[side, start:stop] = True
        trust[side, start:stop] = np.fmax(trust[side, start:stop], accuracy)
    side = np.where(covers[H] & ~covers[A], H, np.where(covers[A] & ~covers[H], A, -1))
    accuracy = np.where(side == H, trust[H], np.where(side == A, trust[A], np.nan))
    starts = np.diff(np.concatenate(([-1], side))) != 0
    run = np.where(side >= 0, np.cumsum(starts) - 1, -1)
    return side.astype(np.int8), accuracy, run
