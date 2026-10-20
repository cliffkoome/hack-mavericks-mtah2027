"""Smart Tagging as frame intervals with a side (H/A) instead of a club name.

Reading and the time base are the starter's (starter/load_smart_tagging.py: round(seconds * 25),
verified against both kickoffs). What this adds is the club -> side mapping, done through an
explicit table so a spelling we have not seen is an error rather than a silent "N/A".
"""

import os
import sys

from possession import ROOT, H, A

sys.path.insert(0, os.path.join(ROOT, "starter"))
import load_smart_tagging  # noqa: E402

# Params
# Lower-cased Team tag -> side. DATA.md documents "Barcelona", "Atletic Club" (sic) and "N/A";
# the correct spelling is listed too in case a re-export fixes the typo.
TEAM_SIDES = {"barcelona": H, "atletic club": A, "athletic club": A, "n/a": None}


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
