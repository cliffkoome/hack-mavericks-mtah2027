"""Smart Tagging side mapping, and the cue inspection tool end to end on synthetic data."""

import os
import subprocess
import sys

import numpy as np
import pytest

import make_synthetic
from possession import H, A, tagging

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.mark.parametrize("tag, side", [("Barcelona", H), ("Atletic Club", A),
                                       ("Athletic Club", A), (" atletic club ", A),
                                       ("N/A", None), (None, None)])
def test_side_of_known_names(tag, side):
    assert tagging.side_of(tag) == side


def test_unknown_club_is_an_error_not_na():
    with pytest.raises(ValueError, match="TEAM_SIDES"):
        tagging.side_of("Real Madrid")


def test_synthetic_tagging_maps_cleanly(synthetic):
    out, _ = synthetic
    events = tagging.load(out)
    assert {e["side"] for e in events} == {H, A, None}
    assert all(e["code"] == e["code"].upper() for e in events)
    assert all(e["side"] is None for e in events if e["type"] == "THROW-IN")


def test_inspect_cues_runs_and_prints_only_tables(synthetic):
    out, _ = synthetic
    result = subprocess.run(
        [sys.executable, os.path.join(ROOT, "tools", "inspect_cues.py"), "--data-dir", out,
         "--gt", os.path.join(out, "synthetic_possession_gt.csv")],
        capture_output=True, text=True, cwd=ROOT)
    assert result.returncode == 0, result.stderr
    for heading in ("1. cues", "2. team", "3. tagging"):
        assert heading in result.stdout
    assert len(result.stdout.splitlines()) < 60  # aggregate tables, never per-frame output
    assert make_synthetic.ATD_TEAM_NAMES[0] not in result.stdout


def _event(code, side, start, end):
    return {"code": code, "type": None, "side": side, "start_frame": start, "end_frame": end}


def _params(attack=0.8, defend=0.7, trim_s=1.0):
    return {"tag_acc_attack": attack, "tag_acc_defend": defend, "tag_trim_s": trim_s}


def test_phase_intervals_trim_flip_and_skip():
    events = [_event("BUILD UP", H, 100, 200),                       # kept, trimmed
              _event("DEFENDING IN MIDDLE THIRD", H, 300, 400),      # flipped to A
              _event("PROGRESSION", A, 500, 570),                    # 20 frames left: skipped
              _event("PROGRESSION", None, 600, 700),                 # N/A: skipped
              _event("BALL IN THE BOX", H, 800, 900)]                # no side for this cue
    assert tagging.phase_intervals(events, 1000, 25, _params()) == [
        (125, 175, H, 0.8), (325, 375, A, 0.7)]


def test_a_group_that_is_off_is_ignored():
    events = [_event("BUILD UP", H, 100, 200), _event("DEFENSIVE TRANSITION", H, 100, 200)]
    assert tagging.phase_intervals(events, 1000, 25, _params(attack=None)) == [
        (125, 175, A, 0.7)]
    assert tagging.phase_intervals(events, 1000, 25, _params(attack=None, defend=None)) == []


def test_phase_cue_merges_agreement_and_drops_conflict():
    events = [_event("BUILD UP", H, 0, 100),                         # H on 25..74
              _event("DEFENDING IN ATTACKING THIRD", A, 50, 150),    # H on 75..124, overlaps
              _event("LONG BALL", A, 200, 300),                      # A on 225..274
              _event("BUILD UP", H, 230, 330)]                       # H on 255..304: conflict
    side, accuracy, run = tagging.phase_cue(events, 400, 25, _params())
    assert (side[25:125] == H).all() and (side[:25] == -1).all()
    assert len(set(run[25:125])) == 1                                # agreeing tags: one run
    assert accuracy[60] == 0.8 and accuracy[50] == 0.8 and accuracy[100] == 0.7
    assert (side[225:255] == A).all() and (side[255:275] == -1).all()
    assert (side[275:305] == H).all() and run[230] != run[280]
    assert np.isnan(accuracy[260]) and run[260] == -1
