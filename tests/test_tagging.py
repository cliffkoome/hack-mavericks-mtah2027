"""Smart Tagging side mapping, and the cue inspection tool end to end on synthetic data."""

import os
import subprocess
import sys

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
