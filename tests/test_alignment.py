"""Frame alignment and the ATD reader, on files small enough to know every value in them.

An off-by-one here moves every transition and costs spellF1 everywhere while macroF1 barely
notices (EVALUATION.md), so each rule gets its own test.
"""

import os

import numpy as np
import pytest

import games
from possession import H, A, atd, spec as spec_module

TINY = spec_module.GameSpec(n_frames=4, fps=25, pitch=(100.0, 50.0), graded_blocks=((1, 3),),
                            kickoffs=(1,))


def write_bundle(folder, raw_lines, teams=("T1", "T2"), slot_teams=("T1", "T2")):
    """A minimal ATD pair under the real file names; returns the data_dir."""
    paths = games.asset_paths(data_dir=str(folder))
    os.makedirs(os.path.dirname(paths["atd_raw"]), exist_ok=True)
    team_xml = "".join('<Team id="{0}"><Name>{0} name</Name></Team>'.format(t) for t in teams)
    player_xml = "".join('<Player id="P{}" teamId="{}"/>'.format(i, t)
                         for i, t in enumerate(slot_teams, start=1))
    with open(paths["atd_meta"], "w") as handle:
        handle.write("<main><Metadata><Teams>{}</Teams><Players>{}</Players></Metadata></main>"
                     .format(team_xml, player_xml))
    with open(paths["atd_raw"], "w") as handle:
        handle.write("\n".join(raw_lines) + "\n")
    return str(folder)


def line(frame_id, slots, ball):
    return "{}:{}:{}".format(frame_id, ";".join(slots), ball)


GOOD = [line(1, ["0.1,0.2,1.5", "0.3,0.4,2.5"], "0.5,0.5,9"),
        line(2, ["NaN,NaN,NaN", "0.3,0.4,2.5"], "NaN,NaN,NaN"),
        line(3, ["0.1,NaN,1.0", "0.6,0.4,0.0"], "0.6,0.4,1"),
        line(4, ["0.1,0.2,1.5", "0.3,0.4,2.5"], "0.25,0.75,0")]


# -- the real match layout ------------------------------------------------------------------------

def test_real_layout_matches_published_numbers():
    spec = spec_module.real()
    assert spec.n_frames == 199319
    assert spec.graded_blocks == ((11185, 90651), (110808, 190197))
    assert int(spec.graded_mask().sum()) == 158857
    assert spec.kickoffs == (11185, 113742)


def test_pre_kickoff_block_is_the_documented_2934_frames():
    pre = np.flatnonzero(spec_module.real().pre_kickoff_mask())
    assert len(pre) == 2934
    assert (pre[0], pre[-1]) == (110808, 113741)


def test_half_of_frame_boundaries():
    half = spec_module.real().half_of_frame()
    assert half[11184] == 0 and half[11185] == 1 and half[90651] == 1 and half[90652] == 0
    assert half[110807] == 0 and half[110808] == 2 and half[190197] == 2 and half[190198] == 0


def test_games_kickoff_anchors_are_not_the_kickoff_frames():
    """Pinned so nobody 'fixes' spec.KICKOFFS from games.py: the anchors are playback positions."""
    anchors = [games.anchor_video_frame(games.GAMES["BAR-ATH"][key])
               for key in ("kickoff", "kickoff_2nd")]
    assert anchors == [11186, 113741]
    assert tuple(anchors) != spec_module.KICKOFFS["BAR-ATH"]


def test_load_picks_synthetic_spec_only_when_present(synthetic, tmp_path):
    out, generated = synthetic
    assert spec_module.load(out).n_frames == generated["n_video_frames"]
    assert spec_module.load(str(tmp_path)) == spec_module.real()
    assert spec_module.load(None) == spec_module.real()


# -- ATD reader -----------------------------------------------------------------------------------

def test_frame_id_one_is_row_zero_and_values_land_in_metres(tmp_path):
    tracking = atd.load(TINY, write_bundle(tmp_path, GOOD))
    assert tracking.players.shape == (4, 2, 2)
    np.testing.assert_allclose(tracking.players[0, 0], [10.0, 10.0])   # 0.1*100, 0.2*50
    np.testing.assert_allclose(tracking.players[0, 1], [30.0, 20.0])
    np.testing.assert_allclose(tracking.ball[0], [50.0, 25.0])
    np.testing.assert_allclose(tracking.ball[3], [25.0, 37.5])
    assert tracking.player_speed[0, 1] == 2.5


def test_missing_values_are_nan_and_half_missing_counts_as_missing(tmp_path):
    tracking = atd.load(TINY, write_bundle(tmp_path, GOOD))
    assert np.isnan(tracking.players[1, 0]).all()
    assert np.isnan(tracking.ball[1]).all()
    assert np.isnan(tracking.players[2, 0]).all()  # x present, y NaN -> whole slot missing


def test_ball_with_two_fields_is_accepted(tmp_path):
    """The ball group's field count is not documented; x,y is the minimum we rely on."""
    lines = [l.rsplit(",", 1)[0] for l in GOOD]
    tracking = atd.load(TINY, write_bundle(tmp_path, lines))
    np.testing.assert_allclose(tracking.ball[0], [50.0, 25.0])


@pytest.mark.parametrize("lines, message", [
    (GOOD[:3], "frame_id 1..4"),                       # a frame missing at the end
    ([GOOD[1], GOOD[0]] + GOOD[2:], "frame_id 1..4"),  # out of order
    ([l.replace("{}:".format(i + 1), "{}:".format(i), 1) for i, l in enumerate(GOOD)],
     "frame_id 1..4"),                                 # counting from 0 instead of 1
    (GOOD[:2] + ["3:0.1,0.2,1.5;0.3,0.4:0.6,0.4,1"] + GOOD[3:], "line 3 has"),
])
def test_malformed_files_fail_loudly(tmp_path, lines, message):
    with pytest.raises(ValueError, match=message):
        atd.load(TINY, write_bundle(tmp_path, lines))


def test_errors_never_quote_values(tmp_path):
    bad = GOOD[:2] + ["3:0.123456,0.2,1.5;0.3,0.4:0.6,0.4,1"] + GOOD[3:]
    with pytest.raises(ValueError) as error:
        atd.load(TINY, write_bundle(tmp_path, bad))
    assert "0.123456" not in str(error.value)


def test_slot_teams_follow_team_id_and_first_team_is_home(tmp_path):
    folder = write_bundle(tmp_path, GOOD, teams=("HOME", "AWAY"), slot_teams=("AWAY", "HOME"))
    tracking = atd.load(TINY, folder)
    assert tracking.slot_team.tolist() == [A, H]
    assert tracking.team_names == ("HOME name", "AWAY name")


def test_unknown_team_id_is_an_error(tmp_path):
    folder = write_bundle(tmp_path, GOOD, slot_teams=("T1", "T9"))
    with pytest.raises(ValueError, match="teamId"):
        atd.load(TINY, folder)
