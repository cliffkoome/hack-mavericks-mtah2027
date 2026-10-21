"""Touches and the last-touch side (possession/touches.py): hand-built and synthetic data only."""

import numpy as np
import pytest

from possession import H, A, D, atd, features, pipeline, touches, spec as spec_module
from config import PARAMS

FPS = 25


class FakeTracking:
    def __init__(self, players, ball, slot_team):
        self.players = np.asarray(players, dtype=float)
        self.ball = np.asarray(ball, dtype=float)
        self.slot_team = np.asarray(slot_team, dtype=np.int8)


def test_savgol_keeps_lines_and_gaps():
    ball = np.column_stack([np.arange(20.0), 2 * np.arange(20.0)])
    ball[8:10] = np.nan
    out = touches.smooth(ball)
    np.testing.assert_allclose(out[~np.isnan(ball[:, 0])], ball[~np.isnan(ball[:, 0])])
    assert np.isnan(out[8:10]).all()
    assert touches.SAVGOL.sum() == pytest.approx(1.0)


def _kick(n=100, turn=50, player_at=(0.5, 0.0), second=None):
    """Ball runs along +x at 10 m/s, then along +y from frame `turn`; a player near the turn."""
    t = np.arange(n) / FPS
    ball = np.where(t[:, None] < turn / FPS, np.column_stack([10 * t, np.zeros(n)]),
                    np.column_stack([np.full(n, 10 * turn / FPS), 10 * (t - turn / FPS)]))
    corner = ball[turn]
    players = np.full((n, 2, 2), 50.0)
    players[:, 0] = corner + np.asarray(player_at)
    if second is not None:
        players[:, 1] = corner + np.asarray(second)
    return FakeTracking(players, ball, [A, H])


def test_a_sharp_turn_next_to_a_player_is_a_touch():
    tracking = _kick()
    dv = touches.velocity_change(tracking.ball, FPS)
    frames, sides = touches.detect(tracking, dv, tau=5.0)
    assert len(frames) == 1 and abs(frames[0] - 50) <= 3 and sides[0] == A


def test_straight_flight_is_not_a_touch():
    n = 100
    ball = np.column_stack([10 * np.arange(n) / FPS, np.zeros(n)])
    tracking = FakeTracking(np.tile([[[5.0, 0.3]]], (n, 1, 1)), ball, [H])
    frames, _ = touches.detect(tracking, touches.velocity_change(ball, FPS), tau=5.0)
    assert len(frames) == 0


def test_a_turn_with_both_sides_in_reach_is_a_duel():
    tracking = _kick(second=(-0.5, 0.0))
    dv = touches.velocity_change(tracking.ball, FPS)
    frames, sides = touches.detect(tracking, dv, tau=5.0)
    assert len(frames) == 1 and sides[0] == touches.NOBODY


def test_free_flight_threshold_uses_far_frames_only():
    dv = np.array([1.0, 2.0, 3.0, 100.0, np.nan])
    distance = np.array([10.0, 10.0, 10.0, 0.5, 10.0])
    where = np.ones(5, dtype=bool)
    assert touches.free_flight_threshold(dv, distance, where) == pytest.approx(
        np.percentile([1.0, 2.0, 3.0], touches.FREE_FLIGHT_PCT))


def test_last_touch_carries_resets_and_ignores_duels():
    n = 200
    visible = np.ones(n, dtype=bool)
    visible[100:130] = False                          # a 1.2 s gap: forgets the last touch
    stoppage = np.zeros(n, dtype=bool)
    stoppage[170:175] = True
    frames = np.array([10, 40, 140, 160])
    sides = np.array([H, touches.NOBODY, A, H], dtype=np.int8)  # 40 is a duel
    out = touches.last_touch(n, frames, sides, visible, stoppage, FPS)
    assert (out[:10] == touches.NOBODY).all()
    assert (out[10:100] == H).all()                   # the duel at 40 keeps H
    assert (out[100:140] == touches.NOBODY).all()     # gap unseen, then reset
    assert (out[140:160] == A).all() and (out[160:170] == H).all()
    assert (out[170:] == touches.NOBODY).all()        # stoppage resets


def test_short_gap_keeps_the_side_on_seen_frames():
    n = 100
    visible = np.ones(n, dtype=bool)
    visible[30:40] = False                            # 0.4 s: kept
    out = touches.last_touch(n, np.array([5]), np.array([A], dtype=np.int8), visible,
                             np.zeros(n, dtype=bool), FPS)
    assert (out[30:40] == touches.NOBODY).all() and (out[40:] == A).all()


def test_detects_touches_on_synthetic_play(synthetic):
    data_dir, _ = synthetic
    spec = spec_module.load(data_dir)
    tracking = atd.load(spec, data_dir)
    dv = touches.velocity_change(tracking.ball, spec.fps)
    _, distance = features.nearest_player(tracking)
    tau = touches.free_flight_threshold(dv, distance, features.ball_visible(tracking))
    frames, sides = touches.detect(tracking, dv, tau)
    assert len(frames) > 0 and set(np.unique(sides)) <= {H, A, touches.NOBODY}
    reference = pipeline.label(tracking, spec, PARAMS)
    side = touches.last_touch(spec.n_frames, frames, sides, features.ball_visible(tracking),
                              reference == D, spec.fps)
    assert (side[reference == D] == touches.NOBODY).all()
