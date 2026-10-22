"""Dead/alive-from-movement classifier (possession/dead_model.py): hand-built and synthetic only."""

import numpy as np
import pytest

from config import PARAMS
from possession import atd, dead_model as dm, pipeline, spec as spec_module

FPS = 25


class FakeTracking:
    def __init__(self, players, speed):
        self.players = np.asarray(players, dtype=float)
        self.player_speed = np.asarray(speed, dtype=float)
        self.ball = np.full((len(self.players), 2), np.nan)


def test_frame_features_on_hand_built_speeds():
    n = 40
    speed = np.tile([0.0, 1.0, 3.0, 5.0], (n, 1))
    players = np.tile([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0], [10.0, 10.0]], (n, 1, 1))
    out = dm.frame_features(FakeTracking(players, speed), FPS)
    row = dict(zip(dm.FEATURES, out[20]))
    assert row["speed_p50"] == pytest.approx(2.0)
    assert row["share_jog"] == pytest.approx(0.5)
    assert row["share_run"] == pytest.approx(0.25)
    assert row["std_x"] == pytest.approx(5.0) and row["std_y"] == pytest.approx(5.0)


def test_frame_features_nan_where_nobody_is_tracked():
    speed = np.full((30, 3), np.nan)
    players = np.full((30, 3, 2), np.nan)
    assert np.isnan(dm.frame_features(FakeTracking(players, speed), FPS)).all()


def test_context_shifts_look_ahead_and_back():
    base = np.arange(300, dtype=float)[:, None].repeat(len(dm.FEATURES), axis=1)
    out = dm.context(base, FPS)
    width = len(dm.FEATURES)
    ahead = dm.OFFSETS_S.index(1.0) * width
    assert out[100, ahead] == 125 and np.isnan(out[-1, ahead])


def test_cross_half_never_trains_on_the_half_it_predicts(synthetic):
    data_dir, _ = synthetic
    spec = spec_module.load(data_dir)
    tracking = atd.load(spec, data_dir)
    reference = pipeline.label(tracking, spec, PARAMS)
    p_dead, folds = dm.cross_half(tracking, spec, reference)
    assert folds  # a half whose training frames hold one class only is skipped
    for predicted, train in folds:
        assert len(train) and not predicted[train].any()
    known = ~np.isnan(p_dead)
    assert ((p_dead[known] >= 0) & (p_dead[known] <= 1)).all()
    again, _ = dm.cross_half(tracking, spec, reference)
    assert np.array_equal(p_dead, again, equal_nan=True)
