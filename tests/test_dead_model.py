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


# E013: Lang et al.'s recipe.

class FakeSpec:
    fps = FPS
    pitch = (100.0, 60.0)

    def __init__(self, n):
        self.n_frames = n
        self.graded_blocks = ((0, n - 1),)


class TeamTracking(FakeTracking):
    def __init__(self, players, speed, slot_team):
        super().__init__(players, speed)
        self.slot_team = np.asarray(slot_team, dtype=np.int8)


def test_team_frame_features_on_hand_built_teams():
    from possession import H, A

    n = 60
    # H at x = 10 and 30 (defends low x), A at x = 70 and 90; A's x is mirrored to 30 and 10.
    players = np.tile([[10.0, 20.0], [30.0, 40.0], [70.0, 20.0], [90.0, 40.0]], (n, 1, 1))
    speed = np.tile([1.0, 3.0, 2.0, 2.0], (n, 1))
    speed[:, 0] += np.arange(n) * 0.04  # player 0 accelerates at 1 m/s^2
    out = dm.team_frame_features(TeamTracking(players, speed, [H, H, A, A]), FakeSpec(n))
    home = dict(zip(dm.TEAM_FEATURES, out[30, 0]))
    away = dict(zip(dm.TEAM_FEATURES, out[30, 1]))
    assert home["centroid_x"] == pytest.approx(20.0) and away["centroid_x"] == pytest.approx(20.0)
    assert home["centroid_y"] == pytest.approx(30.0) and home["std_x"] == pytest.approx(10.0)
    assert home["mean_abs_accel"] == pytest.approx(0.5) and away["mean_abs_accel"] == 0.0
    assert away["speed_p50"] == pytest.approx(2.0)


def test_lang_offsets_reach_80_seconds():
    base = np.arange(5000, dtype=float)[:, None]
    out = dm.context(base, FPS, dm.LANG_OFFSETS_S)
    assert out[2500, dm.LANG_OFFSETS_S.index(80.0)] == 4500
    assert np.isnan(out[10, dm.LANG_OFFSETS_S.index(-80.0)])


def test_swap_exchanges_the_teams_and_leaves_the_rest():
    rng = np.random.default_rng(0)
    matrix = rng.normal(size=(4, len(dm.LANG_OFFSETS_S) * dm.LANG_WIDTH))
    swapped = dm.swap(matrix).reshape(4, len(dm.LANG_OFFSETS_S), dm.LANG_WIDTH)
    shaped = matrix.reshape(swapped.shape)
    common, k = len(dm.FEATURES), len(dm.TEAM_FEATURES)
    assert np.array_equal(swapped[..., :common], shaped[..., :common])
    assert np.array_equal(swapped[..., common:common + k], shaped[..., common + k:])
    assert np.array_equal(dm.swap(dm.swap(matrix)), matrix)


def test_lang_prediction_does_not_depend_on_which_team_is_home():
    rng = np.random.default_rng(1)
    matrix = rng.normal(size=(400, len(dm.LANG_OFFSETS_S) * dm.LANG_WIDTH))
    labels = (matrix[:, len(dm.FEATURES)] > 0).astype(np.int8)  # H's first feature decides
    model = dm.fit(matrix, labels, "lang")
    assert np.allclose(dm.predict(model, matrix, "lang"), dm.predict(model, dm.swap(matrix),
                                                                      "lang"))


def test_unknown_recipe_is_an_error(synthetic):
    data_dir, _ = synthetic
    spec = spec_module.load(data_dir)
    tracking = atd.load(spec, data_dir)
    with pytest.raises(ValueError, match="recipe"):
        dm.cross_half(tracking, spec, np.zeros(spec.n_frames, dtype=np.int8), recipe="nope")


def test_lang_cross_half_never_trains_on_the_half_it_predicts(synthetic, monkeypatch):
    from scipy.ndimage import median_filter

    data_dir, _ = synthetic
    spec = spec_module.load(data_dir)
    tracking = atd.load(spec, data_dir)
    reference = pipeline.label(tracking, spec, PARAMS)
    p_dead, folds = dm.cross_half(tracking, spec, reference, recipe="lang")
    assert folds
    for predicted, train in folds:
        assert len(train) and not predicted[train].any()
        assert not np.isnan(p_dead[predicted]).any()
    again, _ = dm.cross_half(tracking, spec, reference, recipe="lang")
    assert np.array_equal(p_dead, again, equal_nan=True)
    # the filter is a 91-frame median of the raw prediction, within each predicted half
    monkeypatch.setattr(dm, "MEDIAN_S", 1 / FPS)
    raw, _ = dm.cross_half(tracking, spec, reference, recipe="lang")
    size = int(round(3.64 * FPS))
    for predicted, _ in folds:
        assert np.allclose(p_dead[predicted], median_filter(raw[predicted], size=size,
                                                            mode="nearest"))


def test_e010_recipe_is_the_default_and_unchanged(synthetic):
    data_dir, _ = synthetic
    spec = spec_module.load(data_dir)
    tracking = atd.load(spec, data_dir)
    reference = pipeline.label(tracking, spec, PARAMS)
    default, _ = dm.cross_half(tracking, spec, reference)
    named, _ = dm.cross_half(tracking, spec, reference, recipe="e010")
    assert np.array_equal(default, named, equal_nan=True)
    assert PARAMS["evidence"]["gap_dead_recipe"] == "e010"


def test_no_slow_teacher_is_pass_1_without_the_slow_cue(synthetic):
    import copy

    from possession import rules

    data_dir, _ = synthetic
    spec = spec_module.load(data_dir)
    tracking = atd.load(spec, data_dir)
    labels = pipeline.label(tracking, spec, PARAMS)
    assert pipeline.dead_teacher(tracking, spec, PARAMS, labels) is labels  # default: pass1
    params = copy.deepcopy(PARAMS)
    params["evidence"]["gap_dead_teacher"] = "no_slow"
    teacher = pipeline.dead_teacher(tracking, spec, params, labels)
    params["evidence"]["slow_speed"] = None
    expected = rules.finalize(pipeline.first_pass(tracking, spec, params), spec)
    assert np.array_equal(teacher, expected)
    params["evidence"]["gap_dead_teacher"] = "nope"
    with pytest.raises(ValueError, match="teacher"):
        pipeline.dead_teacher(tracking, spec, params, labels)
