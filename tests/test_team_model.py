"""Team-from-movement classifier (possession/team_model.py): hand-built and synthetic data only."""

import numpy as np
import pytest

from config import PARAMS
from possession import H, A, atd, pipeline, spec as spec_module, team_model as tm

FPS = 25
LENGTH = 100.0
N_SIDE = 7


class FakeTracking:
    def __init__(self, players, slot_team):
        self.players = np.asarray(players, dtype=float)
        self.slot_team = np.asarray(slot_team, dtype=np.int8)
        self.ball = np.full((len(self.players), 2), np.nan)


def _spec(n, blocks):
    return spec_module.GameSpec(n_frames=n, fps=FPS, pitch=(LENGTH, 50.0),
                                graded_blocks=blocks, kickoffs=tuple(b[0] for b in blocks))


def _static(n, home_x, away_x, home_y=None, away_y=None):
    """n frames of two 7-player teams standing still at the given x (and y) positions."""
    home_y = np.arange(N_SIDE) * 5.0 if home_y is None else home_y
    away_y = np.arange(N_SIDE) * 5.0 if away_y is None else away_y
    frame = np.concatenate([np.column_stack([home_x, home_y]), np.column_stack([away_x, away_y])])
    return FakeTracking(np.tile(frame, (n, 1, 1)), [H] * N_SIDE + [A] * N_SIDE)


def test_attack_sign_from_where_each_team_stands():
    home_low = np.full(N_SIDE, 30.0)
    tracking = _static(200, home_low, np.full(N_SIDE, 70.0))
    tracking.players[100:, :N_SIDE, 0] = 70.0       # second half: ends swapped
    tracking.players[100:, N_SIDE:, 0] = 30.0
    sign = tm.attack_sign(tracking, _spec(200, ((0, 99), (100, 199))))
    assert (sign[:100] == (1, -1)).all() and (sign[100:] == (-1, 1)).all()


def test_team_features_on_hand_built_positions():
    home_x = np.array([2.0, 20, 30, 40, 50, 60, 70])  # 2.0 is the deepest: excluded as keeper
    away_x = np.full(N_SIDE, 90.0)                    # away defends the high end: x' = 100 - x
    tracking = _static(60, home_x, away_x)
    tracking.players[:, N_SIDE, :] = [72.0, 0.0]       # an away player 2 m from home's 70 at y=0
    out = tm.team_features(tracking, _spec(60, ((0, 59),)))
    home = dict(zip(tm.FEATURES, out[30, 0]))
    assert home["centroid_x"] == pytest.approx(45.0)
    assert home["line_height"] == pytest.approx(20.0)
    assert home["centroid_vx"] == pytest.approx(0.0)
    assert home["running_back"] == 0
    assert home["std_x"] == pytest.approx(np.std([20, 30, 40, 50, 60, 70]))
    # away, attack-normalised: six at 10 and the one at 72 -> 28; one 10 is the keeper
    assert out[30, 1, 0] == pytest.approx((5 * 10.0 + 28.0) / 6)


def test_running_back_counts_players_retreating_fast():
    n = 60
    tracking = _static(n, np.linspace(20, 80, N_SIDE), np.full(N_SIDE, 90.0))
    retreat = 6.0 / FPS                                # 6 m/s towards the own (low-x) goal
    tracking.players[:, 1:4, 0] -= np.arange(n)[:, None] * retreat
    out = tm.team_features(tracking, _spec(n, ((0, n - 1),)))
    assert out[30, 0, tm.FEATURES.index("running_back")] == 3


def test_too_few_players_gives_nan():
    tracking = _static(20, np.linspace(20, 80, N_SIDE), np.full(N_SIDE, 90.0))
    tracking.players[:, 2:N_SIDE, :] = np.nan          # home keeps 2 players only
    out = tm.team_features(tracking, _spec(20, ((0, 19),)))
    assert np.isnan(out[:, 0]).all() and not np.isnan(out[10, 1, 0])


def test_context_shifts_with_nan_at_the_edges():
    team = np.arange(10, dtype=float)[:, None, None] * np.ones((1, 2, tm.N_TEAM))
    matrix = tm.context(team, FPS)
    width = 2 * tm.N_TEAM
    later = tm.OFFSETS_S.index(1.0)                    # +1 s = 25 frames: past the end of 10
    assert matrix[3, 0] == 3.0
    assert np.isnan(matrix[:, later * width]).all()


def test_swap_exchanges_the_teams_at_every_offset():
    matrix = np.random.default_rng(0).normal(size=(4, len(tm.OFFSETS_S) * 2 * tm.N_TEAM))
    swapped = tm.swap(matrix)
    np.testing.assert_array_equal(swapped[:, :tm.N_TEAM], matrix[:, tm.N_TEAM:2 * tm.N_TEAM])
    np.testing.assert_array_equal(tm.swap(swapped), matrix)


@pytest.fixture(scope="module")
def random_model():
    rng = np.random.default_rng(1)
    matrix = rng.normal(size=(600, len(tm.OFFSETS_S) * 2 * tm.N_TEAM))
    labels = (matrix[:, 0] > matrix[:, tm.N_TEAM]).astype(np.int8)
    return matrix, labels, tm.fit(matrix, labels)


def test_prediction_flips_exactly_when_the_teams_swap(random_model):
    matrix, _, model = random_model
    np.testing.assert_allclose(tm.predict(model, matrix), 1 - tm.predict(model, tm.swap(matrix)),
                               atol=1e-12)


def test_fit_is_deterministic(random_model):
    matrix, labels, model = random_model
    np.testing.assert_array_equal(tm.predict(model, matrix),
                                  tm.predict(tm.fit(matrix, labels), matrix))


def test_cross_half_never_trains_on_the_half_it_predicts(synthetic):
    data_dir, _ = synthetic
    spec = spec_module.load(data_dir)
    tracking = atd.load(spec, data_dir)
    reference = pipeline.label(tracking, spec, PARAMS)
    p_home, folds = tm.cross_half(tracking, spec, reference, PARAMS["evidence"]["near_m"])
    assert len(folds) == 2
    for predicted, train in folds:
        assert len(train) and not predicted[train].any()
    decoded = np.zeros(spec.n_frames, dtype=bool)
    for (_, last), kickoff in zip(spec.graded_blocks, spec.kickoffs):
        decoded[kickoff:last + 1] = True
    assert not np.isnan(p_home[decoded]).any() and np.isnan(p_home[~decoded]).all()
