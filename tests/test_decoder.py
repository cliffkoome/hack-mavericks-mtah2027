"""The decoder: exact on tiny inputs, and the behaviours spellF1 needs from it."""

import copy
import itertools

import numpy as np
import pytest

from config import PARAMS
from possession import H, A, D, X, decoder, evidence, features, spec as spec_module

FPS = 25
LOG_TRANS = decoder.transition_matrix(PARAMS["decoder"], FPS)
LOG_INIT = np.log(np.array([0.5, 0.5, 1e-300]))


def emissions(sequence, strength=1.0):
    """Log-emissions favouring the given state per frame by `strength` nats."""
    out = np.zeros((len(sequence), 3))
    out[np.arange(len(sequence)), sequence] = strength
    return out


def segments(path):
    return int(np.sum(path[1:] != path[:-1])) + 1


def brute_force(log_emit, log_trans, log_init):
    best, best_path = -np.inf, None
    for path in itertools.product(range(3), repeat=len(log_emit)):
        score = log_init[path[0]] + log_emit[0, path[0]]
        for t in range(1, len(path)):
            score += log_trans[path[t - 1], path[t]] + log_emit[t, path[t]]
        if score > best:
            best, best_path = score, path
    return np.array(best_path)


@pytest.mark.parametrize("seed", range(20))
def test_viterbi_equals_brute_force(seed):
    rng = np.random.default_rng(seed)
    log_emit = rng.normal(0, 2, (6, 3))
    log_trans = np.log(rng.dirichlet(np.ones(3), 3))
    log_init = np.log(rng.dirichlet(np.ones(3)))
    assert np.array_equal(decoder.viterbi(log_emit, log_trans, log_init),
                          brute_force(log_emit, log_trans, log_init))


def test_clean_evidence_is_reproduced():
    truth = np.repeat([H, A, D, H], 250)
    assert np.array_equal(decoder.viterbi(emissions(truth), LOG_TRANS, LOG_INIT), truth)


def test_single_frame_flicker_is_removed():
    noisy = np.full(500, H)
    noisy[[100, 230, 231, 400]] = A
    assert (decoder.viterbi(emissions(noisy), LOG_TRANS, LOG_INIT) == H).all()


def test_short_possession_with_strong_evidence_survives():
    """The public window has a 19-frame possession; the decoder must not smooth such things away."""
    truth = np.full(600, H)
    truth[300:320] = A
    path = decoder.viterbi(emissions(truth, strength=2.0), LOG_TRANS, LOG_INIT)
    assert (path[302:318] == A).all() and segments(path) == 3


def test_higher_switch_cost_never_adds_segments():
    rng = np.random.default_rng(0)
    log_emit = rng.normal(0, 1, (3000, 3))
    counts = []
    for mean_s in (2, 5, 12, 30, 80):
        params = dict(PARAMS["decoder"], mean_possession_s=mean_s, mean_dead_s=mean_s)
        counts.append(segments(decoder.viterbi(log_emit, decoder.transition_matrix(params, FPS),
                                               LOG_INIT)))
    assert counts == sorted(counts, reverse=True)


def test_constant_offsets_per_frame_do_not_change_the_path():
    rng = np.random.default_rng(1)
    log_emit = rng.normal(0, 1, (400, 3))
    shifted = log_emit + rng.normal(0, 5, (400, 1))
    assert np.array_equal(decoder.viterbi(log_emit, LOG_TRANS, LOG_INIT),
                          decoder.viterbi(shifted, LOG_TRANS, LOG_INIT))


def test_ties_are_deterministic():
    path = decoder.viterbi(np.zeros((50, 3)), LOG_TRANS, LOG_INIT)
    assert (path == H).all()


def test_transition_rows_are_distributions():
    assert np.allclose(np.exp(LOG_TRANS).sum(axis=1), 1.0)


def test_halves_are_decoded_independently():
    """Strong A at the end of half 1 must not leak into half 2, which kicks off strongly H."""
    spec = spec_module.GameSpec(n_frames=1000, fps=FPS, pitch=(105.3, 68.0),
                                graded_blocks=((100, 399), (600, 899)), kickoffs=(100, 650))
    log_emit = np.zeros((1000, 3))
    log_emit[100:400, A] = 1.0
    log_emit[650:900, H] = 1.0
    codes = decoder.decode_match(log_emit, spec, PARAMS["decoder"])
    assert (codes[:100] == X).all() and (codes[400:600] == X).all() and (codes[900:] == X).all()
    assert (codes[600:650] == X).all()          # pre-kickoff is left to rules.finalize
    assert (codes[100:400] == A).all() and (codes[650:900] == H).all()


# -- evidence ---------------------------------------------------------------------------------

class FakeTracking:
    def __init__(self, players, ball, slot_team):
        self.players = np.asarray(players, dtype=float)
        self.ball = np.asarray(ball, dtype=float)
        self.slot_team = np.asarray(slot_team, dtype=np.int8)


SPEC = spec_module.GameSpec(n_frames=4, fps=FPS, pitch=(100.0, 50.0), graded_blocks=((0, 3),),
                            kickoffs=(0,))


def evidence_params(**overrides):
    params = copy.deepcopy(PARAMS["evidence"])
    params.update({"weight": 1.0, "speed_lag": 1, **overrides})
    return params


def test_team_cue_weakens_with_distance():
    players = [[[10, 10], [80, 10]]] * 4
    ball = [[11, 10], [13, 10], [30, 10], [79.5, 10]]  # near H, mid H, far H, near A
    log_emit = evidence.log_emissions(FakeTracking(players, ball, [H, A]), SPEC,
                                      evidence_params())
    margin = log_emit[:, H] - log_emit[:, A]
    assert margin[0] > margin[1] > margin[2] > 0
    assert margin[3] < 0


def test_missing_and_outside_ball_favour_dead():
    players = [[[10, 10]]] * 4
    ball = [[np.nan, np.nan], [50, 25], [50, -5], [50, 25]]
    log_emit = evidence.log_emissions(FakeTracking(players, ball, [H]), SPEC, evidence_params())
    assert log_emit[0, D] > log_emit[0, H]                    # not tracked
    assert log_emit[2, D] - log_emit[2, H] > log_emit[1, D] - log_emit[1, H]  # outside the lines


def test_run_length():
    flag = np.array([0, 1, 1, 0, 1, 1, 1, 0], dtype=bool)
    assert features.run_length(flag).tolist() == [0, 2, 2, 0, 3, 3, 3, 0]


def test_a_long_untracked_run_counts_as_at_most_the_cap():
    """10 s of lost ball must not weigh more towards D than missing_run_cap frames of it."""
    def dead_margin(n_missing):
        n = n_missing + 2
        spec = spec_module.GameSpec(n_frames=n, fps=FPS, pitch=(100.0, 50.0),
                                    graded_blocks=((0, n - 1),), kickoffs=(0,))
        players = np.full((n, 1, 2), np.nan)
        ball = np.full((n, 2), np.nan)
        log_emit = evidence.log_emissions(FakeTracking(players, ball, [H]), spec,
                                          evidence_params())
        return float(np.sum(log_emit[:, D] - log_emit[:, H]))
    cap = PARAMS["evidence"]["missing_run_cap"]
    assert dead_margin(250) == pytest.approx(dead_margin(cap - 2), rel=0.02)
    assert dead_margin(10) < dead_margin(cap - 2)


def test_weight_scales_everything():
    players = [[[10, 10]]] * 4
    ball = [[11, 10], [np.nan, np.nan], [50, -5], [50, -5]]
    tracking = FakeTracking(players, ball, [H])
    full = evidence.log_emissions(tracking, SPEC, evidence_params())
    fifth = evidence.log_emissions(tracking, SPEC, evidence_params(weight=0.2))
    np.testing.assert_allclose(fifth, 0.2 * full)
