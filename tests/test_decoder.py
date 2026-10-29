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
        self.player_speed = np.full(self.players.shape[:2], np.nan)


SPEC = spec_module.GameSpec(n_frames=4, fps=FPS, pitch=(100.0, 50.0), graded_blocks=((0, 3),),
                            kickoffs=(0,))


def evidence_params(**overrides):
    params = copy.deepcopy(PARAMS["evidence"])
    params.update({"weight": 1.0, "speed_lag": 1, **overrides})
    return params


def test_team_cue_weakens_with_distance():
    players = [[[10, 10], [80, 10]]] * 4
    ball = [[11, 10], [13, 10], [30, 10], [79.5, 10]]  # near H, mid H, far H, near A
    tracking = FakeTracking(players, ball, [H, A])
    log_emit = evidence.log_emissions(tracking, SPEC, evidence_params(team_acc_far=0.6))
    margin = log_emit[:, H] - log_emit[:, A]
    assert margin[0] > margin[1] > margin[2] > 0
    assert margin[3] < 0
    no_far = evidence.log_emissions(tracking, SPEC, evidence_params(team_acc_far=None))
    assert no_far[2, H] == no_far[2, A]


def test_missing_and_outside_ball_favour_dead():
    players = [[[10, 10]]] * 4
    ball = [[np.nan, np.nan], [50, 25], [50, -5], [50, 25]]
    log_emit = evidence.log_emissions(FakeTracking(players, ball, [H]), SPEC, evidence_params())
    assert log_emit[0, D] > log_emit[0, H]                    # not tracked
    assert log_emit[2, D] - log_emit[2, H] > log_emit[1, D] - log_emit[1, H]  # outside the lines


def test_run_length():
    flag = np.array([0, 1, 1, 0, 1, 1, 1, 0], dtype=bool)
    assert features.run_length(flag).tolist() == [0, 2, 2, 0, 3, 3, 3, 0]


def _dead_margin(n_missing, **overrides):
    n = n_missing + 2
    spec = spec_module.GameSpec(n_frames=n, fps=FPS, pitch=(100.0, 50.0),
                                graded_blocks=((0, n - 1),), kickoffs=(0,))
    players = np.full((n, 1, 2), np.nan)
    ball = np.full((n, 2), np.nan)
    log_emit = evidence.log_emissions(FakeTracking(players, ball, [H]), spec,
                                      evidence_params(**overrides))
    return float(np.sum(log_emit[:, D] - log_emit[:, H]))


def test_a_long_untracked_run_counts_as_at_most_the_cap():
    """With a cap, 10 s of lost ball weighs no more towards D than `cap` frames of it."""
    cap = 25
    assert _dead_margin(250, missing_run_cap=cap) == pytest.approx(
        _dead_margin(cap - 2, missing_run_cap=cap), rel=0.02)
    assert _dead_margin(10, missing_run_cap=cap) < _dead_margin(cap - 2, missing_run_cap=cap)


def test_no_cap_counts_every_untracked_frame():
    """Uncapped, the evidence grows with every frame: per frame it is the same at any length."""
    per_frame_long = _dead_margin(250, missing_run_cap=None) / 252
    per_frame_short = _dead_margin(25, missing_run_cap=None) / 27
    assert per_frame_long == pytest.approx(per_frame_short)


def test_players_median_speed_and_rolling_mean():
    speed = np.array([[1.0, 3.0, np.nan], [np.nan, np.nan, np.nan], [2.0, 4.0, 6.0]])
    tracking = FakeTracking(np.zeros((3, 3, 2)), np.zeros((3, 2)), [H, H, A])
    tracking.player_speed = speed
    assert features.players_median_speed(tracking, 0)[[0, 2]].tolist() == [2.0, 4.0]
    assert np.isnan(features.players_median_speed(tracking, 0)[1])
    assert features.players_median_speed(tracking, 1).tolist() == [2.0, 3.0, 4.0]


def _walking_tracking(speeds):
    """One player always next to a visible ball, at a constant speed per stretch of frames."""
    n = len(speeds)
    tracking = FakeTracking([[[50, 25]]] * n, [[50.5, 25]] * n, [H])
    tracking.player_speed = np.asarray(speeds, dtype=float)[:, None]
    return tracking


def test_slow_players_favour_dead_and_fast_players_alive():
    spec = spec_module.GameSpec(n_frames=200, fps=FPS, pitch=(100.0, 50.0),
                                graded_blocks=((0, 199),), kickoffs=(0,))
    tracking = _walking_tracking([0.5] * 100 + [4.0] * 100)
    with_cue = evidence.log_emissions(tracking, spec, evidence_params())
    without = evidence.log_emissions(tracking, spec, evidence_params(slow_speed=None))
    shift = (with_cue[:, D] - with_cue[:, H]) - (without[:, D] - without[:, H])
    assert shift[0] == pytest.approx(np.log(0.62 / 0.10))     # walking: towards D
    assert shift[-1] == pytest.approx(np.log(0.38 / 0.90))    # running: away from D
    np.testing.assert_allclose(with_cue[:, H] - with_cue[:, A], without[:, H] - without[:, A])


def test_slow_cue_is_unknown_where_no_player_is_tracked():
    spec = spec_module.GameSpec(n_frames=200, fps=FPS, pitch=(100.0, 50.0),
                                graded_blocks=((0, 199),), kickoffs=(0,))
    tracking = _walking_tracking([np.nan] * 200)
    np.testing.assert_array_equal(evidence.log_emissions(tracking, spec, evidence_params()),
                                  evidence.log_emissions(tracking, spec,
                                                         evidence_params(slow_speed=None)))


def test_no_slow_speed_ignores_the_players_speed():
    """slow_speed = None is E001: the evidence does not depend on how fast anyone runs."""
    spec = spec_module.GameSpec(n_frames=200, fps=FPS, pitch=(100.0, 50.0),
                                graded_blocks=((0, 199),), kickoffs=(0,))
    walking = evidence.log_emissions(_walking_tracking([0.5] * 200), spec,
                                     evidence_params(slow_speed=None))
    unknown = evidence.log_emissions(_walking_tracking([np.nan] * 200), spec,
                                     evidence_params(slow_speed=None))
    np.testing.assert_array_equal(walking, unknown)


def test_gap_team_cue_only_on_untracked_frames():
    players = [[[10, 10]]] * 4
    ball = [[11, 10], [np.nan, np.nan], [np.nan, np.nan], [np.nan, np.nan]]
    tracking = FakeTracking(players, ball, [A])
    p_home = np.array([0.9, 0.9, 0.6, np.nan])        # confident H, weak H, unknown
    with_cue = evidence.log_emissions(tracking, SPEC, evidence_params(), p_home=p_home)
    without = evidence.log_emissions(tracking, SPEC, evidence_params())
    shift = with_cue - without
    np.testing.assert_array_equal(shift[[0, 3]], 0.0)  # ball seen; classifier unknown
    assert shift[1, H] - shift[1, A] == pytest.approx(np.log(0.86 / 0.14))
    assert shift[2, H] - shift[2, A] == pytest.approx(np.log(0.56 / 0.44))
    assert shift[1, D] == pytest.approx(np.log(0.5))


def test_gap_team_cue_favours_away_below_one_half():
    tracking = FakeTracking([[[10, 10]]] * 4, [[np.nan, np.nan]] * 4, [H])
    p_home = np.array([0.1, 0.4, 0.5, 0.5])
    shift = (evidence.log_emissions(tracking, SPEC, evidence_params(), p_home=p_home)
             - evidence.log_emissions(tracking, SPEC, evidence_params()))
    assert shift[0, A] - shift[0, H] == pytest.approx(np.log(0.86 / 0.14))
    assert shift[1, A] > shift[1, H]
    np.testing.assert_array_equal(shift[2:], 0.0)     # exactly 0.5: no side favoured


def test_gap_cue_off_ignores_p_home():
    tracking = FakeTracking([[[10, 10]]] * 4, [[np.nan, np.nan]] * 4, [H])
    off = evidence_params(gap_acc_strong=None)
    np.testing.assert_array_equal(
        evidence.log_emissions(tracking, SPEC, off, p_home=np.full(4, 0.9)),
        evidence.log_emissions(tracking, SPEC, off))


def test_control_cue_replaces_the_nearest_side_where_known():
    players = [[[10, 10]]] * 4
    ball = [[11, 10], [11, 10], [np.nan, np.nan], [30, 10]]   # near H, near H, unseen, far
    tracking = FakeTracking(players, ball, [H])
    control = np.array([A, -1, A, A], dtype=np.int8)
    with_cue = evidence.log_emissions(tracking, SPEC, evidence_params(control_acc=0.87),
                                      control=control)
    without = evidence.log_emissions(tracking, SPEC, evidence_params())
    assert with_cue[0, A] - with_cue[0, H] == pytest.approx(np.log(0.87 / 0.13))  # replaced
    np.testing.assert_array_equal(with_cue[[1, 2]], without[[1, 2]])  # unknown; ball unseen
    assert with_cue[3, A] - with_cue[3, H] == pytest.approx(np.log(0.87 / 0.13))  # far: new cue
    assert with_cue[0, D] == without[0, D]


def test_control_cue_off_ignores_control():
    tracking = FakeTracking([[[10, 10]]] * 4, [[11, 10]] * 4, [H])
    off = evidence_params(control_acc=None)
    np.testing.assert_array_equal(
        evidence.log_emissions(tracking, SPEC, off, control=np.full(4, A, dtype=np.int8)),
        evidence.log_emissions(tracking, SPEC, off))


def test_weight_scales_everything():
    players = [[[10, 10]]] * 4
    ball = [[11, 10], [np.nan, np.nan], [50, -5], [50, -5]]
    tracking = FakeTracking(players, ball, [H])
    full = evidence.log_emissions(tracking, SPEC, evidence_params())
    fifth = evidence.log_emissions(tracking, SPEC, evidence_params(weight=0.2))
    np.testing.assert_allclose(fifth, 0.2 * full)


def test_gap_dead_cue_replaces_the_slow_cue_on_untracked_frames_only():
    """E010c: where the ball is untracked and p_dead is known, the classifier's call stands in
    for the slow cue; elsewhere the slow cue is untouched."""
    spec = spec_module.GameSpec(n_frames=4, fps=FPS, pitch=(100.0, 50.0),
                                graded_blocks=((0, 3),), kickoffs=(0,))
    tracking = _walking_tracking([0.5] * 4)
    tracking.ball = np.array([[50.5, 25], [np.nan, np.nan], [np.nan, np.nan], [np.nan, np.nan]])
    p_dead = np.array([0.9, 0.9, 0.1, np.nan])        # seen; says dead; says alive; unknown
    params = evidence_params(gap_dead_alive=0.02, gap_dead_dead=0.64, weight=1.0)
    on = evidence.log_emissions(tracking, spec, params, p_dead=p_dead)
    off = evidence.log_emissions(tracking, spec, params)
    tilt = (on[:, D] - on[:, H]) - (off[:, D] - off[:, H])
    slow = np.log(0.62 / 0.10)
    assert tilt[0] == 0.0 and tilt[3] == 0.0
    assert tilt[1] == pytest.approx(np.log(0.64 / 0.02) - slow)
    assert tilt[2] == pytest.approx(np.log(0.36 / 0.98) - slow)


def test_gap_dead_cue_off_ignores_p_dead():
    tracking = FakeTracking([[[10, 10]]] * 4, [[np.nan, np.nan]] * 4, [H])
    np.testing.assert_array_equal(
        evidence.log_emissions(tracking, SPEC, evidence_params(), p_dead=np.full(4, 0.9)),
        evidence.log_emissions(tracking, SPEC, evidence_params()))


def _tag(sides, accuracy=0.8):
    side = np.array(sides, dtype=np.int8)
    run = np.where(side >= 0, np.cumsum(np.diff(np.concatenate(([-1], side))) != 0) - 1, -1)
    return side, np.where(side >= 0, accuracy, np.nan), run


TAG_ON = {"tag_acc_attack": 0.8, "tag_acc_defend": 0.8}


def test_tag_cue_only_on_untracked_frames():
    ball = [[11, 10], [np.nan, np.nan], [np.nan, np.nan], [np.nan, np.nan]]
    tracking = FakeTracking([[[10, 10]]] * 4, ball, [A])
    shift = (evidence.log_emissions(tracking, SPEC, evidence_params(**TAG_ON),
                                    tag=_tag([H, H, H, -1]))
             - evidence.log_emissions(tracking, SPEC, evidence_params(**TAG_ON)))
    np.testing.assert_array_equal(shift[[0, 3]], 0.0)  # ball seen; no tag
    assert shift[1, H] - shift[1, A] == pytest.approx(np.log(0.8 / 0.2))
    assert shift[1, D] == pytest.approx(np.log(0.5))


def test_tag_cue_is_capped_per_run():
    tracking = FakeTracking([[[10, 10]]] * 4, [[np.nan, np.nan]] * 4, [H])
    params = evidence_params(tag_cap=2, **TAG_ON)
    shift = (evidence.log_emissions(tracking, SPEC, params, tag=_tag([A, A, A, A]))
             - evidence.log_emissions(tracking, SPEC, params))
    margin = shift[:, A] - shift[:, H]
    np.testing.assert_allclose(margin, 0.5 * np.log(0.8 / 0.2))  # 4 frames share 2 frames' worth
    assert margin.sum() == pytest.approx(2 * np.log(0.8 / 0.2))


def test_tag_cue_unsure_only_skips_confident_classifier_frames():
    tracking = FakeTracking([[[10, 10]]] * 4, [[np.nan, np.nan]] * 4, [H])
    params = evidence_params(tag_unsure_only=True, tag_cap=100, **TAG_ON)
    p_home = np.array([0.9, 0.6, np.nan, 0.1])
    shift = (evidence.log_emissions(tracking, SPEC, params, p_home=p_home, tag=_tag([A] * 4))
             - evidence.log_emissions(tracking, SPEC, params, p_home=p_home))
    np.testing.assert_array_equal(shift[[0, 3]], 0.0)  # confident either way
    assert (shift[[1, 2], A] > shift[[1, 2], H]).all()


# -- E016: the loose-ball state ------------------------------------------------------------------

LOOSE = dict(PARAMS["decoder"], loose_s=1.0)


def test_loose_transition_rows_are_distributions_and_turnovers_go_through_l():
    log_trans = decoder.transition_matrix(LOOSE, FPS)
    assert log_trans.shape == (4, 4)
    assert np.allclose(np.exp(log_trans).sum(axis=1), 1.0)
    trans = np.exp(log_trans)
    assert trans[H, A] == 0 and trans[A, H] == 0 and trans[D, decoder.L] == 0
    assert trans[decoder.L, D] == 0
    assert trans[H, decoder.L] == pytest.approx(np.exp(LOG_TRANS[H, A]))  # E005c's switch rate
    assert trans[decoder.L, H] == pytest.approx(0.5 / FPS)


def test_loose_off_is_the_three_state_decoder():
    assert PARAMS["decoder"]["loose_s"] is None
    assert np.array_equal(decoder.transition_matrix(PARAMS["decoder"], FPS), LOG_TRANS)


@pytest.mark.parametrize("seed", range(10))
def test_four_state_viterbi_equals_brute_force(seed):
    rng = np.random.default_rng(seed)
    log_emit = rng.normal(size=(6, 4))
    log_trans = np.log(rng.dirichlet(np.ones(4), size=4))
    log_init = np.log(rng.dirichlet(np.ones(4)))
    best, best_path = -np.inf, None
    for path in itertools.product(range(4), repeat=6):
        score = log_init[path[0]] + log_emit[0, path[0]]
        for t in range(1, 6):
            score += log_trans[path[t - 1], path[t]] + log_emit[t, path[t]]
        if score > best:
            best, best_path = score, path
    assert np.array_equal(decoder.viterbi(log_emit, log_trans, log_init), best_path)


@pytest.mark.parametrize("path, expected", [
    ([H, 3, 3, 3, 3, A], [H, H, H, A, A, A]),        # between sides: split at the midpoint
    ([H, 3, 3, 3, A], [H, H, A, A, A]),               # odd length: the later side gets the extra
    ([A, 3, 3, A], [A, A, A, A]),                      # one side either end
    ([D, 3, 3, H], [D, H, H, H]),                      # D before: the side after
    ([H, 3, 3], [H, H, H]),                            # half's edge after
    ([D, 3, D], [D, X, D]),                            # no side at either end
])
def test_resolve_loose(path, expected):
    assert np.array_equal(decoder.resolve_loose(np.array(path, dtype=np.int8)), expected)


def test_loose_column_is_d_team_terms_with_alive_dead_terms():
    """L scores the team cue like D and the dead/alive cues like H and A."""
    tracking = _walking_tracking([0.5] * 4)
    tracking.ball = np.array([[50.5, 25], [np.nan, np.nan], [np.nan, np.nan], [50.5, 25]])
    params = evidence_params()
    three = evidence.log_emissions(tracking, SPEC, params)
    four = evidence.log_emissions(tracking, SPEC, params, loose=True)
    assert np.array_equal(four[:, :3], three)
    ball_only = evidence_params(near_m=0.0, mid_m=0.0)  # no team cue anywhere
    no_team = evidence.log_emissions(tracking, SPEC, ball_only, loose=True)
    assert np.allclose(no_team[:, decoder.L], no_team[:, H])
    team_only = evidence.log_emissions(tracking, SPEC, params, loose=True)[:, decoder.L] \
        - no_team[:, decoder.L]
    assert np.allclose(team_only, (three[:, D] - evidence.log_emissions(
        tracking, SPEC, ball_only)[:, D]))


def test_loose_decoder_writes_no_l_and_runs_on_synthetic(synthetic):
    from possession import atd, pipeline

    data_dir, _ = synthetic
    spec = spec_module.load(data_dir)
    tracking = atd.load(spec, data_dir)
    params = copy.deepcopy(PARAMS)
    params["decoder"]["loose_s"] = 1.0
    codes = pipeline.label(tracking, spec, params)
    assert set(np.unique(codes)) <= {H, A, D, X}
    assert np.array_equal(codes, pipeline.label(tracking, spec, params))


def test_loose_run_variants_change_only_what_e016_says():
    import loose_run

    out = loose_run.variants(PARAMS)
    (alone, _), (with_cue, _) = out.values()
    assert alone["decoder"]["loose_s"] == 1.0 and with_cue["decoder"]["loose_s"] == 1.0
    assert alone["evidence"] == PARAMS["evidence"]
    assert (with_cue["evidence"]["gap_alive_alive"], with_cue["evidence"]["gap_alive_dead"]) \
        == (0.98, 0.31)
    assert with_cue["evidence"]["gap_dead_recipe"] == "lang"
    assert PARAMS["decoder"]["loose_s"] is None
