"""Masked-ball harness (possession/masking.py): synthetic data and hand-built labels only."""

import numpy as np
import pytest

from config import PARAMS
from possession import H, A, D, X, atd, features, masking, pipeline, spec as spec_module

FPS = 25
MID_M = PARAMS["evidence"]["mid_m"]


@pytest.fixture(scope="module")
def loaded(synthetic):
    data_dir, _ = synthetic
    spec = spec_module.load(data_dir)
    tracking = atd.load(spec, data_dir)
    return tracking, spec, pipeline.label(tracking, spec, PARAMS)


def test_mask_hides_only_seen_alive_frames_up_to_the_real_share(loaded):
    tracking, spec, reference = loaded
    masked, hidden = masking.mask_ball(tracking, reference, spec, 0, MID_M)
    visible = features.ball_visible(tracking)
    alive = masking.decoded_mask(spec) & ((reference == H) | (reference == A))
    assert visible[hidden].all()                       # only frames where the ball was seen
    assert masking.decoded_mask(spec)[hidden].all()     # never outside a decoded half
    assert np.isnan(masked.ball[hidden]).all()
    np.testing.assert_array_equal(masked.ball[~hidden], tracking.ball[~hidden])
    assert hidden.sum() >= (~visible[alive]).mean() * alive.sum() - 1
    assert masked.players is tracking.players           # only the ball changes


def test_mask_is_deterministic_per_seed(loaded):
    tracking, spec, reference = loaded
    _, first = masking.mask_ball(tracking, reference, spec, 3, MID_M)
    _, again = masking.mask_ball(tracking, reference, spec, 3, MID_M)
    _, other = masking.mask_ball(tracking, reference, spec, 4, MID_M)
    np.testing.assert_array_equal(first, again)
    assert (first != other).any()


def test_gap_lengths_keep_only_gaps_mostly_alive():
    visible = np.array([1, 0, 0, 0, 1, 0, 0, 0, 0, 1], dtype=bool)
    reference = np.array([H, H, H, D, H, D, D, D, H, H])
    decoded = np.ones(10, dtype=bool)
    assert masking.gap_lengths(visible, reference, decoded).tolist() == [3]


SPEC = spec_module.GameSpec(n_frames=400, fps=FPS, pitch=(100.0, 50.0),
                            graded_blocks=((0, 399),), kickoffs=(0,))


def test_compare_identical_codes_is_perfect():
    truth = np.array([H] * 100 + [A] * 100 + [D] * 50 + [H] * 150)
    hidden = np.zeros(400, dtype=bool)
    hidden[80:130] = True
    result = masking.compare(truth, truth.copy(), hidden, SPEC, FPS)
    assert result["n_changes"] == 1
    assert result["change_recall"] == result["mean_iou"] == result["agreement"] == 1.0
    assert result["spell_f1"] == 1.0


def test_compare_counts_a_change_lost_in_a_gap():
    truth = np.array([H] * 200 + [A] * 200)
    carried = np.array([H] * 400)                       # the decoder carried H through the gap
    hidden = np.zeros(400, dtype=bool)
    hidden[150:260] = True
    result = masking.compare(truth, carried, hidden, SPEC, FPS)
    assert result["n_changes"] == 1 and result["change_recall"] == 0.0
    assert result["n_touched"] == 2
    assert result["mean_iou"] == pytest.approx(0.25)    # H: 200/400 overlap, A: none
    assert result["agreement"] == pytest.approx(50 / 110)
    assert result["wrong_side"] == pytest.approx(60 / 110) and result["to_dead"] == 0.0


def test_compare_splits_losses_into_dead_and_wrong_side():
    truth = np.array([H] * 400)
    codes = np.array([H] * 100 + [D] * 30 + [A] * 20 + [H] * 250)
    hidden = np.zeros(400, dtype=bool)
    hidden[100:200] = True
    result = masking.compare(truth, codes, hidden, SPEC, FPS)
    assert result["to_dead"] == pytest.approx(0.3)
    assert result["wrong_side"] == pytest.approx(0.2)
    assert result["agreement"] == pytest.approx(0.5)


def test_compare_ignores_changes_far_from_any_hidden_frame():
    truth = np.array([H] * 200 + [A] * 200)
    hidden = np.zeros(400, dtype=bool)
    hidden[300:350] = True                              # > 1 s after the change at 200
    assert masking.compare(truth, truth.copy(), hidden, SPEC, FPS)["n_changes"] == 0


def test_compare_skips_frames_the_truth_does_not_grade():
    truth = np.array([X] * 100 + [H] * 300)
    hidden = np.zeros(400, dtype=bool)
    hidden[:150] = True
    assert masking.compare(truth, truth.copy(), hidden, SPEC, FPS)["n_hidden"] == 50


def test_start_weights_favour_crowds_and_side_changes():
    n = 400
    players = np.full((n, 4, 2), np.nan)
    ball = np.tile([50.0, 25.0], (n, 1))
    players[:, 0] = [50.5, 25.0]
    players[:100, 1:] = [[51.0, 25.0], [49.0, 25.0], [50.0, 26.0]]   # crowd on frames 0..99

    class Tracking:
        pass
    tracking = Tracking()
    tracking.players, tracking.ball = players, ball
    reference = np.array([H] * 300 + [A] * 100)
    weights = masking.start_weights(tracking, reference, SPEC, MID_M)
    assert weights[0] == masking.CROWD_WEIGHT
    assert weights[150] == 1.0
    assert weights[300] == masking.CHANGE_WEIGHT
    assert weights[300 - 2 * FPS - 1] == 1.0
