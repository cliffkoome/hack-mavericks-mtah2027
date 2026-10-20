"""What we believe the ranking metric does, checked against the scorer on the public ground truth.

If one of these fails, a design decision built on it is wrong -- fix the belief before the code.
"""

import json
import os

import numpy as np
import pytest

import possession_metrics as pm
from possession import H, A, D, X, spec as spec_module, submission

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GT_PATH = os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")


@pytest.fixture(scope="module")
def gt():
    return pm.label_codes(pm.read_label_csv(GT_PATH))


def spell(gt, pred):
    return pm.evaluate(gt, pred)["spell_f1_mean"]


def test_perfect_is_one(gt):
    result = pm.evaluate(gt, gt.copy())
    assert result["spell_f1_mean"] == 1.0 and result["macro_f1"] == 1.0


def test_one_frame_late_is_free(gt):
    assert spell(gt, np.roll(gt, 1)) == 1.0


def test_a_second_late_is_not(gt):
    assert spell(gt, np.roll(gt, 25)) < 0.9


def test_constant_guess_scores_zero(gt):
    for code in (H, A, D):
        assert spell(gt, np.full(len(gt), code, dtype=np.int8)) == 0.0


def test_abstaining_inside_a_possession_is_free_on_spellf1_only(gt):
    runs = pm.valid_runs(gt)
    longest = max(pm.possession_spells(gt, runs), key=lambda s: s[1] - s[0])
    pred = gt.copy()
    pred[longest[0] + 10:longest[1] - 10] = X
    result = pm.evaluate(gt, pred)
    assert result["spell_f1_mean"] == 1.0
    assert result["macro_f1"] < 1.0


def test_abstaining_across_a_hand_over_carries_the_old_label(gt):
    runs = pm.valid_runs(gt)
    handover = next(t for t in pm.transitions(gt, runs) if {t[1], t[2]} == {H, A})
    pred = gt.copy()
    pred[handover[0] - 5:handover[0] + 150] = X
    assert spell(gt, pred) < 1.0


def test_summary_matches_the_published_public_window_stats(gt):
    with open(GT_PATH[:-len(".csv")] + "_meta.json") as handle:
        meta = json.load(handle)
    public = spec_module.GameSpec(n_frames=len(gt), fps=25, pitch=(105.3, 68.0),
                                  graded_blocks=tuple((a, b - 1) for a, b in pm.valid_runs(gt)),
                                  kickoffs=(11185,))
    stats = submission.summary(gt, public)
    assert stats["n_possessions"] == meta["n_possessions"] == 17
    assert stats["mean_possession_s"] == round(meta["mean_possession_duration_s"], 2)
