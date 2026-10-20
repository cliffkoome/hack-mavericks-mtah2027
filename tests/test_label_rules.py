"""Label rules and the submission file contract (SUBMISSION.md)."""

import numpy as np
import pytest

import possession_metrics as pm
from possession import H, A, D, X, rules, spec as spec_module, submission

REAL = spec_module.real()


def test_finalize_writes_x_outside_graded_and_d_before_kickoff():
    out = rules.finalize(np.full(REAL.n_frames, H, dtype=np.int8), REAL)
    graded = REAL.graded_mask()
    assert (out[~graded] == X).all()
    assert int((out == X).sum()) == 40462
    assert (out[110808:113742] == D).all()
    assert (out[113742:190198] == H).all() and (out[11185:90652] == H).all()


def test_finalize_does_not_modify_its_input():
    codes = np.full(REAL.n_frames, A, dtype=np.int8)
    rules.finalize(codes, REAL)
    assert (codes == A).all()


def test_finalize_rejects_the_wrong_length():
    with pytest.raises(ValueError, match="199319"):
        rules.finalize(np.zeros(164488, dtype=np.int8), REAL)


def test_written_file_passes_the_official_validation(tmp_path):
    path = str(tmp_path / "hack_mavericks.csv")
    submission.write(path, rules.finalize(np.full(REAL.n_frames, A, dtype=np.int8), REAL))
    assert submission.validate(path, REAL) is None
    with open(path, "rb") as handle:
        data = handle.read()
    assert data.startswith(b"frame,label\n0,X\n1,X\n")
    assert data.endswith(b"199318,X\n")
    assert data.count(b"\n") == REAL.n_frames + 1
    assert b"\r" not in data


def test_same_labels_same_bytes_same_hash(tmp_path):
    codes = rules.finalize(np.random.default_rng(0).integers(0, 3, REAL.n_frames), REAL)
    first = submission.write(str(tmp_path / "a.csv"), codes)
    second = submission.write(str(tmp_path / "b.csv"), codes)
    assert first == second
    assert (tmp_path / "a.csv").read_bytes() == (tmp_path / "b.csv").read_bytes()


def test_dead_ball_splits_possessions():
    """H H D D H H is two possessions; H H X X H H is one (CHALLENGE.md, EVALUATION.md)."""
    runs = [(0, 6)]
    dead = np.array([H, H, D, D, H, H], dtype=np.int8)
    quiet = np.array([H, H, X, X, H, H], dtype=np.int8)
    assert len(pm.possession_spells(dead, runs)) == 2
    assert len(pm.possession_spells(pm.fill_abstentions(quiet, runs), runs)) == 1


def test_summary_counts_possessions_like_the_scorer():
    spec = spec_module.GameSpec(n_frames=10, fps=25, pitch=(105.3, 68.0),
                                graded_blocks=((1, 8),), kickoffs=(1,))
    codes = np.array([X, H, H, X, H, D, A, A, X, X], dtype=np.int8)
    stats = submission.summary(codes, spec)
    assert stats["n_possessions"] == 2           # H H X H, then A A (X X filled with A)
    assert stats["graded"] == {"H": 3, "A": 2, "D": 1, "X": 2}
