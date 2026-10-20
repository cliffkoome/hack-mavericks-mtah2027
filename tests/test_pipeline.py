"""The pipeline end to end on a synthetic bundle, and its equivalence to the starter baseline."""

import os
import subprocess
import sys

import numpy as np
import pytest

import baseline
import games
import verify_baseline
from config import PARAMS
from possession import atd, features, pipeline, smoothing, spec as spec_module, submission

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope="module")
def starter_team(synthetic):
    """The starter baseline's per-frame nearest team, straight from its own parser."""
    out, spec = synthetic
    paths = games.asset_paths(data_dir=out)
    return baseline.nearest_player_labels(paths["atd_raw"],
                                          baseline.team_of_each_track(paths["atd_meta"]),
                                          spec["n_video_frames"], spec["pitch"])


def test_nearest_team_matches_the_starter_frame_for_frame(synthetic, starter_team):
    out, _ = synthetic
    ours = features.nearest_team(atd.load(spec_module.load(out), out))
    assert np.array_equal(ours, starter_team)


@pytest.mark.parametrize("window", [0, 25, 50, 125])
def test_majority_vote_matches_the_starter(starter_team, window):
    starter = baseline.smooth(starter_team, window)
    ours = smoothing.majority_vote(starter_team, window)
    assert np.array_equal(np.array(list("HAD"))[ours], starter)


def test_verify_baseline_reports_identical_on_starter_output(synthetic, starter_team, tmp_path):
    """The real-data check, run on synthetic data: starter output vs ours must be IDENTICAL."""
    out, _ = synthetic
    for name, window in verify_baseline.VARIANTS:
        baseline.write_submission(str(tmp_path / (name + ".csv")),
                                  baseline.smooth(starter_team, window))
    assert verify_baseline.compare(out, str(tmp_path)) == {
        name: 0 for name, _ in verify_baseline.VARIANTS}


def test_pipeline_output_is_valid_and_deterministic(synthetic, tmp_path):
    out, _ = synthetic
    first, spec = pipeline.build(PARAMS, out)
    second, _ = pipeline.build(PARAMS, out)
    assert np.array_equal(first, second)
    path = str(tmp_path / "hack_mavericks.csv")
    submission.write(path, first)
    assert submission.validate(path, spec) is None


def test_run_pipeline_cli(synthetic, tmp_path):
    out, _ = synthetic
    target = str(tmp_path / "cli.csv")
    result = subprocess.run([sys.executable, os.path.join(ROOT, "run_pipeline.py"),
                             "--data-dir", out, "--out", target],
                            capture_output=True, text=True, cwd=ROOT)
    assert result.returncode == 0, result.stderr
    assert '"valid": true' in result.stdout
    assert os.path.isfile(target)
