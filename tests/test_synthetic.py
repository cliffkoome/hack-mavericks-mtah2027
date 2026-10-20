"""The synthetic generator: deterministic, readable by the provided loaders, consistent with itself.

These tests are what lets every other test trust a synthetic bundle as a stand-in for the real one.
"""

import hashlib
import os

import numpy as np

import baseline
import games
import load_homography
import load_smart_tagging
import make_synthetic
import possession_metrics as pm
from conftest import SYNTHETIC_SEED


def _digests(out):
    digests = {}
    for folder, _, files in os.walk(out):
        for name in files:
            path = os.path.join(folder, name)
            with open(path, "rb") as handle:
                digests[os.path.relpath(path, out)] = hashlib.sha256(handle.read()).hexdigest()
    return digests


def _gt(out):
    return pm.label_codes(pm.read_label_csv(os.path.join(out, "synthetic_possession_gt.csv")))


def test_same_seed_same_bytes(synthetic, tmp_path):
    out, _ = synthetic
    make_synthetic.generate(str(tmp_path), seed=SYNTHETIC_SEED, size="small")
    assert _digests(str(tmp_path)) == _digests(out)


def test_different_seed_different_match(synthetic, tmp_path):
    out, _ = synthetic
    make_synthetic.generate(str(tmp_path), seed=SYNTHETIC_SEED + 1, size="small")
    assert _digests(str(tmp_path)) != _digests(out)


def test_full_layout_is_the_real_match():
    """The full-size layout must be the published one, or tests at full size test the wrong thing."""
    spec = make_synthetic.build_spec("full")
    assert spec["n_video_frames"] == games.GAMES["BAR-ATH"]["n_video_frames"] == 199319
    blocks = [(first, last) for _, first, last in games.half_video_blocks("BAR-ATH")]
    assert [tuple(block) for block in spec["graded_blocks"]] == blocks
    assert spec["kickoffs"] == [11185, 113742]


def test_bundle_uses_real_file_names(synthetic):
    out, _ = synthetic
    for name in ("atd_meta", "atd_raw", "homography", "smart_tagging"):
        assert os.path.isfile(games.asset_paths(data_dir=out)[name])


def test_atd_raw_is_one_line_per_frame_from_one(synthetic):
    out, spec = synthetic
    with open(games.asset_paths(data_dir=out)["atd_raw"]) as handle:
        lines = handle.read().splitlines()
    assert len(lines) == spec["n_video_frames"]
    parts = [line.split(":") for line in lines]
    assert [int(p[0]) for p in parts] == list(range(1, spec["n_video_frames"] + 1))
    assert all(len(p) == 3 for p in parts)
    assert all(p[1].count(";") == make_synthetic.N_SLOTS - 1 for p in parts)


def test_starter_baseline_reads_it(synthetic):
    """The organisers' own parser is the format reference available to us."""
    out, spec = synthetic
    paths = games.asset_paths(data_dir=out)
    teams = baseline.team_of_each_track(paths["atd_meta"])
    assert len(teams) == make_synthetic.N_SLOTS
    assert sorted(np.bincount(teams).tolist()) == [23, 23]
    assert baseline.team_names(paths["atd_meta"]) == make_synthetic.ATD_TEAM_NAMES
    codes = baseline.nearest_player_labels(paths["atd_raw"], teams, spec["n_video_frames"],
                                           spec["pitch"])
    assert (codes >= 0).any()


def test_slot_teams_are_not_in_block_order(synthetic):
    """Real code must read teamId per slot; a generator that lists home slots first would hide it."""
    out, _ = synthetic
    teams = baseline.team_of_each_track(games.asset_paths(data_dir=out)["atd_meta"])
    assert not (teams[:23] == teams[0]).all()


def test_homography_loader_reads_it(synthetic):
    out, spec = synthetic
    matrices, error, valid = load_homography.load(data_dir=out)
    assert matrices.shape == (spec["n_video_frames"], 3, 3)
    assert valid.all() and (error > 0).all()
    x, y = load_homography.project(matrices[spec["kickoffs"][0]], 0.5, 0.5)
    assert 0 < x < 1920 and 0 < y < 1080


def test_smart_tagging_loader_reads_it(synthetic):
    out, spec = synthetic
    events = load_smart_tagging.load(data_dir=out)
    assert events
    assert {e["tags"]["Team"] for e in events} <= set(make_synthetic.TAGGING_TEAM_NAMES) | {"N/A"}
    kickoffs = [e for e in events
                if e["tags"].get("Type") == "Kick Off (Start)"]
    for frame in spec["kickoffs"]:
        assert any(e["start_frame"] <= frame <= e["end_frame"] for e in kickoffs)
    throw_ins = [e for e in events if e["tags"].get("Type") == "Throw-in"]
    assert all(e["tags"]["Team"] == "N/A" for e in throw_ins)


def test_ground_truth_is_scorer_compatible(synthetic):
    out, spec = synthetic
    table = pm.read_label_csv(os.path.join(out, "synthetic_possession_gt.csv"))
    assert pm.validate_submission(table, spec["n_video_frames"]) is None
    gt = pm.label_codes(table)
    graded = make_synthetic.graded_mask(spec)
    assert (gt[~graded] == pm.X).all()
    assert (gt[graded] != pm.X).all()
    assert pm.valid_runs(gt) == [(a, b + 1) for a, b in spec["graded_blocks"]]


def test_second_half_starts_dead(synthetic):
    out, spec = synthetic
    gt = _gt(out)
    first = spec["graded_blocks"][1][0]
    assert (gt[first:spec["kickoffs"][1]] == pm.D).all()
    assert gt[spec["kickoffs"][1]] == pm.H


def test_short_possessions_exist(synthetic):
    out, _ = synthetic
    gt = _gt(out)
    lengths = [b - a for a, b, _ in pm.possession_spells(gt, pm.valid_runs(gt))]
    assert min(lengths) < 2 * make_synthetic.FPS


def test_statistics_are_football_like(synthetic):
    _, spec = synthetic
    stats = spec["stats"]
    assert 0.3 < stats["ball_missing_graded"] < 0.55
    assert 18 < stats["mean_objects_graded"] < 24
    assert 5 < stats["mean_possession_duration_s"] < 25


def test_signal_is_learnable(synthetic):
    """Nearest player names the right team most of the time, and smoothing helps spellF1, as on
    the real match. If either fails, the synthetic match is not a useful stand-in."""
    out, spec = synthetic
    paths = games.asset_paths(data_dir=out)
    teams = baseline.team_of_each_track(paths["atd_meta"])
    codes = baseline.nearest_player_labels(paths["atd_raw"], teams, spec["n_video_frames"],
                                           spec["pitch"])
    gt = _gt(out)
    seen = (gt <= pm.A) & (codes >= 0)
    assert (codes[seen] == gt[seen]).mean() > 0.7
    raw = pm.evaluate(gt, pm.label_codes(baseline.smooth(codes, 0)))["spell_f1_mean"]
    smooth = pm.evaluate(gt, pm.label_codes(baseline.smooth(codes, 50)))["spell_f1_mean"]
    assert smooth > raw + 0.1
