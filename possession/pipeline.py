"""The whole chain from data bundle to labels. run_pipeline.py is a thin CLI over build()."""

from possession import atd, features, rules, smoothing, spec as spec_module


def build(params, data_dir=None):
    """(codes, spec): final per-frame label codes for the match in data_dir (default: data/)."""
    spec = spec_module.load(data_dir)
    tracking = atd.load(spec, data_dir)
    team = features.nearest_team(tracking)
    codes = smoothing.majority_vote(team, params["smooth_window"])
    return rules.finalize(codes, spec), spec
