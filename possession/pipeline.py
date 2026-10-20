"""The whole chain from data bundle to labels. run_pipeline.py is a thin CLI over build()."""

from possession import atd, decoder, evidence, features, rules, smoothing, spec as spec_module

MODELS = ("vote", "hmm")


def build(params, data_dir=None):
    """(codes, spec): final per-frame label codes for the match in data_dir (default: data/)."""
    spec = spec_module.load(data_dir)
    tracking = atd.load(spec, data_dir)
    return label(tracking, spec, params), spec


def label(tracking, spec, params):
    """Per-frame codes from loaded tracking, with the fixed rules applied."""
    if params["model"] == "vote":
        codes = smoothing.majority_vote(features.nearest_team(tracking), params["smooth_window"])
    elif params["model"] == "hmm":
        log_emit = evidence.log_emissions(tracking, spec, params["evidence"])
        codes = decoder.decode_match(log_emit, spec, params["decoder"])
    else:
        raise ValueError("unknown model {!r}; have {}".format(params["model"], MODELS))
    return rules.finalize(codes, spec)
