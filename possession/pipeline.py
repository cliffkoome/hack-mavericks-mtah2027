"""The whole chain from data bundle to labels. run_pipeline.py is a thin CLI over build()."""

from possession import atd, decoder, evidence, features, rules, smoothing, team_model
from possession import spec as spec_module

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
        codes = _hmm(tracking, spec, params)
    else:
        raise ValueError("unknown model {!r}; have {}".format(params["model"], MODELS))
    return rules.finalize(codes, spec)


def _hmm(tracking, spec, params):
    """Decoder codes. With the gap team cue on (E005c), two passes: pass 1 without it labels the
    match; those labels train the team-from-movement classifier half-to-half (each half predicted
    by the other half's model, as validated in E005b); pass 2 decodes with the cue added."""
    ev = params["evidence"]
    log_emit = evidence.log_emissions(tracking, spec, ev)
    first = decoder.decode_match(log_emit, spec, params["decoder"])
    if ev["gap_acc_strong"] is None:
        return first
    p_home, _ = team_model.cross_half(tracking, spec, rules.finalize(first, spec), ev["near_m"])
    log_emit = evidence.log_emissions(tracking, spec, ev, p_home=p_home)
    return decoder.decode_match(log_emit, spec, params["decoder"])
