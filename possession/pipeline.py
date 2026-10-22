"""The whole chain from data bundle to labels. run_pipeline.py is a thin CLI over build()."""

from possession import D, atd, dead_model, decoder, evidence, features, rules, smoothing
from possession import team_model
from possession import touches
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
    """Decoder codes. With the gap team cue (E005c) or the control cue (E009) on, two passes:
    pass 1 decodes without them. Its labels train the team-from-movement classifier half-to-half
    (each half predicted by the other half's model, as validated in E005b), and its stoppages
    reset the last-control side. Pass 2 decodes with the cues added."""
    ev = params["evidence"]
    log_emit = evidence.log_emissions(tracking, spec, ev)
    first = decoder.decode_match(log_emit, spec, params["decoder"])
    gap_dead = ev["gap_dead_alive"] is not None or ev["gap_alive_alive"] is not None
    if ev["gap_acc_strong"] is None and ev["control_acc"] is None and not gap_dead:
        return first
    labels = rules.finalize(first, spec)
    p_home = control = p_dead = None
    if ev["gap_acc_strong"] is not None:
        p_home, _ = team_model.cross_half(tracking, spec, labels, ev["near_m"])
    if gap_dead:
        p_dead, _ = dead_model.cross_half(tracking, spec, labels)
    if ev["control_acc"] is not None:
        holder, _ = touches.held(tracking, ev["near_m"])
        frames, sides = touches.controls(holder)
        control = touches.last_touch(spec.n_frames, frames, sides,
                                     features.ball_visible(tracking), labels == D, spec.fps)
    log_emit = evidence.log_emissions(tracking, spec, ev, p_home=p_home, control=control,
                                      p_dead=p_dead)
    return decoder.decode_match(log_emit, spec, params["decoder"])
