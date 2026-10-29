"""The whole chain from data bundle to labels. run_pipeline.py is a thin CLI over build()."""

from possession import D, atd, dead_model, decoder, evidence, features, rules, smoothing
from possession import team_model
from possession import tagging, touches
from possession import spec as spec_module

MODELS = ("vote", "hmm")


def build(params, data_dir=None):
    """(codes, spec): final per-frame label codes for the match in data_dir (default: data/)."""
    spec = spec_module.load(data_dir)
    tracking = atd.load(spec, data_dir)
    return label(tracking, spec, params, tags_for(params, data_dir)), spec


def tag_cue_on(params):
    """Whether either phase tag group (E012) is on."""
    ev = params["evidence"]
    return ev["tag_acc_attack"] is not None or ev["tag_acc_defend"] is not None


def tags_for(params, data_dir=None):
    """Smart Tagging events when the phase tag cue (E012) is on, else None: off reads no file."""
    return tagging.load(data_dir) if params["model"] == "hmm" and tag_cue_on(params) else None


def label(tracking, spec, params, tags=None):
    """Per-frame codes from loaded tracking, with the fixed rules applied.

    tags: Smart Tagging events (tags_for), required when the phase tag cue is on.
    """
    if params["model"] == "vote":
        codes = smoothing.majority_vote(features.nearest_team(tracking), params["smooth_window"])
    elif params["model"] == "hmm":
        codes = _hmm(tracking, spec, params, tags)
    else:
        raise ValueError("unknown model {!r}; have {}".format(params["model"], MODELS))
    return rules.finalize(codes, spec)


def first_pass(tracking, spec, params):
    """Pass-1 decoder codes: the evidence without any of the cues that need labels."""
    log_emit = evidence.log_emissions(tracking, spec, params["evidence"])
    return decoder.decode_match(log_emit, spec, params["decoder"])


def _hmm(tracking, spec, params, tags=None):
    """Decoder codes. With the gap team cue (E005c) or the control cue (E009) on, two passes:
    pass 1 decodes without them. Its labels train the team-from-movement classifier half-to-half
    (each half predicted by the other half's model, as validated in E005b), and its stoppages
    reset the last-control side. Pass 2 decodes with the cues added, and with the phase tag cue
    (E012), which needs no labels but stays out of pass 1 so the classifier trains as in E005c."""
    ev = params["evidence"]
    first = first_pass(tracking, spec, params)
    gap_dead = ev["gap_dead_alive"] is not None or ev["gap_alive_alive"] is not None
    tag = None
    if tag_cue_on(params):
        if tags is None:
            raise ValueError("the phase tag cue (E012) is on: pass tags=pipeline.tags_for(...)")
        tag = tagging.phase_cue(tags, spec.n_frames, spec.fps, ev)
    if (ev["gap_acc_strong"] is None and ev["control_acc"] is None and not gap_dead
            and tag is None):
        return first
    labels = rules.finalize(first, spec)
    p_home = control = p_dead = None
    if ev["gap_acc_strong"] is not None:
        p_home, _ = team_model.cross_half(tracking, spec, labels, ev["near_m"])
    if gap_dead:
        p_dead, _ = dead_model.cross_half(tracking, spec, labels,
                                          recipe=ev["gap_dead_recipe"])
    if ev["control_acc"] is not None:
        holder, _ = touches.held(tracking, ev["near_m"])
        frames, sides = touches.controls(holder)
        control = touches.last_touch(spec.n_frames, frames, sides,
                                     features.ball_visible(tracking), labels == D, spec.fps)
    log_emit = evidence.log_emissions(tracking, spec, ev, p_home=p_home, control=control,
                                      p_dead=p_dead, tag=tag)
    return decoder.decode_match(log_emit, spec, params["decoder"])
