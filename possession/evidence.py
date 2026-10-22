"""Per-frame log-likelihoods of the three states (H, A, D) from tracking evidence.

Each frame contributes independent cues, each a probability with a plain meaning:

  team      the nearest player's side, trusted more the closer they are to the ball
  visible   whether the tracker sees the ball at all (it loses it more often when it is dead).
            The tracker loses the ball in long bursts, so a run of untracked frames counts as at
            most `missing_run_cap` frames of evidence however long it lasts (None: no cap).
            Uncapped, a few seconds of lost ball in open play can outvote the switching costs;
            capped at 1 s (E002), dead stretches lose nearly all their evidence too.
  outside   whether a seen ball is off the pitch (rare in play, common at a stoppage)
  still     whether a seen ball has stopped moving (a restart spot, or a ball waiting for one)
  slow      whether the players are walking (median speed over +/- 1 s). Players stay tracked
            when the ball does not, so this also argues *against* D in open play with the ball
            lost, where the visible cue alone over-calls it (E003).
  control   where the ball is seen and a side was last in control of it (touches.py, E009):
            that side, trusted at `control_acc`, replaces the nearest-player cue on that frame.
  gap team  where the ball is untracked: the side the team-from-movement classifier favours
            (p_home, team_model.py), trusted by how confident it is (E005c). Frames with the
            ball seen keep the nearest-player cue alone, so no frame has two team cues.
  gap dead  where the ball is untracked: the dead/alive-from-movement classifier's call
            (p_dead, dead_model.py) replaces the slow cue, so no frame has two movement cues
            (E010c, off by default).

The values in config.py come from documented facts about the tracking (DATA.md) and football
priors, never from fitting the public window. Consecutive frames are far from independent, so
everything is multiplied by `weight` < 1 before decoding: without it, a second of the same noisy
reading counts as 25 independent votes and drowns the switching cost in the decoder.
"""

import numpy as np

from possession import H, A, D, features


def log_emissions(tracking, spec, params, p_home=None, control=None, p_dead=None):
    """(n, 3) array: log P(evidence at frame t | state), columns in H, A, D order.

    p_home: per-frame P(H has the ball) from team_model, NaN where unknown; None = no gap cue.
    control: per-frame last-control side (H/A, negative where unknown); None = no control cue.
    p_dead: per-frame P(dead) from dead_model, NaN where unknown; None = no gap dead cue.
    """
    team, distance = features.nearest_player(tracking)
    visible = features.ball_visible(tracking)
    speed = features.ball_speed(tracking, spec.fps, params["speed_lag"])
    outside = features.ball_outside(tracking, spec.pitch, params["outside_margin_m"])
    out = np.zeros((len(team), 3))

    # Team cue. How far to trust the nearest side depends on how near it is. Under D the nearest
    # side carries no information either way, so D scores log 0.5 where the alive states score
    # log(acc) for agreeing with it and log(1 - acc) for not.
    far = np.nan if params["team_acc_far"] is None else params["team_acc_far"]
    accuracy = np.where(distance < params["near_m"], params["team_acc_near"],
                        np.where(distance < params["mid_m"], params["team_acc_mid"], far))
    cue = (team >= 0) & ~np.isnan(accuracy)
    for state in (H, A):
        out[cue, state] = np.where(team[cue] == state, np.log(accuracy[cue]),
                                   np.log(1 - accuracy[cue]))
    out[cue, D] = np.log(0.5)
    if control is not None and params["control_acc"] is not None:
        held = visible & (control >= 0)
        for state in (H, A):
            out[held, state] = np.where(control[held] == state, np.log(params["control_acc"]),
                                        np.log(1 - params["control_acc"]))
        out[held, D] = np.log(0.5)

    missing = ~visible
    burst = np.ones(len(missing))
    if params["missing_run_cap"] is not None:
        run = features.run_length(missing)
        burst = np.where(missing, np.minimum(1.0, params["missing_run_cap"] / np.maximum(run, 1)),
                         1.0)
    alive_dead = (
        _binary(missing, params["missing_alive"], params["missing_dead"]) * burst[:, None]
        + _binary(outside, params["outside_alive"], params["outside_dead"], where=visible)
        + _binary(speed < params["still_speed"], params["still_alive"], params["still_dead"],
                  where=~np.isnan(speed))
    )
    gap_dead = np.zeros(len(missing), dtype=bool)
    if p_dead is not None and params["gap_dead_alive"] is not None:
        gap_dead = missing & ~np.isnan(p_dead)
        alive_dead += _binary(p_dead >= 0.5, params["gap_dead_alive"], params["gap_dead_dead"],
                              where=gap_dead)
    if params["slow_speed"] is not None:
        walking = features.players_median_speed(tracking, spec.fps)
        alive_dead += _binary(walking < params["slow_speed"], params["slow_alive"],
                              params["slow_dead"], where=~np.isnan(walking) & ~gap_dead)
    if p_home is not None and params["gap_acc_strong"] is not None:
        _add_gap_team(out, p_home, missing, params)
    out[:, H] += alive_dead[:, 0]
    out[:, A] += alive_dead[:, 0]
    out[:, D] += alive_dead[:, 1]
    return params["weight"] * out


def _add_gap_team(out, p_home, missing, params):
    """Team cue from the classifier on untracked frames, in the nearest-player cue's form."""
    cue = missing & ~np.isnan(p_home) & (p_home != 0.5)
    sure = (p_home >= params["gap_conf"]) | (p_home <= 1 - params["gap_conf"])
    accuracy = np.where(sure, params["gap_acc_strong"], params["gap_acc_weak"])[cue]
    favoured = np.where(p_home[cue] > 0.5, H, A)
    for state in (H, A):
        out[cue, state] += np.where(favoured == state, np.log(accuracy), np.log(1 - accuracy))
    out[cue, D] += np.log(0.5)


def _binary(flag, p_alive, p_dead, where=None):
    """(n, 2) log-likelihood of a yes/no cue under (alive, dead); zero where the cue is unknown."""
    flag = np.asarray(flag, dtype=bool)
    out = np.column_stack([np.where(flag, np.log(p_alive), np.log(1 - p_alive)),
                           np.where(flag, np.log(p_dead), np.log(1 - p_dead))])
    if where is not None:
        out[~np.asarray(where, dtype=bool)] = 0.0
    return out
