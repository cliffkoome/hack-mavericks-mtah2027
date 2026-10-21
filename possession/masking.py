"""Hide the ball where it is seen, to measure what the pipeline loses in a gap (E005a).

The tracker loses the ball on 17% of the frames we call alive, and inside those gaps the decoder
has no team evidence. The public window holds too few such frames to say how much that costs, and
the full match has no ground truth. So: take frames where the ball *is* seen, hide it in stretches
shaped like the real gaps, run the pipeline again, and compare with the unmasked run. The unmasked
labels on those frames are the best label we have there (the ball was seen), so the comparison
measures how much worse the pipeline gets without the ball -- not how good it is.

Gaps are placed where real ones tend to happen: lengths drawn from the real gaps in play, starts
biased towards crowds and towards the seconds around a change of side. Everything here is
deterministic for a given seed.
"""

import dataclasses

import numpy as np

from possession import H, A, D, X, features
import possession_metrics as pm

# Params
CROWD_PLAYERS = 3       # this many players within crowd_m of the ball make a crowd
CROWD_WEIGHT = 2.0      # start weight multiplier in a crowd
CHANGE_WINDOW_S = 2.0   # seconds either side of a change of side
CHANGE_WEIGHT = 2.0     # start weight multiplier near a change of side
ALIVE_SHARE = 0.5       # a real gap counts as "in play" if more than this share of it is alive


def decoded_mask(spec):
    """Frames the decoder labels: each half from its kickoff to its last graded frame."""
    mask = np.zeros(spec.n_frames, dtype=bool)
    for (_, last), kickoff in zip(spec.graded_blocks, spec.kickoffs):
        mask[kickoff:last + 1] = True
    return mask


def _runs(flag):
    """[(start, stop)] half-open runs of True."""
    edges = np.diff(np.concatenate(([0], np.asarray(flag, dtype=np.int8), [0])))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def gap_lengths(visible, reference, decoded):
    """Lengths (frames) of the real untracked runs that are mostly alive in the reference."""
    alive = (reference == H) | (reference == A)
    return np.array([stop - start for start, stop in _runs(~visible & decoded)
                     if alive[start:stop].mean() > ALIVE_SHARE], dtype=np.int64)


def side_changes(codes, spec):
    """Frames where the label goes straight from one side to the other (H->A or A->H)."""
    return np.array([frame for frame, old, new in pm.transitions(codes, _graded_runs(spec))
                     if {old, new} == {H, A}], dtype=np.int64)


def start_weights(tracking, reference, spec, crowd_m):
    """Per-frame weight for starting a gap: zero unless the ball is seen and the reference is
    alive; x CROWD_WEIGHT in a crowd, x CHANGE_WEIGHT within CHANGE_WINDOW_S of a side change."""
    visible = features.ball_visible(tracking)
    alive = (reference == H) | (reference == A)
    gap = tracking.players - tracking.ball[:, None, :]
    with np.errstate(invalid="ignore"):
        crowd = (np.hypot(gap[..., 0], gap[..., 1]) < crowd_m).sum(axis=1) >= CROWD_PLAYERS
    near = np.zeros(spec.n_frames, dtype=bool)
    reach = int(round(CHANGE_WINDOW_S * spec.fps))
    for frame in side_changes(reference, spec):
        near[max(0, frame - reach):frame + reach + 1] = True
    eligible = visible & alive & decoded_mask(spec)
    return (eligible * np.where(crowd, CROWD_WEIGHT, 1.0) * np.where(near, CHANGE_WEIGHT, 1.0))


def mask_ball(tracking, reference, spec, seed, crowd_m):
    """(masked tracking, hidden): a copy with the ball hidden on `hidden` frames.

    Gaps are added one at a time until the hidden frames reach the real share of untracked frames
    among the reference's alive frames, so the masked run sees about twice the real gap density.
    Each gap starts on a seen, alive frame (weighted as in start_weights), takes a length from the
    real gaps in play, and stops at the end of its half.
    """
    visible = features.ball_visible(tracking)
    decoded = decoded_mask(spec)
    alive = decoded & ((reference == H) | (reference == A))
    lengths = gap_lengths(visible, reference, decoded)
    if not len(lengths):
        raise ValueError("no untracked runs in play to copy gap lengths from")
    target = int(round((~visible[alive]).mean() * alive.sum()))
    weights = start_weights(tracking, reference, spec, crowd_m)
    half_end = np.zeros(spec.n_frames, dtype=np.int64)
    for (_, last), kickoff in zip(spec.graded_blocks, spec.kickoffs):
        half_end[kickoff:last + 1] = last + 1

    rng = np.random.default_rng(seed)
    hidden = np.zeros(spec.n_frames, dtype=bool)
    while hidden.sum() < target:
        total = np.cumsum(weights)
        if total[-1] <= 0:
            break  # nothing seen and alive is left to hide
        start = int(np.searchsorted(total, rng.random() * total[-1], side="right"))
        stop = min(start + int(rng.choice(lengths)), int(half_end[start]))
        hidden[start:stop] |= visible[start:stop]
        weights[start:stop] = 0.0
    ball = tracking.ball.copy()
    ball[hidden] = np.nan
    return dataclasses.replace(tracking, ball=ball), hidden


def compare(truth, codes, hidden, spec, tol):
    """Aggregates of `codes` against `truth` around the hidden frames (graded truth only).

    change_recall  share of truth side changes within tol of a hidden frame that codes also has
                   (one-to-one, matched on the destination side, within tol: the scorer's rule)
    mean_iou       truth possessions touching a hidden frame: mean best same-side IoU in codes
    agreement      share of hidden frames where codes equals truth
    to_dead        share of hidden frames alive in truth that codes calls D
    wrong_side     share of hidden frames alive in truth that codes gives to the other side
    spell_f1       the official spellF1 of codes against truth, over all graded truth frames
    """
    runs = pm.valid_runs(truth)
    filled = pm.fill_abstentions(codes, runs)
    hidden = hidden & (truth != X)
    near = np.convolve(hidden.astype(float), np.ones(2 * tol + 1), mode="same") > 0
    changes = [t for t in pm.transitions(truth, runs) if {t[1], t[2]} == {H, A} and near[t[0]]]
    pairs, _, _ = pm.match_transitions(changes, pm.transitions(filled, runs), tol)

    truth_spells = pm.possession_spells(truth, runs)
    touched = [i for i, (start, stop, _) in enumerate(truth_spells) if hidden[start:stop].any()]
    alive = hidden & ((truth == H) | (truth == A))
    best = np.zeros(len(truth_spells))
    for iou, i, _ in pm.spell_overlaps(truth_spells, pm.possession_spells(filled, runs)):
        best[i] = max(best[i], iou)
    return {
        "change_recall": len(pairs) / len(changes) if changes else float("nan"),
        "n_changes": len(changes),
        "mean_iou": float(best[touched].mean()) if touched else float("nan"),
        "n_touched": len(touched),
        "agreement": float(np.mean(codes[hidden] == truth[hidden])) if hidden.any()
        else float("nan"),
        "to_dead": float(np.mean(codes[alive] == D)) if alive.any() else float("nan"),
        "wrong_side": float(np.mean((codes[alive] <= A) & (codes[alive] != truth[alive])))
        if alive.any() else float("nan"),
        "n_hidden": int(hidden.sum()),
        "n_possessions": pm.possession_stats(filled, runs)["n_spells"],
        "spell_f1": pm.spell_scores(truth, filled, runs)["mean_f1"],
    }


def _graded_runs(spec):
    return [(first, last + 1) for first, last in spec.graded_blocks]
