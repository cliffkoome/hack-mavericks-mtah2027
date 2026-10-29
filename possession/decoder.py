"""Segment-level decoding: the most likely H/A/D sequence given per-frame evidence (Viterbi).

Why not a window vote: a vote has no idea that possessions are intervals. It cuts every
possession it is unsure about into pieces (raw baseline: 131 possessions where the public window
has 17) or, widened, swallows short ones whole. A decoder pays a fixed cost per state change and
keeps the change only when the evidence on both sides outweighs it -- so a short possession with
strong evidence survives and a flicker does not, which is exactly what spellF1 rewards.

Transitions are set from interpretable rates (config.py): how long a possession lasts on average,
what share end in a stoppage rather than a turnover, how long a stoppage lasts. Each half is
decoded on its own from its kickoff, so nothing carries across halftime, and the frames before a
kickoff are left to rules.finalize (they are D).

With `loose_s` set (E016), a fourth state L, "loose ball": alive but committed to neither side
(a duel, a ball in flight). Every turnover goes H -> L -> A. Inside a ball gap the side is often
unclear, and without L the only team-neutral state is D, so side changes there went through D
and every alive cue cost side-change recall (E010c/d, E015). L is never written: resolve_loose
gives its frames to the sides around it.
"""

import numpy as np

from possession import H, A, D, X

# Params
STATES = 3  # H, A, D
L = 3       # E016's loose-ball state: an internal column only, never a label


def transition_matrix(params, fps):
    """(3, 3) per-frame log P(to | from), rows = from; (4, 4) with L when loose_s is set."""
    end = 1.0 / (params["mean_possession_s"] * fps)
    restart = 1.0 / (params["mean_dead_s"] * fps)
    stop, switch = end * params["stoppage_share"], end * (1 - params["stoppage_share"])
    if params.get("loose_s") is None:
        matrix = np.array([[1 - end, switch, stop],
                           [switch, 1 - end, stop],
                           [restart / 2, restart / 2, 1 - restart]])
    else:
        leave = 1.0 / (params["loose_s"] * fps)
        matrix = np.array([[1 - end, 0.0, stop, switch],
                           [0.0, 1 - end, stop, switch],
                           [restart / 2, restart / 2, 1 - restart, 0.0],
                           [leave / 2, leave / 2, 0.0, 1 - leave]])
    with np.errstate(divide="ignore"):
        return np.log(matrix)


def viterbi(log_emit, log_trans, log_init):
    """Most likely state path. Ties go to the lowest state index, so the result is deterministic."""
    n = len(log_emit)
    states = len(log_trans)
    back = np.zeros((n, states), dtype=np.int8)
    score = log_init + log_emit[0]
    columns = np.arange(states)
    for t in range(1, n):
        candidates = score[:, None] + log_trans
        back[t] = np.argmax(candidates, axis=0)
        score = candidates[back[t], columns] + log_emit[t]
    path = np.empty(n, dtype=np.int8)
    path[-1] = np.argmax(score)
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def resolve_loose(path):
    """Codes from a path with L: an L run between H and A is split at its midpoint (the side
    before, then the side after); next to one side only, or between two frames of one side, it
    takes that side; with no side at either end, X."""
    out = path.copy()
    edges = np.diff(np.concatenate(([0], (path == L).astype(np.int8), [0])))
    for start, stop in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        before = path[start - 1] if start > 0 else X
        after = path[stop] if stop < len(path) else X
        sides = [side for side in (before, after) if side in (H, A)]
        if not sides:
            out[start:stop] = X
        elif before == after or len(sides) == 1:
            out[start:stop] = sides[0]
        else:
            middle = start + (stop - start) // 2
            out[start:middle], out[middle:stop] = before, after
    return out


def decode_match(log_emit, spec, params):
    """Per-frame codes: decoded from each kickoff to the end of its half, X everywhere else.

    Play starts alive at a kickoff, with either side equally likely a priori; the evidence picks.
    log_emit has a fourth column, L's, when loose_s is set.
    """
    log_trans = transition_matrix(params, spec.fps)
    init = np.zeros(len(log_trans))
    init[[H, A]] = 0.5
    with np.errstate(divide="ignore"):
        log_init = np.log(init)
    out = np.full(spec.n_frames, X, dtype=np.int8)
    for (_, last), kickoff in zip(spec.graded_blocks, spec.kickoffs):
        path = viterbi(log_emit[kickoff:last + 1], log_trans, log_init)
        out[kickoff:last + 1] = resolve_loose(path) if len(log_trans) > STATES else path
    return out


assert (H, A, D) == (0, 1, 2), "decoder columns assume H, A, D are 0, 1, 2"
