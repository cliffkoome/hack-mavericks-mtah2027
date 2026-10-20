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
"""

import numpy as np

from possession import H, A, D, X

# Params
STATES = 3  # H, A, D


def transition_matrix(params, fps):
    """(3, 3) per-frame log P(to | from), rows = from."""
    end = 1.0 / (params["mean_possession_s"] * fps)
    restart = 1.0 / (params["mean_dead_s"] * fps)
    stop, switch = end * params["stoppage_share"], end * (1 - params["stoppage_share"])
    matrix = np.array([[1 - end, switch, stop],
                       [switch, 1 - end, stop],
                       [restart / 2, restart / 2, 1 - restart]])
    return np.log(matrix)


def viterbi(log_emit, log_trans, log_init):
    """Most likely state path. Ties go to the lowest state index, so the result is deterministic."""
    n = len(log_emit)
    back = np.zeros((n, STATES), dtype=np.int8)
    score = log_init + log_emit[0]
    columns = np.arange(STATES)
    for t in range(1, n):
        candidates = score[:, None] + log_trans
        back[t] = np.argmax(candidates, axis=0)
        score = candidates[back[t], columns] + log_emit[t]
    path = np.empty(n, dtype=np.int8)
    path[-1] = np.argmax(score)
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def decode_match(log_emit, spec, params):
    """Per-frame codes: decoded from each kickoff to the end of its half, X everywhere else.

    Play starts alive at a kickoff, with either side equally likely a priori; the evidence picks.
    """
    log_trans = transition_matrix(params, spec.fps)
    with np.errstate(divide="ignore"):
        log_init = np.log(np.array([0.5, 0.5, 0.0]))
    out = np.full(spec.n_frames, X, dtype=np.int8)
    for (_, last), kickoff in zip(spec.graded_blocks, spec.kickoffs):
        out[kickoff:last + 1] = viterbi(log_emit[kickoff:last + 1], log_trans, log_init)
    return out


assert (H, A, D) == (0, 1, 2), "decoder columns assume H, A, D are 0, 1, 2"
