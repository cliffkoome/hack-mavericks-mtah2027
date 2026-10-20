"""Majority vote over a centred window: the reference baseline's step 2, kept as our floor.

Identical to starter/baseline.py's smooth() including its tie rule (home wins), but returning
label codes rather than letters, so it composes with the rest of the pipeline.
"""

import numpy as np

from possession import H, A, D


def majority_vote(team, window):
    """Codes H/A/D: per frame, the majority of H vs A within +/- window frames, D if neither seen.

    team is features.nearest_team output: H, A, or NOBODY for no answer.
    """
    if window <= 0:
        return np.where(team == H, H, np.where(team == A, A, D)).astype(np.int8)
    home = np.concatenate(([0], np.cumsum(team == H)))
    away = np.concatenate(([0], np.cumsum(team == A)))
    index = np.arange(len(team))
    low = np.maximum(0, index - window)
    high = np.minimum(len(team), index + window + 1)
    home, away = home[high] - home[low], away[high] - away[low]
    out = np.full(len(team), D, dtype=np.int8)
    seen = (home + away) > 0
    out[seen] = np.where(home[seen] >= away[seen], H, A)
    return out
