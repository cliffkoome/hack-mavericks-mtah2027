"""Per-frame evidence about who has the ball."""

import numpy as np

# Params
NOBODY = -1


def nearest_team(tracking):
    """Team (H/A) of the tracked slot nearest the ball, NOBODY where the ball or every player is
    missing.

    The same decision as starter/baseline.py's nearest_player_labels, arithmetic included:
    squared distance in metres, first slot wins a tie. test_pipeline.py checks the two agree
    frame for frame, which is also what proves atd.load reads the file the way the starter does.
    """
    gap = tracking.players - tracking.ball[:, None, :]
    distance = gap[..., 0] * gap[..., 0] + gap[..., 1] * gap[..., 1]
    seen = ~np.isnan(distance)
    out = np.full(len(distance), NOBODY, dtype=np.int8)
    answer = seen.any(axis=1)
    nearest = np.argmin(np.where(seen, distance, np.inf)[answer], axis=1)
    out[answer] = tracking.slot_team[nearest]
    return out
