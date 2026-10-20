"""Per-frame evidence about who has the ball and whether it is in play."""

import numpy as np

# Params
NOBODY = -1


def nearest_player(tracking):
    """(team, distance): team (H/A) of the tracked slot nearest the ball and its distance in
    metres; NOBODY and inf where the ball or every player is missing.

    The same decision as starter/baseline.py's nearest_player_labels, arithmetic included:
    squared distance in metres, first slot wins a tie. test_pipeline.py checks the two agree
    frame for frame, which is also what proves atd.load reads the file the way the starter does.
    """
    gap = tracking.players - tracking.ball[:, None, :]
    squared = gap[..., 0] * gap[..., 0] + gap[..., 1] * gap[..., 1]
    seen = ~np.isnan(squared)
    team = np.full(len(squared), NOBODY, dtype=np.int8)
    distance = np.full(len(squared), np.inf)
    answer = seen.any(axis=1)
    masked = np.where(seen, squared, np.inf)[answer]
    nearest = np.argmin(masked, axis=1)
    team[answer] = tracking.slot_team[nearest]
    distance[answer] = np.sqrt(masked[np.arange(len(nearest)), nearest])
    return team, distance


def nearest_team(tracking):
    """Team of the nearest slot only; see nearest_player."""
    return nearest_player(tracking)[0]


def ball_visible(tracking):
    return ~np.isnan(tracking.ball[:, 0])


def run_length(flag):
    """Per frame, the length of the run of True it belongs to; 0 where flag is False."""
    flag = np.asarray(flag, dtype=bool)
    edges = np.diff(np.concatenate(([0], flag.astype(np.int8), [0])))
    starts, stops = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    out = np.zeros(len(flag), dtype=np.int64)
    for start, stop in zip(starts, stops):
        out[start:stop] = stop - start
    return out


def players_median_speed(tracking, window):
    """Median speed (m/s) of the tracked players, averaged over a centred +/- window frames.

    Players stay tracked when the ball does not, so this is a dead-ball cue that survives the
    41% of frames without a ball: in play they jog and sprint, at a stoppage they walk.
    NaN where no player is tracked anywhere in the window.
    """
    speed = tracking.player_speed
    tracked = ~np.isnan(speed)
    filled = np.where(tracked, speed, np.inf)
    count = tracked.sum(axis=1)
    ordered = np.sort(filled, axis=1)
    median = np.full(len(speed), np.nan)
    rows = np.flatnonzero(count > 0)
    low, high = (count[rows] - 1) // 2, count[rows] // 2
    median[rows] = 0.5 * (ordered[rows, low] + ordered[rows, high])
    return rolling_mean(median, window)


def rolling_mean(values, window):
    """Mean over a centred +/- window, ignoring NaN; NaN where the window holds no value."""
    known = ~np.isnan(values)
    total = np.concatenate(([0.0], np.cumsum(np.where(known, values, 0.0))))
    count = np.concatenate(([0], np.cumsum(known)))
    index = np.arange(len(values))
    low = np.maximum(0, index - window)
    high = np.minimum(len(values), index + window + 1)
    n = count[high] - count[low]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, (total[high] - total[low]) / n, np.nan)


def ball_speed(tracking, fps, lag):
    """m/s over the last `lag` frames, NaN unless the ball is seen at both ends."""
    speed = np.full(len(tracking.ball), np.nan)
    step = tracking.ball[lag:] - tracking.ball[:-lag]
    speed[lag:] = np.hypot(step[:, 0], step[:, 1]) * fps / lag
    return speed


def ball_outside(tracking, pitch, margin):
    """True where the ball is seen more than `margin` metres outside the pitch lines."""
    x, y = tracking.ball[:, 0], tracking.ball[:, 1]
    length, width = pitch
    with np.errstate(invalid="ignore"):
        return (x < -margin) | (x > length + margin) | (y < -margin) | (y > width + margin)
