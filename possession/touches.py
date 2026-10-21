"""Ball touches or control, and the side that last had the ball, from the seen ball (E006, E009).

The ground truth names an owner, and the owner plausibly stays with the side of the last
controlled touch while the ball travels (Vidal-Codina et al. 2022; Link & Hoernig 2017). A touch
shows as a sharp change in the ball's velocity next to a player. Our ball is automatic tracking
with jitter, so the papers' thresholds do not transfer: the threshold here is the velocity change
the ball shows in free flight (no player within FREE_FLIGHT_M), measured on the match itself.

E006 (stopped): on this tracking a touch is no sharper than free flight, so touches are hardly
ever found. E009 uses proximity instead: a side *holds* the ball when its player is within reach
and no opponent is, and *controls* it after more than CONTROL_FRAMES held frames in a row. last_touch()
carries either one forward.

Rules: a touch needs a player within reach; with players of both sides within reach it is a duel
and attributes nothing. The last touch's side carries forward while the ball stays seen; it resets
to unknown after an untracked run longer than RESET_GAP_S and on every stoppage.
"""

import numpy as np

from possession import H, A, features

# Params
SAVGOL = np.array([-2.0, 3.0, 6.0, 7.0, 6.0, 3.0, -2.0]) / 21.0  # 7 frames, order 2
DV_STEP = 3              # frames either side for the velocity change (0.12 s)
REACH_M = 1.0            # possession zone (Vidal-Codina et al.)
FREE_FLIGHT_M = 5.0      # nearest player farther than this: the ball is in free flight
FREE_FLIGHT_PCT = 99.0   # a touch must beat this percentile of the free-flight velocity change
RESET_GAP_S = 1.0        # an untracked run longer than this forgets the last touch
CONTROL_FRAMES = 10      # held longer than this = control: a >= 10 m/s fly-by crosses a 2 m disc
                         # in at most 10 frames
NOBODY = -1


def _runs(flag):
    edges = np.diff(np.concatenate(([0], np.asarray(flag, dtype=np.int8), [0])))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def smooth(ball):
    """Savitzky-Golay smoothing inside each run of seen frames; the 3 frames at each end of a run,
    and runs shorter than the filter, keep their raw positions. NaN stays NaN."""
    out = ball.copy()
    half = len(SAVGOL) // 2
    for start, stop in _runs(~np.isnan(ball[:, 0])):
        if stop - start < len(SAVGOL):
            continue
        for axis in (0, 1):
            out[start + half:stop - half, axis] = np.convolve(ball[start:stop, axis], SAVGOL,
                                                              mode="valid")
    return out


def velocity_change(ball, fps):
    """|v(t + DV_STEP) - v(t - DV_STEP)| in m/s from the smoothed ball; NaN where unknown."""
    position = smooth(ball)
    v = np.full_like(position, np.nan)
    v[1:-1] = (position[2:] - position[:-2]) * fps / 2.0
    dv = np.full(len(ball), np.nan)
    step = v[2 * DV_STEP:] - v[:-2 * DV_STEP]
    dv[DV_STEP:-DV_STEP] = np.hypot(step[:, 0], step[:, 1])
    return dv


def free_flight_threshold(dv, distance, where):
    """FREE_FLIGHT_PCT percentile of dv where the nearest player is beyond FREE_FLIGHT_M."""
    free = where & (distance > FREE_FLIGHT_M) & np.isfinite(distance) & ~np.isnan(dv)
    if not free.any():
        raise ValueError("no free-flight frames to set the touch threshold from")
    return float(np.percentile(dv[free], FREE_FLIGHT_PCT))


def detect(tracking, dv, tau):
    """(frames, sides): touches -- local maxima of dv above tau with a player within REACH_M.
    sides is H or A, or NOBODY for a duel (both sides within reach)."""
    with np.errstate(invalid="ignore"):
        peak = np.zeros(len(dv), dtype=bool)
        peak[1:-1] = (dv[1:-1] > tau) & (dv[1:-1] >= dv[:-2]) & (dv[1:-1] > dv[2:])
    frames = np.flatnonzero(peak)
    gap = tracking.players[frames] - tracking.ball[frames, None, :]
    with np.errstate(invalid="ignore"):
        near = np.hypot(gap[..., 0], gap[..., 1]) < REACH_M
    home = (near & (tracking.slot_team == H)).any(axis=1)
    away = (near & (tracking.slot_team == A)).any(axis=1)
    keep = home | away
    sides = np.where(home & away, NOBODY, np.where(home, H, A))
    return frames[keep], sides[keep].astype(np.int8)


def last_touch(n, frames, sides, visible, stoppage, fps):
    """Per frame: side of the last attributed touch (H/A), NOBODY where unknown or ball unseen.

    Resets on stoppage frames and after an untracked run longer than RESET_GAP_S; a duel keeps
    the previous side.
    """
    reset = stoppage.copy()
    gap = features.run_length(~visible)
    reset |= gap > RESET_GAP_S * fps
    touch = np.full(n, NOBODY, dtype=np.int8)
    is_touch = np.zeros(n, dtype=bool)
    touch[frames], is_touch[frames] = sides, True
    out = np.full(n, NOBODY, dtype=np.int8)
    side = NOBODY
    for t in range(n):
        if reset[t]:
            side = NOBODY
        elif is_touch[t] and touch[t] != NOBODY:
            side = touch[t]
        out[t] = side
    out[~visible] = NOBODY
    return out


def held(tracking, reach_m):
    """(side, duel) per frame: H or A where only that side has a player within reach_m of the seen
    ball, NOBODY otherwise; duel where both sides do."""
    gap = tracking.players - tracking.ball[:, None, :]
    with np.errstate(invalid="ignore"):
        near = np.hypot(gap[..., 0], gap[..., 1]) < reach_m
    home = (near & (tracking.slot_team == H)).any(axis=1)
    away = (near & (tracking.slot_team == A)).any(axis=1)
    side = np.where(home & ~away, H, np.where(away & ~home, A, NOBODY)).astype(np.int8)
    return side, home & away


def controls(side, max_fly_by=CONTROL_FRAMES):
    """(frames, sides): the first frame of every run of one side holding for > max_fly_by."""
    frames, sides = [], []
    for team in (H, A):
        for start, stop in _runs(side == team):
            if stop - start > max_fly_by:
                frames.append(start)
                sides.append(team)
    order = np.argsort(frames, kind="stable")
    return (np.asarray(frames, dtype=np.int64)[order],
            np.asarray(sides, dtype=np.int8)[order])
