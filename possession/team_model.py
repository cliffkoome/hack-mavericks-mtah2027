"""Which side has the ball, from the players alone (E005b).

The tracker loses the ball in long blocks (median 7 s in play), and there the decoder has no team
evidence. Players stay tracked, and their shape gives the side away: the side in possession pushes
up and spreads out, the other drops back, closes up and presses (Ball Radar, TranSPORTmer).

Per team and frame, eight numbers in attack-normalised coordinates (each team attacking +x); both
teams' at 11 offsets from -8 s to +8 s; a gradient-boosted classifier with library defaults. It is
self-supervised: trained on frames where the ball is seen and a player is on it, never on ground
truth, and refit from the bundle on every run. Each example is used with the teams swapped too, and
predictions are averaged with the swapped prediction, so no home prior is learned and swapping the
teams flips the probability exactly.
"""

import warnings

import numpy as np

from possession import H, A, features

# Params
OFFSETS_S = (0.0, -0.4, 0.4, -1.0, 1.0, -2.0, 2.0, -4.0, 4.0, -8.0, 8.0)
VELOCITY_S = 0.2       # x-velocity from positions this far either side
RUNNING_BACK = -4.0    # m/s towards the own goal: the usual 14.4 km/h "running" band
PRESS_PAIRS = 3        # pressing distance = mean of this many smallest player-to-opponent gaps
MIN_PLAYERS = 6        # fewer tracked players than this: the team's features are NaN
EVERY = 5              # keep every 5th training frame: neighbours are near-duplicates
SEED = 0
CHUNK = 5000           # frames per chunk for the player-to-opponent distances (~45 MB each)
FEATURES = ("centroid_x", "centroid_vx", "std_x", "std_y", "compactness", "line_height",
            "running_back", "press_distance")
N_TEAM = len(FEATURES)


def attack_sign(tracking, spec):
    """(n, 2) +1/-1: whether H (column 0) and A (column 1) attack +x; 0 outside the halves.

    Label-free: over each half, the team whose players' mean x is lower defends the low-x goal.
    """
    sign = np.zeros((spec.n_frames, 2))
    x = tracking.players[..., 0]
    for first, last in spec.graded_blocks:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            home = np.nanmean(x[first:last + 1][:, tracking.slot_team == H])
            away = np.nanmean(x[first:last + 1][:, tracking.slot_team == A])
        home_sign = 1.0 if home <= away else -1.0
        sign[first:last + 1] = (home_sign, -home_sign)
    return sign


def team_features(tracking, spec):
    """(n, 2, N_TEAM) per-frame features of H and A, in attack-normalised coordinates."""
    n = len(tracking.players)
    sign = attack_sign(tracking, spec)
    length = spec.pitch[0]
    lag = max(1, int(round(VELOCITY_S * spec.fps)))
    x, y = tracking.players[..., 0], tracking.players[..., 1]
    vx = np.full_like(x, np.nan)
    vx[lag:-lag] = (x[2 * lag:] - x[:-2 * lag]) / (2 * lag / spec.fps)
    out = np.full((n, 2, N_TEAM), np.nan)
    with warnings.catch_warnings(), np.errstate(invalid="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        for column, team in enumerate((H, A)):
            slots = tracking.slot_team == team
            s = sign[:, column][:, None]
            tx = np.where(s > 0, x[:, slots], length - x[:, slots])
            tx[s[:, 0] == 0] = np.nan
            ty, tvx = y[:, slots], vx[:, slots] * s
            enough = (~np.isnan(tx)).sum(axis=1) >= MIN_PLAYERS
            deepest = np.argmin(np.where(np.isnan(tx), np.inf, tx), axis=1)
            outfield = ~np.isnan(tx)
            outfield[np.arange(n), deepest] = False
            ox, oy, ovx = (np.where(outfield, v, np.nan) for v in (tx, ty, tvx))
            cx, cy = np.nanmean(ox, axis=1), np.nanmean(oy, axis=1)
            values = (cx, np.nanmean(ovx, axis=1), np.nanstd(ox, axis=1), np.nanstd(oy, axis=1),
                      np.nanmean(np.hypot(ox - cx[:, None], oy - cy[:, None]), axis=1),
                      np.nanmin(ox, axis=1), np.sum(ovx < RUNNING_BACK, axis=1).astype(float),
                      _press_distance(tracking.players, slots, outfield))
            for k, value in enumerate(values):
                out[:, column, k] = np.where(enough, value, np.nan)
    return out


def _press_distance(players, slots, outfield):
    """Mean of the PRESS_PAIRS smallest distances from an outfield player to any opponent."""
    out = np.full(len(players), np.nan)
    for start in range(0, len(players), CHUNK):
        own = players[start:start + CHUNK][:, slots]
        own = np.where(outfield[start:start + CHUNK][..., None], own, np.nan)
        other = players[start:start + CHUNK][:, ~slots]
        gap = own[:, :, None, :] - other[:, None, :, :]
        nearest = np.nanmin(np.hypot(gap[..., 0], gap[..., 1]), axis=2)
        nearest = np.sort(np.where(np.isnan(nearest), np.inf, nearest), axis=1)[:, :PRESS_PAIRS]
        out[start:start + CHUNK] = np.where(np.isfinite(nearest).all(axis=1),
                                            nearest.mean(axis=1), np.nan)
    return out


def context(team, fps):
    """(n, len(OFFSETS_S) * 2 * N_TEAM): both teams' features at each offset, NaN past the edges.
    Layout per offset: H's N_TEAM features, then A's."""
    n = len(team)
    flat = team.reshape(n, 2 * N_TEAM)
    blocks = []
    for offset in OFFSETS_S:
        shift = int(round(offset * fps))
        block = np.full_like(flat, np.nan)
        if shift >= 0:
            block[:n - shift] = flat[shift:]
        else:
            block[-shift:] = flat[:shift]
        blocks.append(block)
    return np.hstack(blocks)


def swap(matrix):
    """The same features with H and A exchanged."""
    shaped = matrix.reshape(len(matrix), len(OFFSETS_S), 2, N_TEAM)
    return shaped[:, :, ::-1].reshape(matrix.shape)


def training_frames(tracking, reference, decoded, near_m):
    """(frames, labels): reference-alive, ball seen, nearest player within near_m; label 1 = H."""
    team, distance = features.nearest_player(tracking)
    alive = (reference == H) | (reference == A)
    keep = decoded & alive & (distance < near_m) & (team >= 0)
    frames = np.flatnonzero(keep)
    return frames, (team[frames] == H).astype(np.int8)


def fit(matrix, labels):
    """Classifier fitted on the examples and their swapped copies, single-threaded."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from threadpoolctl import threadpool_limits

    model = HistGradientBoostingClassifier(early_stopping=False, random_state=SEED)
    with threadpool_limits(limits=1):
        model.fit(np.vstack([matrix, swap(matrix)]), np.concatenate([labels, 1 - labels]))
    return model


def predict(model, matrix):
    """P(H has the ball), averaged with the swapped prediction so it is exactly antisymmetric."""
    from threadpoolctl import threadpool_limits

    with threadpool_limits(limits=1):
        direct = model.predict_proba(matrix)[:, 1]
        swapped = model.predict_proba(swap(matrix))[:, 1]
    return 0.5 * (direct + 1.0 - swapped)


def cross_half(tracking, spec, reference, near_m, matrix=None):
    """(p_home, folds): every decoded frame predicted by a model fitted on the *other* half.

    folds is [(predicted half's frames mask, training frames)] for the check tool and the tests.
    """
    from possession import masking

    if matrix is None:
        matrix = context(team_features(tracking, spec), spec.fps)
    decoded = masking.decoded_mask(spec)
    half = spec.half_of_frame()
    frames, labels = training_frames(tracking, reference, decoded, near_m)
    p_home = np.full(spec.n_frames, np.nan)
    folds = []
    for target in np.unique(half[decoded]):
        train = frames[half[frames] != target][::EVERY]
        train_labels = labels[half[frames] != target][::EVERY]
        model = fit(matrix[train], train_labels)
        predicted = decoded & (half == target)
        p_home[predicted] = predict(model, matrix[predicted])
        folds.append((predicted, train))
    return p_home, folds
