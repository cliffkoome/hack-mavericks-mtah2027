"""Whether the ball is in play, from the players alone (E010).

Inside a ball gap the decoder knows only that the ball is missing (more often at a stoppage) and
whether the median player is walking. Lang et al. separate in from out of play from the players
alone, with movement features at several time offsets: at a stoppage nobody sprints, the speed
spread collapses, and the shape reorganises for the restart.

Per frame, eight numbers over all tracked players (speed quantiles, running shares, spread),
smoothed over +/- 0.5 s and taken at offsets from -8 s to +8 s; a gradient-boosted classifier with
library defaults. Self-supervised like team_model: trained on ball-seen frames labelled by the
pass-1 decoder, each half predicted by the other half's model. Never sees the ball or the truth.

Two recipes. "e010" is the one above (E010). "lang" (E013) is Lang et al. 2022's in full: offsets
out to +/- 80 s, so a stoppage's regrouping and restart burst are in view where a slow build-up
looks the same over a few seconds; seven more features per team (attack-normalised shape, speed,
acceleration), with home and away swapped as augmentation; and a 3.64 s median filter on P(dead).
"""

import warnings

import numpy as np

from possession import H, A, D, features, team_model

# Params
RECIPES = ("e010", "lang")
OFFSETS_S = (0.0, -1.0, 1.0, -2.0, 2.0, -4.0, 4.0, -8.0, 8.0)
SMOOTH_S = 0.5          # each feature is a centred mean over +/- this
QUANTILES = (25, 50, 75, 90)
JOG, RUN = 2.0, 4.0     # m/s: share of players above each
EVERY = 5               # keep every 5th training frame: neighbours are near-duplicates
SEED = 0
FEATURES = ("speed_p25", "speed_p50", "speed_p75", "speed_p90", "share_jog", "share_run",
            "std_x", "std_y")
# E013, Lang et al. 2022 as published: offsets, and the median filter on the predictions.
LANG_OFFSETS_S = (0.0, -0.4, 0.4, -0.8, 0.8, -2.0, 2.0, -4.0, 4.0, -20.0, 20.0, -40.0, 40.0,
                  -80.0, 80.0)
MEDIAN_S = 3.64
TEAM_FEATURES = ("centroid_x", "centroid_y", "std_x", "std_y", "speed_p50", "speed_p90",
                 "mean_abs_accel")
LANG_WIDTH = len(FEATURES) + 2 * len(TEAM_FEATURES)  # per offset: all players, then H, then A
CHUNK = 20000          # rows per prediction call: bounds memory on the full match


def frame_features(tracking, fps):
    """(n, len(FEATURES)) per-frame features over all tracked players, smoothed; NaN if none."""
    speed = tracking.player_speed
    x, y = tracking.players[..., 0], tracking.players[..., 1]
    tracked = ~np.isnan(speed)
    count = tracked.sum(axis=1)
    with warnings.catch_warnings(), np.errstate(invalid="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        columns = [np.nanpercentile(speed, q, axis=1) for q in QUANTILES]
        columns += [np.where(count > 0, (speed > v).sum(axis=1) / np.maximum(count, 1), np.nan)
                    for v in (JOG, RUN)]
        columns += [np.nanstd(x, axis=1), np.nanstd(y, axis=1)]
    window = max(1, int(round(SMOOTH_S * fps)))
    return np.column_stack([features.rolling_mean(c, window) for c in columns])


def team_frame_features(tracking, spec):
    """(n, 2, len(TEAM_FEATURES)) per-team features of H and A, smoothed; NaN where a team has no
    tracked player or outside the halves (no attack direction there)."""
    fps = spec.fps
    sign = team_model.attack_sign(tracking, spec)
    length = spec.pitch[0]
    lag = max(1, int(round(team_model.VELOCITY_S * fps)))
    x, y = tracking.players[..., 0], tracking.players[..., 1]
    speed = tracking.player_speed
    accel = np.full_like(speed, np.nan)
    accel[lag:-lag] = np.abs(speed[2 * lag:] - speed[:-2 * lag]) / (2 * lag / fps)
    window = max(1, int(round(SMOOTH_S * fps)))
    out = np.full((len(x), 2, len(TEAM_FEATURES)), np.nan)
    with warnings.catch_warnings(), np.errstate(invalid="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        for column, team in enumerate((H, A)):
            slots = tracking.slot_team == team
            s = sign[:, column][:, None]
            tx = np.where(s > 0, x[:, slots], length - x[:, slots])
            tx[s[:, 0] == 0] = np.nan
            ty = y[:, slots]
            values = (np.nanmean(tx, axis=1), np.nanmean(ty, axis=1), np.nanstd(tx, axis=1),
                      np.nanstd(ty, axis=1), np.nanpercentile(speed[:, slots], 50, axis=1),
                      np.nanpercentile(speed[:, slots], 90, axis=1),
                      np.nanmean(accel[:, slots], axis=1))
            for k, value in enumerate(values):
                out[:, column, k] = features.rolling_mean(value, window)
    return out


def context(base, fps, offsets=OFFSETS_S, dtype=None):
    """(n, len(offsets) * width): the features at each offset, NaN past the edges."""
    n, width = base.shape
    out = np.full((n, len(offsets) * width), np.nan, dtype=dtype or base.dtype)
    for index, offset in enumerate(offsets):
        shift = int(round(offset * fps))
        block = out[:, index * width:(index + 1) * width]
        if shift >= 0:
            block[:n - shift] = base[shift:]
        else:
            block[-shift:] = base[:shift]
    return out


def lang_matrix(tracking, spec):
    """(n, len(LANG_OFFSETS_S) * LANG_WIDTH) float32: E013's features at Lang's offsets."""
    team = team_frame_features(tracking, spec)
    base = np.hstack([frame_features(tracking, spec.fps), team.reshape(len(team), -1)])
    return context(base, spec.fps, LANG_OFFSETS_S, dtype=np.float32)


def swap(matrix):
    """The lang matrix with H's and A's features exchanged at every offset."""
    shaped = matrix.reshape(len(matrix), len(LANG_OFFSETS_S), LANG_WIDTH)
    common, k = len(FEATURES), len(TEAM_FEATURES)
    out = shaped.copy()
    out[:, :, common:common + k] = shaped[:, :, common + k:]
    out[:, :, common + k:] = shaped[:, :, common:common + k]
    return out.reshape(matrix.shape)


def fit(matrix, labels, recipe="e010"):
    """Classifier, single-threaded so the result does not depend on the machine. The lang recipe
    also fits every example with the teams exchanged."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from threadpoolctl import threadpool_limits

    if recipe == "lang":
        matrix, labels = np.vstack([matrix, swap(matrix)]), np.concatenate([labels, labels])
    model = HistGradientBoostingClassifier(early_stopping=False, random_state=SEED)
    with threadpool_limits(limits=1):
        model.fit(matrix, labels)
    return model


def predict(model, matrix, recipe="e010"):
    """P(dead) per row; the lang recipe averages it with the teams-exchanged prediction."""
    from threadpoolctl import threadpool_limits

    with threadpool_limits(limits=1):
        if recipe != "lang":
            return model.predict_proba(matrix)[:, 1]
        out = np.empty(len(matrix))
        for start in range(0, len(matrix), CHUNK):
            rows = matrix[start:start + CHUNK]
            out[start:start + CHUNK] = 0.5 * (model.predict_proba(rows)[:, 1]
                                              + model.predict_proba(swap(rows))[:, 1])
        return out


def cross_half(tracking, spec, reference, matrix=None, recipe="e010"):
    """(p_dead, folds): every decoded frame predicted by a model fitted on the *other* half's
    ball-seen frames, labelled dead where the reference says D. The lang recipe median-filters
    each predicted half's P(dead) over MEDIAN_S.

    folds is [(predicted half's frames mask, training frames)] for the check tool and the tests.
    """
    from possession import masking

    if recipe not in RECIPES:
        raise ValueError("unknown recipe {!r}; have {}".format(recipe, RECIPES))
    if matrix is None:
        if recipe == "lang":
            matrix = lang_matrix(tracking, spec)
        else:
            matrix = context(frame_features(tracking, spec.fps), spec.fps)
    decoded = masking.decoded_mask(spec)
    half = spec.half_of_frame()
    frames = np.flatnonzero(decoded & features.ball_visible(tracking) & (reference <= D))
    labels = (reference[frames] == D).astype(np.int8)
    p_dead = np.full(spec.n_frames, np.nan)
    folds = []
    for target in np.unique(half[decoded]):
        other = half[frames] != target
        train, train_labels = frames[other][::EVERY], labels[other][::EVERY]
        if len(np.unique(train_labels)) < 2:
            continue  # one class only: no model to fit, leave NaN
        model = fit(matrix[train], train_labels, recipe)
        predicted = decoded & (half == target)
        p = predict(model, matrix[predicted], recipe)
        if recipe == "lang":
            from scipy.ndimage import median_filter

            p = median_filter(p, size=int(round(MEDIAN_S * spec.fps)), mode="nearest")
        p_dead[predicted] = p
        folds.append((predicted, train))
    return p_dead, folds
