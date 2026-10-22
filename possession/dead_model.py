"""Whether the ball is in play, from the players alone (E010).

Inside a ball gap the decoder knows only that the ball is missing (more often at a stoppage) and
whether the median player is walking. Lang et al. separate in from out of play from the players
alone, with movement features at several time offsets: at a stoppage nobody sprints, the speed
spread collapses, and the shape reorganises for the restart.

Per frame, eight numbers over all tracked players (speed quantiles, running shares, spread),
smoothed over +/- 0.5 s and taken at offsets from -8 s to +8 s; a gradient-boosted classifier with
library defaults. Self-supervised like team_model: trained on ball-seen frames labelled by the
pass-1 decoder, each half predicted by the other half's model. Never sees the ball or the truth.
"""

import warnings

import numpy as np

from possession import D, features

# Params
OFFSETS_S = (0.0, -1.0, 1.0, -2.0, 2.0, -4.0, 4.0, -8.0, 8.0)
SMOOTH_S = 0.5          # each feature is a centred mean over +/- this
QUANTILES = (25, 50, 75, 90)
JOG, RUN = 2.0, 4.0     # m/s: share of players above each
EVERY = 5               # keep every 5th training frame: neighbours are near-duplicates
SEED = 0
FEATURES = ("speed_p25", "speed_p50", "speed_p75", "speed_p90", "share_jog", "share_run",
            "std_x", "std_y")


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


def context(base, fps):
    """(n, len(OFFSETS_S) * len(FEATURES)): the features at each offset, NaN past the edges."""
    n = len(base)
    blocks = []
    for offset in OFFSETS_S:
        shift = int(round(offset * fps))
        block = np.full_like(base, np.nan)
        if shift >= 0:
            block[:n - shift] = base[shift:]
        else:
            block[-shift:] = base[:shift]
        blocks.append(block)
    return np.hstack(blocks)


def fit(matrix, labels):
    """Classifier, single-threaded so the result does not depend on the machine."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from threadpoolctl import threadpool_limits

    model = HistGradientBoostingClassifier(early_stopping=False, random_state=SEED)
    with threadpool_limits(limits=1):
        model.fit(matrix, labels)
    return model


def predict(model, matrix):
    """P(dead) per row."""
    from threadpoolctl import threadpool_limits

    with threadpool_limits(limits=1):
        return model.predict_proba(matrix)[:, 1]


def cross_half(tracking, spec, reference, matrix=None):
    """(p_dead, folds): every decoded frame predicted by a model fitted on the *other* half's
    ball-seen frames, labelled dead where the reference says D.

    folds is [(predicted half's frames mask, training frames)] for the check tool and the tests.
    """
    from possession import masking

    if matrix is None:
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
        model = fit(matrix[train], train_labels)
        predicted = decoded & (half == target)
        p_dead[predicted] = predict(model, matrix[predicted])
        folds.append((predicted, train))
    return p_dead, folds
