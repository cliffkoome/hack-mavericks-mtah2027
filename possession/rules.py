"""Label rules every submission obeys, whatever model produced the labels.

Applied last, so no model has to know about them:
  - X on every frame outside the graded blocks: those frames are never scored, and X there is
    the honest answer (CHALLENGE.md).
  - D on graded frames before a half's kickoff: the 2,934 frames before the second-half kickoff
    are all dead ball (FAQ.md). The rule is general, so it also holds on synthetic layouts.
"""

import numpy as np

from possession import D, X


def finalize(codes, spec):
    """Copy of per-frame codes with the fixed rules applied."""
    if len(codes) != spec.n_frames:
        raise ValueError("labels have {} frames, the match has {}".format(len(codes), spec.n_frames))
    out = np.asarray(codes, dtype=np.int8).copy()
    out[~spec.graded_mask()] = X
    out[spec.pre_kickoff_mask()] = D
    return out
