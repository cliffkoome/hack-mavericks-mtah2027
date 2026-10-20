"""Write, validate and summarise a frame,label CSV.

The summary is the sanity check CLAUDE.md asks for before every submission: possession count and
mean duration, measured exactly as the scorer measures them (abstentions filled), plus a sha256 so
"the same file" is a checkable claim rather than a hope.
"""

import hashlib
import os

import numpy as np

import possession_metrics as pm
from possession import LABELS, H, A, D, X

_LETTERS = np.array(list(LABELS))


def to_text(codes):
    """The exact bytes of the submission: header, then one 'frame,label' line per frame, '\\n'."""
    letters = _LETTERS[np.asarray(codes)]
    return "frame,label\n" + "".join("{},{}\n".format(i, c) for i, c in enumerate(letters))


def write(path, codes):
    """Write the CSV and return its sha256."""
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    data = to_text(codes).encode("ascii")
    with open(path, "wb") as handle:
        handle.write(data)
    return hashlib.sha256(data).hexdigest()


def validate(path, spec):
    """None if the scorer would accept the file, else its one-line reason."""
    return pm.validate_submission(pm.read_label_csv(path), spec.n_frames)


def summary(codes, spec):
    """Counts on graded frames, and possessions as the scorer sees them."""
    codes = np.asarray(codes, dtype=np.int8)
    graded = spec.graded_mask()
    runs = [(first, last + 1) for first, last in spec.graded_blocks]
    filled = pm.fill_abstentions(codes, runs)
    stats = pm.possession_stats(filled, runs)
    return {
        "graded": {letter: int(np.sum(codes[graded] == code))
                   for code, letter in zip((H, A, D, X), LABELS)},
        "n_possessions": stats["n_spells"],
        "mean_possession_s": round(stats["mean_duration_s"], 2),
        "home_share": round(stats["home_share"], 3),
    }
