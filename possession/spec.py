"""The frame layout of a match: how long, which frames are graded, where each half kicks off.

The real match comes from games.py and the published kickoff frames. A synthetic bundle from
tools/make_synthetic.py carries its own layout in synthetic_spec.json, and load() prefers that file
when it is present, so the same pipeline runs on both without a flag.
"""

import json
import os
from dataclasses import dataclass

import numpy as np

import possession  # noqa: F401  (puts the repository root on the path for games)
from games import GAME, GAMES, half_video_blocks

# Params
# Kickoff video frames, as published in DATA.md and FAQ.md. Deliberately not games.py's
# "kickoff" anchors: those are playback positions and compute one frame off (11186, 113741).
KICKOFFS = {"BAR-ATH": (11185, 113742)}
SYNTHETIC_SPEC = "synthetic_spec.json"


@dataclass(frozen=True)
class GameSpec:
    n_frames: int
    fps: int
    pitch: tuple          # (length, width) in metres
    graded_blocks: tuple  # ((first, last), ...) inclusive video frames, one per half
    kickoffs: tuple       # first frame of play in each half

    def graded_mask(self):
        mask = np.zeros(self.n_frames, dtype=bool)
        for first, last in self.graded_blocks:
            mask[first:last + 1] = True
        return mask

    def half_of_frame(self):
        """int8 per frame: 1 or 2 inside a graded block, 0 outside."""
        half = np.zeros(self.n_frames, dtype=np.int8)
        for index, (first, last) in enumerate(self.graded_blocks, start=1):
            half[first:last + 1] = index
        return half

    def pre_kickoff_mask(self):
        """Graded frames before their half's kickoff: dead ball by definition (FAQ.md)."""
        mask = np.zeros(self.n_frames, dtype=bool)
        for (first, _), kickoff in zip(self.graded_blocks, self.kickoffs):
            mask[first:kickoff] = True
        return mask


def real(game=GAME):
    """The match the submission is scored on."""
    entry = GAMES[game]
    blocks = tuple((first, last) for _, first, last in half_video_blocks(game))
    return GameSpec(n_frames=entry["n_video_frames"], fps=entry["fps"],
                    pitch=tuple(entry["pitch"]), graded_blocks=blocks,
                    kickoffs=tuple(KICKOFFS[game]))


def synthetic(data_dir):
    """The layout written by tools/make_synthetic.py into its bundle."""
    with open(os.path.join(data_dir, SYNTHETIC_SPEC)) as handle:
        raw = json.load(handle)
    return GameSpec(n_frames=raw["n_video_frames"], fps=raw["fps"], pitch=tuple(raw["pitch"]),
                    graded_blocks=tuple(tuple(block) for block in raw["graded_blocks"]),
                    kickoffs=tuple(raw["kickoffs"]))


def load(data_dir=None, game=GAME):
    """Synthetic layout if data_dir holds a synthetic bundle, otherwise the real match."""
    if data_dir and os.path.isfile(os.path.join(data_dir, SYNTHETIC_SPEC)):
        return synthetic(data_dir)
    return real(game)
