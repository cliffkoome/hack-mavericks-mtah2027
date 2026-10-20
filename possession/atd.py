"""Read the raw ATD (FIFA/EPTS text + metadata XML) into arrays indexed by video frame.

Alignment is the documented one and nothing else: row = frame_id - 1 (DATA.md, starter/load_atd.py).
The per-half offsets in games.py belong to a different counter and are never applied here.

Every line must be "<frame_id>:<slot>;<slot>;...:<ball>", each slot "x,y,speed" and the ball at
least "x,y", coordinates normalised 0-1, NaN for missing. Anything else is an error that names the
line number and the field count -- never a value, so an error message can be shared safely.

numpy only. A line-by-line float() loop over 199,319 x 46 slots costs minutes; translating the
separators and converting one chunk of lines at a time costs seconds and bounded memory.
"""

import xml.etree.ElementTree as ET
from dataclasses import dataclass

import numpy as np

from possession import H, A
from games import GAME, require

# Params
SLOT_FIELDS = 3          # x, y, speed
CHUNK_BYTES = 40_000_000  # ~20k lines of 46 slots; bounds peak memory on the full match
_SEPARATORS = str.maketrans(":;", ",,")


@dataclass
class Tracking:
    players: np.ndarray       # (n, slots, 2) metres, NaN where missing
    player_speed: np.ndarray  # (n, slots) m/s
    ball: np.ndarray          # (n, 2) metres, NaN where missing
    slot_team: np.ndarray     # (slots,) H or A
    team_names: tuple         # (home, away) as the metadata spells them


def read_meta(meta_path):
    """(slot_team, (home, away)). First Team in the file is home -- the FIFA/EPTS convention the
    starter baseline uses; Player order is slot order."""
    root = ET.parse(meta_path).getroot()
    teams = [(team.get("id"), team.findtext("Name")) for team in root.iter("Team")]
    if len(teams) != 2:
        raise ValueError("{}: expected 2 teams, found {}".format(meta_path, len(teams)))
    sides = {teams[0][0]: H, teams[1][0]: A}
    slot_team = [sides.get(player.get("teamId")) for player in root.iter("Player")]
    if not slot_team or None in slot_team:
        raise ValueError("{}: every Player needs a teamId naming one of the two teams"
                         .format(meta_path))
    return np.array(slot_team, dtype=np.int8), (teams[0][1], teams[1][1])


def read_raw(raw_path, n_frames, n_slots):
    """(n_frames, fields) float64 array of every number in the file, row i = frame_id i + 1."""
    width = None
    rows = []
    with open(raw_path) as handle:
        line_number = 0
        while True:
            lines = handle.readlines(CHUNK_BYTES)
            if not lines:
                break
            if width is None:
                width = lines[0].translate(_SEPARATORS).count(",") + 1
                ball_fields = width - 1 - SLOT_FIELDS * n_slots
                if ball_fields < 2 or lines[0].split(":")[1].count(";") != n_slots - 1:
                    raise ValueError("{}: line 1 does not have {} slots of {} fields plus a ball"
                                     .format(raw_path, n_slots, SLOT_FIELDS))
            text = ",".join(line.rstrip("\n") for line in lines).translate(_SEPARATORS)
            flat = np.array(text.split(","), dtype=np.float64)
            if flat.size != width * len(lines):
                for offset, line in enumerate(lines):
                    count = line.translate(_SEPARATORS).count(",") + 1
                    if count != width:
                        raise ValueError("{}: line {} has {} fields, expected {}".format(
                            raw_path, line_number + offset + 1, count, width))
            rows.append(flat.reshape(len(lines), width))
            line_number += len(lines)
    if not rows:
        raise ValueError("{}: empty".format(raw_path))
    table = np.concatenate(rows)
    frame_ids = table[:, 0]
    if len(table) != n_frames or not np.array_equal(frame_ids, np.arange(1, n_frames + 1)):
        raise ValueError("{}: expected frame_id 1..{} in order, one line each; got {} lines"
                         .format(raw_path, n_frames, len(table)))
    return table


def load(spec, data_dir=None, game=GAME):
    """Tracking for the tactical source, in metres, row = video frame."""
    meta_path, raw_path = require(["atd_meta", "atd_raw"], game, data_dir)
    slot_team, names = read_meta(meta_path)
    n_slots = len(slot_team)
    table = read_raw(raw_path, spec.n_frames, n_slots)
    length, width = spec.pitch
    slots = table[:, 1:1 + SLOT_FIELDS * n_slots].reshape(spec.n_frames, n_slots, SLOT_FIELDS)
    ball = table[:, 1 + SLOT_FIELDS * n_slots:1 + SLOT_FIELDS * n_slots + 2]
    scale = np.array([length, width])
    # A slot or ball counts as missing when either coordinate is, so x and y are NaN together.
    players = slots[..., :2] * scale
    players[np.isnan(players).any(axis=-1)] = np.nan
    ball = ball * scale
    ball[np.isnan(ball).any(axis=-1)] = np.nan
    return Tracking(players=players, player_speed=slots[..., 2], ball=ball, slot_team=slot_team,
                    team_names=names)
