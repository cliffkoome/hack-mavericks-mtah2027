"""Generate a fake match bundle in the provided formats, so the pipeline can be developed and tested
without opening the real data.

Nothing here is derived from the match data. Every number is either a documented property of the
bundle (DATA.md, the starter loaders, the public example homography row) or a hand-picked football
prior. The synthetic ground truth is ours, made up by this file, so it is good for checking that
the code is *correct* and useless for choosing between two versions of it -- exactly like the
public 5 minutes, only more so.

    python tools/make_synthetic.py                         # small match -> .synthetic/small-seed0/
    python tools/make_synthetic.py --size full --seed 3    # real-length match, ~200 MB of text
    MTAH_DATA_DIR=.synthetic/small-seed0 python ...        # any games.asset_paths() caller reads it

Output layout mirrors the real bundle, same file names, tactical source only:

    <out>/tactical/BAR-ATH_tactical_atd.xml
    <out>/tactical/BAR-ATH_tactical_atd.txt
    <out>/tactical/BAR-ATH_tactical_homography.csv
    <out>/tactical/BAR-ATH_tactical_smart_tagging.xml
    <out>/synthetic_possession_gt.csv          scorer-compatible: run_evaluation.py --gt <this>
    <out>/synthetic_possession_gt_meta.json
    <out>/synthetic_spec.json                  frame layout, team names, seed, achieved statistics

How it is built: truth first, observations second.
  1. A possession timeline per half: possessions (some under a second), turnovers, and stoppages
     with typed restarts. This *is* the ground truth.
  2. Motion that agrees with it: the ball is carried and passed between teammates, goes out of
     play at a stoppage and waits at the restart spot; 22 players follow formation anchors that
     shift with the ball, chase it when they hold it and run onto passes.
  3. An observation layer with the documented failure modes: 46 track slots in shuffled team order,
     tracklets that break and reappear in another slot, short dropouts, occasional duplicates,
     and a bursty missing-ball process that loses the ball in the air, in crowds and when dead.

Format details that cannot be confirmed without the real files (number of fields in the ball
group, the exact EPTS XML schema, how the ATD names the clubs) are best guesses, kept compatible
with starter/baseline.py's parser. tools/check_format.py, run by a human, is what confirms them.
"""

import argparse
import json
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from games import GAME, GAMES, asset_paths  # noqa: E402

GENERATOR_VERSION = 1

# Params
# Frame layout. "full" reproduces the real match exactly (DATA.md, "Which frames are graded"),
# including the 2,934 graded dead frames before the second-half kickoff. "small" is the same
# shape at two five-minute halves, for tests.
LAYOUTS = {
    "small": {"pre": 500, "half1": 7500, "halftime": 1000, "half2": 7500, "pre_kickoff2": 300,
              "post": 500},
    "full": {"pre": 11185, "half1": 79467, "halftime": 20156, "half2": 79390, "pre_kickoff2": 2934,
             "post": 9121},
}
FPS = GAMES[GAME]["fps"]
PITCH = GAMES[GAME]["pitch"]
N_SLOTS = 46

H, A, D, X = 0, 1, 2, 3
LABEL_CHARS = np.array(["H", "A", "D", "X"])

# Club names as the two files spell them. The Smart Tagging spelling, typo included, is the one
# DATA.md documents; the ATD one is a guess. They differ on purpose, so code that maps clubs to
# sides through an explicit table is exercised and code that string-matches them breaks here first.
ATD_TEAM_NAMES = ("FC Barcelona", "Athletic Club")
TAGGING_TEAM_NAMES = ("Barcelona", "Atletic Club")

# Possession timeline priors. Short possessions are real (the public window has a 19-frame one),
# so a quarter of them are drawn under ~2 s.
P_SHORT_POSSESSION = 0.2
SHORT_POSSESSION_FRAMES = (10, 50)
POSSESSION_MEDIAN_S, POSSESSION_SIGMA = 9.0, 0.9
POSSESSION_CLIP_S = (2.0, 90.0)
P_STOPPAGE = 0.3  # chance a possession ends in a stoppage rather than a turnover
RESTARTS = ("Throw-in", "Free Kick", "Goal Kick", "Corner Kick", "Kick Off (After Goal)")
RESTART_P = (0.46, 0.22, 0.15, 0.15, 0.02)
DEAD_MEDIAN_S, DEAD_SIGMA, DEAD_CLIP_S = 14.0, 0.5, (3.0, 60.0)
GOAL_DEAD_S = (45.0, 70.0)

# 4-3-3 for a team attacking +x, in metres on a 105.3 x 68 pitch. Index 0 is the goalkeeper.
FORMATION = np.array([[5, 34], [22, 10], [20, 26], [20, 42], [22, 58], [40, 18], [38, 34],
                      [40, 50], [58, 12], [60, 34], [58, 56]], dtype=np.float64)
MAX_PLAYER_SPEED = 8.0     # m/s
# Wander around the anchor is smooth: noise drives a velocity (AR(1), per frame), and the offset it
# integrates into is pulled back towards zero. Noise straight on position would make every target
# jump each frame and every player sprint at the speed cap.
WANDER_VEL_PHI, WANDER_VEL_SIGMA, WANDER_PULL = 0.97, 0.012, 0.995

# Observation noise.
POSITION_JITTER_M, BALL_JITTER_M, SPEED_JITTER = 0.12, 0.2, 0.25
TRACKLET_BREAK_EVERY_S = 40.0
TRACKLET_GAP_FRAMES = (0, 15)
DROPOUT_EVERY_S, DROPOUT_FRAMES = 30.0, (2, 25)
DUPLICATE_EVERY_S, DUPLICATE_FRAMES = 300.0, (10, 60)
# Missing ball: target probability by context, realised by a two-state Markov chain whose missing
# runs last BALL_MISSING_RUN frames on average, so gaps come in bursts as they do in real tracking.
BALL_MISSING_BASE, BALL_MISSING_AIR, BALL_MISSING_CROWD = 0.22, 0.35, 0.2
BALL_MISSING_DEAD, BALL_MISSING_OFF = 0.8, 0.6
BALL_MISSING_RUN = 25.0

# The example row in DATA.md, used as the base camera; it is the only homography value published.
BASE_HOMOGRAPHY = np.array([[1822.663, -511.2005, 44.48168],
                            [1.698323, 169.2971, 255.7303],
                            [-0.01907188, -0.5269586, 1.0]])


def build_spec(size):
    """Frame layout of a synthetic match: length, graded blocks (inclusive) and kickoff frames."""
    layout = LAYOUTS[size]
    half1 = (layout["pre"], layout["pre"] + layout["half1"] - 1)
    half2_start = half1[1] + 1 + layout["halftime"]
    half2 = (half2_start, half2_start + layout["half2"] - 1)
    return {
        "size": size,
        "n_video_frames": half2[1] + 1 + layout["post"],
        "fps": FPS,
        "pitch": list(PITCH),
        "graded_blocks": [list(half1), list(half2)],
        "kickoffs": [half1[0], half2_start + layout["pre_kickoff2"]],
    }


# -- 1. the truth ---------------------------------------------------------------------------------

def attack_dir(team, half):
    """+1 if the team attacks towards x = length in that half. Home attacks +x first."""
    return (1 if team == H else -1) * (1 if half == 1 else -1)


def possession_frames(rng):
    if rng.random() < P_SHORT_POSSESSION:
        return int(rng.integers(*SHORT_POSSESSION_FRAMES))
    seconds = np.clip(POSSESSION_MEDIAN_S * np.exp(POSSESSION_SIGMA * rng.standard_normal()),
                      *POSSESSION_CLIP_S)
    return int(round(seconds * FPS))


def dead_frames(rng, kind):
    if kind == "Kick Off (After Goal)":
        return int(round(rng.uniform(*GOAL_DEAD_S) * FPS))
    seconds = np.clip(DEAD_MEDIAN_S * np.exp(DEAD_SIGMA * rng.standard_normal()), *DEAD_CLIP_S)
    return int(round(seconds * FPS))


def restart_team_for(rng, kind, last_team):
    """Who restarts. Corners go to the attacking side, goal kicks and goals to the other one."""
    if kind == "Corner Kick":
        return last_team
    if kind in ("Goal Kick", "Kick Off (After Goal)"):
        return 1 - last_team
    return 1 - last_team if rng.random() < 0.6 else last_team


def simulate_truth(rng, spec):
    """(labels, possessions, stoppages). labels is the per-frame ground truth, X off the blocks."""
    labels = np.full(spec["n_video_frames"], X, dtype=np.int8)
    possessions, stoppages = [], []
    for half, ((first, last), kickoff) in enumerate(
            zip(spec["graded_blocks"], spec["kickoffs"]), start=1):
        end = last + 1
        if kickoff > first:
            labels[first:kickoff] = D
            stoppages.append({"start": first, "stop": kickoff, "half": half,
                              "kind": "Kick Off (Start)", "restart_team": H, "last_team": None})
        team = A if half == 1 else H
        restart, t = "Kick Off (Start)", kickoff
        while t < end:
            stop = min(t + possession_frames(rng), end)
            possessions.append({"start": t, "stop": stop, "team": team, "half": half,
                                "restart": restart})
            labels[t:stop] = team
            t = stop
            if t >= end:
                break
            if rng.random() < P_STOPPAGE:
                kind = str(rng.choice(RESTARTS, p=RESTART_P))
                stop = min(t + dead_frames(rng, kind), end)
                restart_team = restart_team_for(rng, kind, team)
                stoppages.append({"start": t, "stop": stop, "half": half, "kind": kind,
                                  "restart_team": restart_team, "last_team": team})
                labels[t:stop] = D
                t, team, restart = stop, restart_team, kind
            else:
                team, restart = 1 - team, None
    return labels, possessions, stoppages


# -- 2. motion ------------------------------------------------------------------------------------

def formation_base(half):
    """(22, 2) anchor positions with the ball at the centre: home 0..10, away 11..21."""
    out = np.empty((22, 2))
    for team in (H, A):
        block = FORMATION.copy()
        if attack_dir(team, half) < 0:
            block = np.array(PITCH) - block
        out[team * 11:(team + 1) * 11] = block
    return out


def shifted(base, ball):
    """Anchors pulled towards the ball, the way a team block moves with play."""
    length, width = PITCH
    shift = np.array([0.5 * (ball[0] - length / 2), 0.25 * (ball[1] - width / 2)])
    return np.clip(base + shift, 1.0, np.array(PITCH) - 1.0)


def clip_pitch(point, margin=0.5):
    return np.clip(point, margin, np.array(PITCH) - margin)


def simulate_motion(rng, spec, possessions, stoppages):
    """True positions in metres: (ball (n, 2), players (n, 22, 2), in_air (n,), holder (n,), passes)."""
    n = spec["n_video_frames"]
    length, width = PITCH
    centre = np.array([length / 2, width / 2])
    keys = {0: centre.copy()}            # frame -> ball position, interpolated linearly between
    holder = np.full(n, -1, dtype=np.int16)
    in_air = np.zeros(n, dtype=bool)
    runners = []                         # (start, stop, player, target): someone running to a spot
    passes = []
    bases = {1: formation_base(1), 2: formation_base(2)}

    events = sorted([("poss", p) for p in possessions] + [("dead", s) for s in stoppages],
                    key=lambda item: item[1]["start"])
    ball = centre.copy()
    for kind, event in events:
        start, stop, half = event["start"], event["stop"], event["half"]
        if kind == "dead":
            ball = _stoppage_ball(rng, event, ball, keys, runners, bases[half])
            continue
        if event["restart"] == "Kick Off (Start)":
            ball = centre.copy()
        team = event["team"]
        mates = np.arange(11) + 11 * team
        direction = attack_dir(team, half)
        anchors = shifted(bases[half], ball)
        nearest = mates[np.argsort(np.hypot(*(anchors[mates] - ball).T))]
        current = int(nearest[0] if event["restart"] else rng.choice(nearest[:3]))
        t = start
        keys[t] = ball.copy()
        while True:
            carry_end = min(t + int(rng.integers(10, 70)), stop)
            velocity = np.array([direction * rng.uniform(0.5, 4.0), rng.normal(0, 1.5)])
            carried = clip_pitch(ball + velocity * (carry_end - t) / FPS)
            keys[carry_end] = carried
            holder[t:carry_end] = current
            if carry_end >= stop:
                ball = carried
                break
            anchors = shifted(bases[half], carried)
            others = mates[mates != current]
            ahead = direction * (anchors[others, 0] - carried[0])
            distance = np.hypot(*(anchors[others] - carried).T)
            weight = np.exp(ahead / 20.0 - distance / 30.0)
            receiver = int(rng.choice(others, p=weight / weight.sum()))
            target = clip_pitch(anchors[receiver] + rng.normal(0, 3.0, 2))
            length_m = float(np.hypot(*(target - carried)))
            travel = max(4, int(round(length_m / rng.uniform(10.0, 20.0) * FPS)))
            arrive = carry_end + travel
            in_air[carry_end:min(arrive, stop)] = length_m > 30.0
            passes.append({"start": carry_end, "team": team, "half": half, "length": length_m})
            runners.append((carry_end, min(arrive, stop), receiver, target))
            if arrive >= stop:
                # The possession ends mid-pass: an interception at wherever the ball has got to.
                ball = carried + (stop - carry_end) / travel * (target - carried)
                keys[stop] = ball
                break
            keys[arrive] = target
            ball, current, t = target, receiver, arrive
    frames = np.array(sorted(keys))
    points = np.array([keys[f] for f in frames])
    track = np.column_stack([np.interp(np.arange(n), frames, points[:, k]) for k in (0, 1)])
    players = _players(rng, spec, track, holder, runners, bases)
    return track, players, in_air, holder, passes


def _stoppage_ball(rng, event, ball, keys, runners, base):
    """Ball out of play, carried back, then still on the restart spot until the restart frame."""
    length, width = PITCH
    start, stop, kind, half = event["start"], event["stop"], event["kind"], event["half"]
    centre = np.array([length / 2, width / 2])
    restart_team = event["restart_team"]
    if kind == "Kick Off (Start)":
        keys[start] = centre.copy()
        keys[stop] = centre.copy()
        return centre
    if kind == "Throw-in":
        side = 0.0 if ball[1] < width / 2 else width
        out = np.array([ball[0], side + (-2.0 if side == 0 else 2.0)])
        spot = np.array([np.clip(ball[0], 1, length - 1), side])
    elif kind == "Corner Kick":
        end = length if attack_dir(restart_team, half) > 0 else 0.0
        out = np.array([end + (2.0 if end else -2.0), ball[1]])
        spot = np.array([end, 0.0 if ball[1] < width / 2 else width])
    elif kind == "Goal Kick":
        own = 0.0 if attack_dir(restart_team, half) > 0 else length
        out = np.array([own + (-2.0 if own == 0 else 2.0), np.clip(ball[1], 5, width - 5)])
        spot = np.array([own + attack_dir(restart_team, half) * 5.5,
                         width / 2 + rng.uniform(-9, 9)])
    elif kind == "Free Kick":
        out = ball.copy()
        spot = clip_pitch(ball + rng.normal(0, 1.0, 2))
    else:  # goal, then kick off from the centre
        goal = length if attack_dir(event["last_team"], half) > 0 else 0.0
        out = np.array([goal + (1.5 if goal else -1.5), width / 2 + rng.normal(0, 2)])
        spot = centre.copy()
    span = stop - start
    t_out = start + min(25, max(1, span // 4))
    t_spot = max(t_out + 1, stop - max(25, span // 3))
    keys[t_out] = out
    if t_spot < stop:
        keys[t_spot] = spot
    keys[stop] = spot
    anchors = shifted(base, spot)
    mates = np.arange(11) + 11 * restart_team
    taker = int(mates[np.argmin(np.hypot(*(anchors[mates] - spot).T))])
    runners.append((t_out, stop, taker, spot))
    return spot


def _players(rng, spec, ball, holder, runners, bases):
    """Integrate 22 players towards per-frame targets at a capped speed."""
    n = spec["n_video_frames"]
    half_split = spec["graded_blocks"][0][1] + 1 + (
        spec["graded_blocks"][1][0] - spec["graded_blocks"][0][1] - 1) // 2
    window = 50
    cumulative = np.cumsum(np.vstack([np.zeros((1, 2)), ball]), axis=0)
    index = np.arange(n)
    low, high = np.maximum(0, index - window), np.minimum(n, index + window + 1)
    smooth = (cumulative[high] - cumulative[low]) / (high - low)[:, None]

    length, width = PITCH
    shift = np.column_stack([0.5 * (smooth[:, 0] - length / 2), 0.25 * (smooth[:, 1] - width / 2)])
    base = np.where((index < half_split)[:, None, None], bases[1][None], bases[2][None])
    targets = np.clip(base + shift[:, None, :], 1.0, np.array(PITCH) - 1.0)
    free = np.ones((n, 22), dtype=bool)

    for start, stop, player, target in runners:
        targets[start:stop, player] = target
        free[start:stop, player] = False
    live = np.flatnonzero(holder >= 0)
    held = holder[live].astype(np.int64)
    behind = np.where(held < 11, 1.0, -1.0)  # roughly behind the ball; direction is only cosmetic
    targets[live, held] = ball[live] - np.column_stack([0.7 * behind, np.zeros(len(live))])
    free[live, held] = False

    # The nearest opponent of the holder closes the ball down to ~1.8 m.
    opponents = np.where(held[:, None] < 11, np.arange(11, 22)[None], np.arange(11)[None])
    gaps = targets[live[:, None], opponents] - ball[live][:, None]
    pick = np.argmin(np.hypot(gaps[..., 0], gaps[..., 1]), axis=1)
    presser = opponents[np.arange(len(live)), pick]
    direction = gaps[np.arange(len(live)), pick]
    direction /= np.maximum(np.hypot(direction[:, 0], direction[:, 1]), 1e-9)[:, None]
    targets[live, presser] = ball[live] + 1.8 * direction
    free[live, presser] = False

    innovations = rng.standard_normal((n, 22, 2)) * WANDER_VEL_SIGMA
    out = np.empty((n, 22, 2))
    position = targets[0].copy()
    wander, velocity = np.zeros((22, 2)), np.zeros((22, 2))
    step_max = MAX_PLAYER_SPEED / FPS
    for t in range(n):
        velocity = WANDER_VEL_PHI * velocity + innovations[t]
        wander = WANDER_PULL * wander + velocity
        goal = targets[t] + wander * free[t][:, None]
        step = goal - position
        distance = np.hypot(step[:, 0], step[:, 1])
        position = position + step * np.minimum(1.0, step_max / np.maximum(distance, 1e-9))[:, None]
        out[t] = position
    return out


# -- 3. observation -------------------------------------------------------------------------------

def speeds(track):
    """m/s from consecutive positions, last axis = (x, y)."""
    delta = np.diff(track, axis=0, prepend=track[:1])
    return np.hypot(delta[..., 0], delta[..., 1]) * FPS


def graded_mask(spec):
    mask = np.zeros(spec["n_video_frames"], dtype=bool)
    for first, last in spec["graded_blocks"]:
        mask[first:last + 1] = True
    return mask


def observe_ball(rng, spec, ball, players, labels, in_air, scale):
    """Ball positions with NaN where the tracker lost it."""
    near = np.hypot(*(players - ball[:, None, :]).transpose(2, 0, 1)) < 3.0
    crowded = near.sum(axis=1) >= 3
    target = BALL_MISSING_BASE + BALL_MISSING_AIR * in_air + BALL_MISSING_CROWD * crowded
    target = np.where(labels == D, BALL_MISSING_DEAD, target)
    target = np.where(graded_mask(spec), target, BALL_MISSING_OFF)
    target = np.clip(target * scale, 0.0, 0.95)

    leave = 1.0 / BALL_MISSING_RUN
    enter = np.minimum(1.0, target / (1.0 - target) * leave)
    draws = rng.random(len(ball)).tolist()
    missing = np.zeros(len(ball), dtype=bool)
    state = False
    for t, (u, p_enter) in enumerate(zip(draws, enter.tolist())):
        state = (u >= leave) if state else (u < p_enter)
        missing[t] = state
    observed = ball + rng.normal(0, BALL_JITTER_M, ball.shape)
    observed[missing] = np.nan
    return observed


def observe_players(rng, spec, players):
    """(slot_team (46,), slots (n, 46, 2) metres with NaN, slot_speed (n, 46))."""
    n = spec["n_video_frames"]
    slot_team = rng.permutation(np.repeat([H, A], N_SLOTS // 2))
    team_slots = {team: np.flatnonzero(slot_team == team) for team in (H, A)}
    free_from = np.zeros(N_SLOTS, dtype=np.int64)
    current = {}
    for player in range(22):
        team = player // 11
        slot = int(next(s for s in team_slots[team] if s not in current.values()))
        current[player] = slot
    segments = {player: [] for player in range(22)}
    seg_start = {player: 0 for player in range(22)}

    breaks = []
    for player in range(22):
        t = 0
        while True:
            t += int(rng.exponential(TRACKLET_BREAK_EVERY_S * FPS)) + 1
            if t >= n:
                break
            breaks.append((t, player))
    for t, player in sorted(breaks):
        team = player // 11
        old = current[player]
        segments[player].append((old, seg_start[player], t))
        free_from[old] = t
        resume = t + int(rng.integers(*TRACKLET_GAP_FRAMES))
        taken = set(current.values())
        options = [s for s in team_slots[team] if s not in taken and free_from[s] <= resume]
        new = int(rng.choice(options))
        current[player], seg_start[player] = new, resume
        free_from[new] = n  # occupied until the next break frees it
    for player in range(22):
        segments[player].append((current[player], seg_start[player], n))

    true_speed = speeds(players)
    jittered = players + rng.normal(0, POSITION_JITTER_M, players.shape)
    slots = np.full((n, N_SLOTS, 2), np.nan)
    slot_speed = np.full((n, N_SLOTS), np.nan)
    for player, spans in segments.items():
        for slot, start, stop in spans:
            if stop <= start:
                continue
            slots[start:stop, slot] = jittered[start:stop, player]
            slot_speed[start:stop, slot] = true_speed[start:stop, player]
        dropouts = int(rng.poisson(n / (DROPOUT_EVERY_S * FPS)))
        for start in rng.integers(0, n, dropouts):
            stop = start + int(rng.integers(*DROPOUT_FRAMES))
            for slot, a, b in spans:
                lo, hi = max(a, start), min(b, stop)
                if lo < hi:
                    slots[lo:hi, slot] = np.nan
                    slot_speed[lo:hi, slot] = np.nan

    for _ in range(int(rng.poisson(2 * n / (DUPLICATE_EVERY_S * FPS)))):
        player = int(rng.integers(22))
        start = int(rng.integers(0, n))
        stop = min(n, start + int(rng.integers(*DUPLICATE_FRAMES)))
        empty = [s for s in team_slots[player // 11] if np.isnan(slots[start:stop, s, 0]).all()]
        if empty:
            slot = int(rng.choice(empty))
            slots[start:stop, slot] = players[start:stop, player] + rng.normal(0, 0.4, 2)
            slot_speed[start:stop, slot] = true_speed[start:stop, player]

    off = ~graded_mask(spec)
    slots[off] = np.nan
    slot_speed[off] = np.nan
    slot_speed = np.maximum(0.0, slot_speed + rng.normal(0, SPEED_JITTER, slot_speed.shape))
    return slot_team, slots, slot_speed


# -- writers --------------------------------------------------------------------------------------

def _groups(xs, ys, ss):
    """'x,y,s' strings in normalised coordinates, 'NaN,NaN,NaN' where missing."""
    text = np.char.add(np.char.add(np.char.add(np.char.add(
        np.char.mod("%.5f", xs), ","), np.char.mod("%.5f", ys)), ","), np.char.mod("%.2f", ss))
    return np.where(np.isnan(xs), "NaN,NaN,NaN", text)


def write_atd_raw(path, slots, slot_speed, ball, ball_speed, chunk=10000):
    length, width = PITCH
    with open(path, "w", newline="\n") as handle:
        for start in range(0, len(slots), chunk):
            stop = min(len(slots), start + chunk)
            players = _groups(slots[start:stop, :, 0] / length, slots[start:stop, :, 1] / width,
                              slot_speed[start:stop])
            balls = _groups(ball[start:stop, 0] / length, ball[start:stop, 1] / width,
                            ball_speed[start:stop])
            handle.writelines("{}:{}:{}\n".format(start + i + 1, ";".join(players[i]), balls[i])
                              for i in range(stop - start))


def write_atd_meta(path, slot_team, n_frames):
    """FIFA/EPTS-style metadata. Home team first: starter/baseline.py relies on that order."""
    main = ET.Element("main")
    metadata = ET.SubElement(main, "Metadata")
    config = ET.SubElement(metadata, "GlobalConfig")
    ET.SubElement(config, "TrackingType").text = "Optical"
    ET.SubElement(config, "FrameRate").text = str(FPS)
    field = ET.SubElement(config, "FieldSize")
    ET.SubElement(field, "Width").text = "105"
    ET.SubElement(field, "Height").text = "68"
    teams = ET.SubElement(metadata, "Teams")
    team_ids = ("FIFATMA", "FIFATMB")
    for team_id, name in zip(team_ids, ATD_TEAM_NAMES):
        ET.SubElement(ET.SubElement(teams, "Team", id=team_id), "Name").text = name
    players = ET.SubElement(metadata, "Players")
    channels = ET.SubElement(metadata, "PlayerChannels")
    for slot, team in enumerate(slot_team.tolist(), start=1):
        player = ET.SubElement(players, "Player", id="P{}".format(slot), teamId=team_ids[team])
        ET.SubElement(player, "Name").text = "Track {}".format(slot)
        ET.SubElement(player, "ShirtNumber").text = str(slot)
        for channel in ("x", "y", "s"):
            ET.SubElement(channels, "PlayerChannel", id="player{}_{}".format(slot, channel),
                          playerId="P{}".format(slot), channelId=channel)
    specs = ET.SubElement(main, "DataFormatSpecifications")
    spec = ET.SubElement(specs, "DataFormatSpecification", separator=":", startFrame="1",
                         endFrame=str(n_frames))
    ET.SubElement(spec, "StringRegister", name="frameCount")
    all_players = ET.SubElement(spec, "SplitRegister", separator=";")
    for slot in range(1, N_SLOTS + 1):
        group = ET.SubElement(all_players, "SplitRegister", separator=",")
        for channel in ("x", "y", "s"):
            ET.SubElement(group, "PlayerChannelRef",
                          playerChannelId="player{}_{}".format(slot, channel))
    ball = ET.SubElement(spec, "SplitRegister", separator=",")
    for channel in ("x", "y", "s"):
        ET.SubElement(ball, "BallChannelRef", channelId=channel)
    _write_xml(main, path)


def write_homography(rng, path, spec):
    """One plausible camera per frame: the DATA.md example matrix under a slow pixel-space pan."""
    n = spec["n_video_frames"]
    pan = np.cumsum(rng.normal(0, 0.6, (n, 2)), axis=0)
    pan -= np.linspace(0, 1, n)[:, None] * pan[-1]  # end where it started; keeps the walk bounded
    matrices = np.repeat(BASE_HOMOGRAPHY[None], n, axis=0)
    matrices[:, 0, :] += pan[:, :1] * BASE_HOMOGRAPHY[2][None]
    matrices[:, 1, :] += pan[:, 1:] * BASE_HOMOGRAPHY[2][None]
    matrices /= matrices[:, 2:3, 2:3]
    error = np.where(graded_mask(spec), 0.24, 1.5) * np.exp(rng.normal(0, 0.3, n))
    table = np.column_stack([np.arange(n), matrices.reshape(n, 9), error, np.ones(n)])
    header = "frame,h00,h01,h02,h10,h11,h12,h20,h21,h22,reprojection_error,valid"
    np.savetxt(path, table, delimiter=",", header=header, comments="",
               fmt=["%d"] + ["%.7g"] * 10 + ["%d"])


def smart_tagging_instances(rng, spec, ball, possessions, stoppages, passes):
    """Phases and set pieces derived from the truth with jitter, misses and false positives."""
    length = PITCH[0]
    team_name = {H: TAGGING_TEAM_NAMES[0], A: TAGGING_TEAM_NAMES[1]}
    out = []

    def add(start, stop, code, team, half, **extra):
        start_s = max(0.0, start / FPS + rng.normal(0, 1.0))
        end_s = max(start_s + 1.0, stop / FPS + rng.normal(0, 1.0))
        tags = {"Team": team_name.get(team, "N/A"), "Half": ("1st Half", "2nd Half")[half - 1]}
        tags.update(extra)
        out.append((round(start_s, 2), round(end_s, 2), code, tags))

    for i, poss in enumerate(possessions):
        start, stop, team, half = poss["start"], poss["stop"], poss["team"], poss["half"]
        if stop - start < 3 * FPS:
            continue
        forward = attack_dir(team, half) * (ball[(start + stop) // 2, 0] - length / 2) / length
        third = 0 if forward < -1 / 6 else (1 if forward < 1 / 6 else 2)
        attack = ("BUILD UP", "PROGRESSION",
                  str(rng.choice(["BALL IN FINAL THIRD", "CREATING CHANCES"])))[third]
        defend = ("DEFENDING IN ATTACKING THIRD", "DEFENDING IN MIDDLE THIRD",
                  "DEFENDING IN DEFENSIVE THIRD")[third]
        if rng.random() < 0.8:
            add(start, stop, attack, team, half)
        if rng.random() < 0.6:
            add(start, stop, defend, 1 - team, half)
        if poss["restart"] is None and rng.random() < 0.5:
            add(start, min(stop, start + 4 * FPS), "ATTACKING TRANSITION", team, half)
            add(start, min(stop, start + 4 * FPS), "DEFENSIVE TRANSITION", 1 - team, half)
    for kickoff, half in zip(spec["kickoffs"], (1, 2)):
        team = A if half == 1 else H
        add(kickoff - 2 * FPS, kickoff + 3 * FPS, "SET PIECES", team, half, Type="Kick Off (Start)")
    for stop_event in stoppages:
        kind, half = stop_event["kind"], stop_event["half"]
        if kind == "Kick Off (Start)":
            continue
        team = None if kind == "Throw-in" else stop_event["restart_team"]
        add(stop_event["stop"] - 3 * FPS, stop_event["stop"] + 2 * FPS, "SET PIECES", team, half,
            Type=kind)
        if kind == "Kick Off (After Goal)":
            add(stop_event["start"] - 2 * FPS, stop_event["start"] + FPS, "GOALS",
                stop_event["last_team"], half)
    for pass_event in passes:
        if pass_event["length"] > 35 and rng.random() < 0.5:
            add(pass_event["start"], pass_event["start"] + 2 * FPS, "LONG BALL",
                pass_event["team"], pass_event["half"])

    # Misses and false positives, as the docs warn: phases dropped, phases invented.
    kept = [item for item in out if rng.random() >= 0.1]
    for _ in range(int(0.05 * len(kept))):
        first, last = spec["graded_blocks"][int(rng.integers(2))]
        start = int(rng.integers(first, last - 10 * FPS))
        add(start, start + int(rng.integers(3, 10)) * FPS,
            str(rng.choice(["PROGRESSION", "BUILD UP", "DEFENDING IN MIDDLE THIRD"])),
            int(rng.integers(2)), 1 if start <= spec["graded_blocks"][0][1] else 2)
        kept.append(out[-1])
    return sorted(kept, key=lambda item: (item[0], item[2]))


def write_smart_tagging(path, instances):
    """SportsCode XML in the shape starter/load_smart_tagging.py reads."""
    root = ET.Element("file")
    all_instances = ET.SubElement(root, "ALL_INSTANCES")
    for index, (start_s, end_s, code, tags) in enumerate(instances, start=1):
        instance = ET.SubElement(all_instances, "instance")
        ET.SubElement(instance, "ID").text = str(index)
        ET.SubElement(instance, "start").text = "{:.2f}".format(start_s)
        ET.SubElement(instance, "end").text = "{:.2f}".format(end_s)
        ET.SubElement(instance, "code").text = code
        for group, text in tags.items():
            label = ET.SubElement(instance, "label")
            ET.SubElement(label, "group").text = group
            ET.SubElement(label, "text").text = text
    _write_xml(root, path)


def _write_xml(root, path):
    ET.indent(root)
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def write_labels(path, labels):
    """frame,label CSV, the submission format."""
    with open(path, "w", newline="\n") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(frame, label)
                          for frame, label in enumerate(LABEL_CHARS[labels]))


def label_stats(labels, spec):
    """Possession count and mean duration, measured the way the scorer measures them."""
    spells = []
    for first, last in spec["graded_blocks"]:
        window = labels[first:last + 1]
        breaks = np.flatnonzero(window[1:] != window[:-1]) + 1
        bounds = np.concatenate(([0], breaks, [len(window)]))
        spells += [bounds[i + 1] - bounds[i] for i in range(len(bounds) - 1)
                   if window[bounds[i]] in (H, A)]
    return len(spells), float(np.mean(spells)) / FPS if spells else 0.0


# -- entry point ----------------------------------------------------------------------------------

def generate(out_dir, seed=0, size="small", ball_missing_scale=1.0):
    """Write a full synthetic bundle under out_dir and return its spec (with achieved stats)."""
    rng = np.random.default_rng(seed)
    spec = build_spec(size)
    labels, possessions, stoppages = simulate_truth(rng, spec)
    ball, players, in_air, holder, passes = simulate_motion(rng, spec, possessions, stoppages)
    observed_ball = observe_ball(rng, spec, ball, players, labels, in_air, ball_missing_scale)
    slot_team, slots, slot_speed = observe_players(rng, spec, players)
    ball_speed = np.maximum(0.0, speeds(ball) + rng.normal(0, SPEED_JITTER, len(ball)))
    instances = smart_tagging_instances(rng, spec, ball, possessions, stoppages, passes)

    paths = asset_paths(data_dir=out_dir)
    os.makedirs(os.path.dirname(paths["atd_raw"]), exist_ok=True)
    write_atd_raw(paths["atd_raw"], slots, slot_speed, observed_ball, ball_speed)
    write_atd_meta(paths["atd_meta"], slot_team, spec["n_video_frames"])
    write_homography(rng, paths["homography"], spec)
    write_smart_tagging(paths["smart_tagging"], instances)

    gt_path = os.path.join(out_dir, "synthetic_possession_gt.csv")
    write_labels(gt_path, labels)
    graded = graded_mask(spec)
    n_poss, mean_dur = label_stats(labels, spec)
    gt_meta = {
        "game": "SYNTHETIC", "n_video_frames": spec["n_video_frames"], "fps": FPS,
        "covered_video_frames": spec["graded_blocks"],
        "label_counts": {c: int(np.sum(labels == k)) for k, c in enumerate("HAD")},
        "n_possessions": n_poss, "mean_possession_duration_s": round(mean_dur, 3),
        "note": "Synthetic ground truth from tools/make_synthetic.py. Not derived from match data.",
    }
    with open(gt_path[:-len(".csv")] + "_meta.json", "w") as handle:
        json.dump(gt_meta, handle, indent=2, sort_keys=True)

    visible_objects = (~np.isnan(slots[..., 0])).sum(axis=1)
    spec.update({
        "seed": seed, "generator_version": GENERATOR_VERSION,
        "ball_missing_scale": ball_missing_scale,
        "atd_team_names": list(ATD_TEAM_NAMES), "tagging_team_names": list(TAGGING_TEAM_NAMES),
        "stats": {
            "n_possessions": n_poss, "mean_possession_duration_s": round(mean_dur, 3),
            "n_stoppages": len(stoppages), "n_tagging_instances": len(instances),
            "ball_missing_graded": round(float(np.isnan(observed_ball[graded, 0]).mean()), 4),
            "mean_objects_graded": round(float(visible_objects[graded].mean()), 2),
        },
    })
    with open(os.path.join(out_dir, "synthetic_spec.json"), "w") as handle:
        json.dump(spec, handle, indent=2, sort_keys=True)
    return spec


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--size", choices=sorted(LAYOUTS), default="small")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", help="default: .synthetic/<size>-seed<seed>")
    parser.add_argument("--ball-missing-scale", type=float, default=1.0,
                        help="multiply every missing-ball probability (harder > 1 > easier)")
    args = parser.parse_args()
    out = args.out or os.path.join(ROOT, ".synthetic", "{}-seed{}".format(args.size, args.seed))
    result = generate(out, args.seed, args.size, args.ball_missing_scale)
    print("wrote {} ({} frames)".format(os.path.relpath(out, ROOT), result["n_video_frames"]))
    for key, value in sorted(result["stats"].items()):
        print("  {:<28} {}".format(key, value))
    print("score a submission against it:\n  .venv/bin/python evaluation/run_evaluation.py "
          "--gt {} --results <dir>".format(os.path.relpath(
              os.path.join(out, "synthetic_possession_gt.csv"), ROOT)))
