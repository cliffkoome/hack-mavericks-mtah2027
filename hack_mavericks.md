# Hack Mavericks — Ball possession from tracking alone

More than a Hack 2027, Phase 1. Team **Hack Mavericks**.
Submission file: `hack_mavericks.csv` (199,319 rows, sha256 `a8ee1b287dadc674…`).
Code: this repository. Full experiment log with every number: [EXPERIMENTS.md](EXPERIMENTS.md).

## In one paragraph

We treat possession as what it is in the metric: a sequence of **intervals**, not a stream of
independent frames. A three-state hidden Markov model (home, away, dead) is decoded with Viterbi
over each half. It pays a fixed cost for every change of state, so a change survives only when
the evidence on both sides of it is strong enough. The evidence per frame comes from the automatic
tracking only: who is nearest the ball, whether the ball is seen, outside the lines or still,
whether the players are walking, and, where the ball is lost, which side the players' shape says
is attacking. That last cue comes from a classifier trained on the match itself. Everything is
deterministic CPU code. One command rebuilds the submitted file byte-for-byte from the data
bundle in about 20 seconds.

## Results we can report

The public 5 minutes is the only ground truth we had. We used it to check correctness, never to
choose between versions: 17 possessions are far too few to rank models. The scorer itself reports
gaps below ~0.36 spellF1 as noise there.

| public window (frames 11,185–18,684) | spellF1 | 5–95% bootstrap | macroF1 | 5–95% bootstrap | possessions (truth 17) |
|---|---|---|---|---|---|
| **hack_mavericks** | **0.4242** | 0.24 – 0.61 | **0.6980** | 0.51 – 0.82 | 16 |
| baseline_smooth125 | 0.4242 | | 0.5812 | | 16 |
| baseline_smooth50 (the published floor) | 0.3256 | | 0.6207 | | 26 |
| baseline_raw | 0.0378 | | 0.5972 | | 131 |

Per class on the public window: H F1 0.80, A F1 0.81, D F1 0.49 (precision 0.998, recall 0.32).
Alive/dead F1 0.92, team accuracy on alive frames 0.87, home share error 0.006.

Over the full match, where there is no ground truth for us, we checked that the output looks like
football. We fixed these plausibility bands before each run:

| full match | value |
|---|---|
| possessions | 308 |
| mean possession | 13.3 s |
| dead ball, share of graded frames | 35.4% (56,166 of 158,857) |
| home share of alive frames | 0.594 (published possession for Barcelona ≈ 0.65, a pass-based figure, used only as a sanity check) |

For comparison, the published baseline's window vote (smooth50) gives 584 possessions of 7.2 s
on the full match. It fragments possessions, which spellF1 punishes.

## Approach

### Why a decoder, not a classifier with smoothing

spellF1 matches whole possessions by temporal overlap. The baselines show the problem: the same
nearest-player rule scores spellF1 0.04 raw and 0.42 smoothed over 125 frames, while macroF1
barely moves. A window vote has no notion of an interval. Narrow, it cuts possessions into pieces;
wide, it swallows short possessions whole, and 6 of the 17 public possessions are under 4 s.
A Viterbi decoder with a switching cost does what spellF1 rewards. A short possession with strong
evidence survives; a flicker of weak evidence does not. We never enforce a minimum possession
length.

### The model

```
tracking (ATD, tactical feed) ──► per-frame cues ──► log-likelihoods of H / A / D
                                                         │
                     pass 1: Viterbi per half ◄──────────┘
                               │ labels (self-supervision)
                               ▼
            team-from-movement classifier, trained half-to-half
                               │ P(home has the ball) on frames with the ball lost
                               ▼
                     pass 2: Viterbi per half with the gap team cue ──► fixed rules ──► CSV
```

**Transitions** (`config.py`, football priors, not fitted): a possession lasts 12 s on average;
30% of possessions end in a stoppage rather than a turnover; a stoppage lasts 15 s on average.
That makes a change of side cost about 6 nats.

**Per-frame evidence** (`possession/evidence.py`). Each cue is a probability with a stated meaning.
The value comes from a documented fact, a football prior, or a rate we measured. None was tuned
for score.

| cue | what it says | values |
|---|---|---|
| nearest player | the side of the tracked player nearest the ball, trusted by distance | < 2 m: right 85%; 2–5 m: 65%; beyond 5 m: no cue |
| ball tracked | the tracker loses the ball more often at a stoppage | P(lost \| alive) 0.40, P(lost \| dead) 0.75 |
| ball outside the lines | a seen ball more than 0.5 m outside the pitch | 1% alive, 30% dead |
| ball still | a seen ball under 0.3 m/s | 5% alive, 50% dead (E001 priors; measured to fire on under 1% of frames in either state, so it barely acts) |
| players walking | the players' median speed, averaged over ±1 s, below 1.25 m/s | 10% alive, 62% dead (measured) |
| gap team cue | where the ball is lost: the side the movement classifier favours | right 86% when confident (p ≥ 0.7 or ≤ 0.3), 56% otherwise (measured) |

Consecutive frames are strongly correlated, so all evidence is multiplied by a weight of 0.2
(about one independent reading per 5 frames) before decoding. Without it, one second of the same
noisy reading would count as 25 votes and drown the switching cost.

**Team from movement inside ball gaps** (`possession/team_model.py`). The ball is lost on 41% of
graded frames. The gaps come in long blocks (median 7.2 s in play), not flickers, and they cover
17% of the frames we call alive. Without the ball the decoder can only carry the last side
through the gap, so possessions that change hands there disappear. The players stay tracked, and
their shape gives the side away: the attacking side pushes up and spreads out, the defending side
drops back, closes up and presses.

- Per team and frame, eight features in attack-normalised coordinates: centroid x and
  x-velocity, spread in x and y, compactness, defensive-line height, players running back faster
  than 4 m/s, and pressing distance. Attack direction per half comes from where each team stands,
  with no labels.
- Both teams' features are taken at 11 time offsets from −8 s to +8 s, giving 176 inputs.
- The model is a gradient-boosted classifier (scikit-learn defaults, seed 0, single-threaded).
- It is **self-supervised**. It is trained on frames where pass 1 calls the ball alive, the ball
  is seen and a player is within 2 m; the label is that player's side. No ground truth is used.
- It is **cross-fitted half-to-half**: each half is predicted by the model trained on the other
  half.
- Every example is also used with the teams swapped, and predictions are averaged with the swapped
  prediction, so the model learns no home prior.
- Held out, it agrees with the pipeline's labels on 79% of all alive frames. On the public truth
  it is right on 86% of alive frames and 95% of alive frames with the ball lost (391 frames).
- In the decoder, it adds evidence only on frames where the ball is lost, so no frame has two team
  cues.

### Possession rules we implemented

- **Who owns the ball:** the side the evidence favours enough to pay for a change. Nearest-player
  evidence is strong within 2 m and weak at 2–5 m.
- **Passes and loose balls:** a pass in flight, or a loose ball near an opponent, gives a few
  frames of weak evidence for the other side. That does not outweigh the switching cost, so the
  possession carries on. A real turnover gives sustained evidence and the side changes.
- **Challenges and duels:** players of both sides near the ball pull in opposite directions; the
  side changes only once one side's evidence is sustained.
- **Dead ball (D):** inferred from the ball outside the lines, the ball still, the ball lost, and
  above all the players walking. Players walking is the strongest dead-ball cue on this tracking.
  We do not ask who owns a dead ball, since the task does not either.
- **Restarts:** after D, the decoder picks the side from the evidence (the nearest player at the
  restart, then the play). There is no rule forcing "the restarting team". Smart Tagging was
  measured for this and not used (see Assets).
- **Fixed rules applied last** (`possession/rules.py`): `X` on every frame outside the graded
  blocks, and `D` on graded frames before a half's kickoff (the 2,934 frames before the
  second-half kickoff).
- **No abstention on graded frames:** spellF1 forward-fills `X` and macroF1 counts it as a miss,
  so abstaining never helped.

## Assets: used and dropped

| asset | use |
|---|---|
| **Tracking (ATD), tactical feed** | **Used**: everything. Our reader rebuilds the four published baselines frame-for-frame (`make verify-baseline` → IDENTICAL ×4), which verifies alignment (`video_frame = frame_id − 1`), slot-team mapping and arithmetic. |
| Public 5-minute ground truth | Correctness checks and measuring cue rates (e.g. the players-walking rates). Never used to choose between versions. |
| Smart Tagging | **Measured, dropped.** Team semantics differ by code: for attacking phases Team is the side with the ball, for defending phases the side without it. Phases are not possessions. Our labels already have a dead ball within 3 s of 86% of set-piece starts, and after a corner tag the first side in our labels matches the tagged team only 25% of the time (the public window holds just 2 set-piece tags, too few to say which is right), so set pieces add little (EXPERIMENTS.md, E002, E007a). |
| Homography | Not needed: the tracking is already in pitch coordinates. |
| Tactical and broadcast video | Not used. Ball detection from video was the last step of our plan, kept for only if the tracking-only model plateaued. We chose explainable CPU code that the organisers can rerun instead. |
| Broadcast-feed tracking | Not used; the submission is indexed by tactical-video frames. |

## What we tried and dropped

Every change had its values and a keep/drop criterion written down before the real-data run, and
the criterion decided. The log has every number.

| experiment | idea | outcome |
|---|---|---|
| E002 | recalibrate five cues at once from measured rates | **dropped**: the D share collapsed from 45% to 8%. Lesson: lost-ball runs are long in both states, so the ball's absence cannot carry dead-ball detection. |
| E004 | long lost-ball runs as dead-ball evidence | **dropped before code**: by its pre-set rule, the cue would have pointed away from D |
| E006 | possession change from ball touches (velocity peaks next to a player) | **stopped**: kicks do not stand out on this ball track; it appears smoothed at source |
| E009 | the last side in control instead of the nearest player | **stopped**: more accurate as a raw cue (0.865 vs 0.780) but no gain after decoding, since its errors run long |
| E010 | dead/alive from player movement inside ball gaps | **dropped** (three versions): it cut wrong dead calls in gaps (14% → 4.5–10%) but cost side-change recall in gaps or the overall D share |
| E007 | set-piece tags as restart evidence | **not built**: our labels already have D near 86% of set-piece starts |

Kept, in order: E001 (the decoder: possessions 584 → 316), E003 (players walking: D share
45% → 37%), E005c (gap team cue: possessions 275 → 308). When we hid the ball on seen frames to
imitate gaps, E005c raised the share of side changes the decoder still found from 12% to 28%.

## How we validated without a full-match ground truth

- **Baseline identity.** Our own reader and vote rebuild the organisers' four baseline files
  exactly.
- **Masked-ball harness** (`tools/masked_harness.py`). We hide the ball where it *is* seen, in
  stretches shaped like the real gaps, and compare against the unmasked run. This measures how
  much worse the model gets without the ball, on tens of thousands of frames, which the public
  window cannot do. Its blind spot: it only hides in-play frames.
- **Pre-registered plausibility bands** for the full match: possessions, mean duration, D share.
- **Synthetic matches** (`tools/make_synthetic.py`). Our generator writes fake bundles in the real
  formats, with known truth. They test correctness and robustness only, never select a version.
  All 139 tests run on synthetic data and the public ground truth.

## Limitations

- **The real score is unknown.** The public window is small (17 possessions; spellF1 interval
  0.24–0.61) and better tracked than the rest of the match: the ball is lost on 6.5% of its
  in-play frames, against 17% over the match. Expect the full-match score to be lower than the
  public one, as it is for the baselines (smooth50: 0.33 public, 0.24 full).
- **Dead ball is under-called on the public window** (D recall 0.32 at precision 0.998): short
  stoppages with players still moving read as play. Over the full match the D share (35.4%) is
  plausible, but stoppage edges are soft.
- **Wrong dead calls in gaps.** When the ball is lost during slow build-up play, the
  players-walking cue fires and the model can call play dead (14% of hidden in-play frames in the
  harness). Every fix we tried traded this against side-change recall.
- **Transitions are early by a few frames** (public median −4 frames). The nearest player flips
  to the receiver before control. That is free within the metric's tolerance, but it shortens
  matched possessions slightly.
- **The team classifier learns from the model's own labels**, so it inherits their errors (about
  14% disagreement with the truth on the public window). It is fitted per match and holds no
  pretrained weights.
- **With more time:** ball recovery from the tactical video in the long gaps (the largest
  remaining loss), and a joint dead/alive and team model trained by masking, as in TranSPORTmer.

## Reproduction

Requirements: Python 3.12 (tested 3.12.7), CPU only, about 1 GB of RAM. Pinned dependencies in
`requirements.txt` (numpy, pandas, scikit-learn 1.5.2, scipy, joblib, threadpoolctl).

```bash
git clone <this repository> && cd <it>
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt   # or: make venv (pyenv)
# unpack the match bundle into ./data, or set MTAH_DATA_DIR to it
.venv/bin/python run_pipeline.py              # -> submissions/hack_mavericks.csv, prints sha256
.venv/bin/python evaluation/validate_submission.py submissions/hack_mavericks.csv
```

Or with make: `make submission` (build and validate) and `make reproduce` (build twice from
scratch, require byte-identical files). The expected sha256 is
`a8ee1b287dadc674…`. Every hyperparameter is in `config.py`, with its reason next to it. The
pipeline is deterministic: fixed seeds, single-threaded model fitting, no GPU.

Code map: `possession/atd.py` (reader), `features.py`, `evidence.py`, `decoder.py`,
`team_model.py`, `rules.py`, `pipeline.py`, `submission.py`; `run_pipeline.py` is the one
command. Experiments and diagnostics are in `tools/`, tests in `tests/` (`make test`, synthetic
data only).

## References

Borghesi et al. 2023 (ESWA, TCN possession); Ball Radar (KDD 2023); TranSPORTmer (ACCV 2024);
Lang et al. 2022 (Sci. Rep., in/out of play from players); Vidal-Codina et al. 2022 (rule-based
events from tracking); Link & Hoernig 2017 (individual ball possession).
