# Experiments — Hack Mavericks

Every change to what produces `hack_mavericks.csv` gets an entry. The log is how we choose between
versions without a leaderboard, and it becomes the backbone of the technical document.

## Rules for reading any number here

- **The public 5 minutes is a correctness check, never a selector.** 17 possessions: one miss
  moves spellF1 by ~6 points, and the smoothing window that wins there loses on the full match
  (EVALUATION.md). Report it as pass/fail plus counts, never to pick a version.
- **Synthetic results are an upper bound.** `tools/make_synthetic.py` is ours, so its assumptions
  overlap with our models. Use it to show that a change works as intended and is stable across
  seeds, never to pick a value.
- **Noise band:** full-match spellF1 differences below ~0.04 are noise; macroF1 below ~0.01.
  Run `--bootstrap 1000` before believing a gap.
- **Real-data numbers are aggregates only** (scores, counts, durations, shares), produced by a
  human running the code locally. No frame-level output, rows or tracking values in this file or
  in any conversation with a hosted LLM (CLAUDE.md).
- **Parameter values are set by reasoning recorded here, before seeing results.** If a value is
  changed after seeing a result, say so in the entry.
- **External sanity check, never a target:** published match stats give Barcelona ~65% of the
  possession (a pass-based broadcast figure, not the tracking-based definition here). Our home
  share while the ball is alive is 0.60 (E003). A change that moves it clearly *away* from ~0.65
  deserves a second look; no value is ever set to move it towards 0.65.

## Background: literature (team review, 2026-10-21)

Figures are as the papers report them, collected in the team's review. They are not reproduced
here, and **they are not comparable with our task**: they are per-frame accuracies on corrected or
GPS tracking, while ours is automatic tracking that loses the ball on 41% of graded frames, scored
per possession. Treat them as rough targets, not promises.

| source | what it does | what we take from it |
|---|---|---|
| Borghesi et al. 2023, ESWA (TCN possession) | dead/home/away from tracking: 89% dead-alive, 86% possession | two stages (dead/alive, then team) beat one ternary model; 64-frame acausal windows; missing ball interpolated plus a missing flag |
| Ball Radar (KDD'23, github.com/pientist/ballradar); TranSPORTmer (ACCV'24) | team possession from player trajectories alone, ~81–86% | ~10 s windows of position, velocity, acceleration; TranSPORTmer trains by masking 80–90% of ball observations, the idea behind E005a |
| Lang et al. 2022, Sci Rep | in/out of play from players only, 92% (AdaBoost, CPU) | features at offsets ±0.4/0.8/2/4/20/40/80 s; the seconds around a status change matter most |
| Vidal-Codina et al. 2022 | rule-based events from tracking | possession zone ~1 m; duel zone (1 m, 2 opponents) does not flip possession; a gain needs a ball direction or speed change; Savitzky–Golay 7-frame ball smoothing; set-piece triggers from player configuration; executor = closest player at the first in-play frame (E006, E007) |
| Link & Hoernig 2017 | individual ball possession | a kick is a ball acceleration peak ≥ 4 m/s² (E006) |
| Morra et al. 2020, "Slicing and dicing soccer"; databallpy docs | possession and event tooling | reference implementations to read, not to depend on |
| github.com/metrica-sports/sample-data | 3 public games, tracking + events (game 3 is EPTS, like ours) | public, so it may be read directly (E008) |

Two cautions that bind the plan:

- **Paper accuracies are per frame on cleaner tracking.** Our gates and criteria stay our own.
- **Metrica sample data is a development aid, not validation.** Its labels come from human event
  tagging, our ground truth from corrected tracking, so they disagree at possession edges. The
  real checks stay half-to-half on our own match plus simulated gaps (E005a).

## Roadmap from E005 (one PR per step; design approved by the human before code)

| step | idea | gate before the next step |
|---|---|---|
| E005a | Masked-ball harness: hide the ball where it *is* seen, measure what the pipeline loses | runs on synthetic and real; numbers are a baseline, not a decision |
| E005b | Team-from-movement classifier, half-to-half cross-validated | pooled hold-out accuracy ≥ 0.60, else stop E005 |
| E005c | Its probabilities as gap-only team emissions in the decoder | E005's real-run criterion |
| E006 | Visible-ball possession-change rules (duel zone, direction/speed change, kick peak) | own criterion, set at design |
| E007 | Restart detection and attribution from player configuration, incl. N/A throw-ins | own criterion, set at design |
| E008 | Metrica sample-data harness: weak H/A/D labels from events, for development and pre-training | development aid only |

Each step: design proposed and approved first; few hyperparameters, fixed by written rules,
never chosen on the public 5 minutes; tests on synthetic data; an entry here; the exact local
command and what to paste back.

## Index

| ID | date | commit | idea | decision |
|---|---|---|---|---|
| E000 | 2026-10-20 | b7292ff | Reproduce the reference baseline with our own code | keep: IDENTICAL ×4 on real data |
| E001 | 2026-10-20 | PR #5 | 3-state Viterbi decoder instead of the window vote | keep: default model since PR #5 |
| E002 | 2026-10-20 | PR #8 | Recalibrate the evidence cues against measured rates | drop: D share 45% -> 8%; reverted in PR #9 |
| E003 | 2026-10-21 | PR #11 | Player-movement cue (players walk at stoppages) | keep: D share 45.3% -> 36.7% |
| E004 | 2026-10-21 | PR #13 | Long untracked-ball runs (> 10 s) as dead-ball evidence | drop: value rule gives alive rate 0.41 ≥ 0.15; not implemented |
| E005a | 2026-10-21 | PR #17 | Masked-ball harness (yardstick for gaps) | measured: E003 catches 11.5% of side changes in gaps |
| E005b | 2026-10-21 | PR #19 | Team-from-movement classifier, half-to-half CV | gate PASS: (b) 0.789, (d) 0.949 |
| E005c | 2026-10-21 | PR #21 | Classifier probabilities as gap-only emissions | keep: 275 -> 308 possessions; gap change recall 0.115 -> 0.284 |
| E006 | 2026-10-21 | PR #24, #25 | Last-touch team on seen-ball frames (touch, duel rules) | stop: Δv touches do not stand out on this ball track |
| E007 | — | not yet | Restart detection and attribution from player configuration | planned |
| E008 | — | not yet | Metrica sample-data harness (development aid) | planned |
| E009 | — | not yet | Last-control side on seen-ball frames (proximity, no velocity) | design proposed: measure first |

## Entry template

```markdown
### E0NN — <one-line idea>

- **Date / author / commit:**
- **Hypothesis:** what should improve, and why (which failure mode it addresses)
- **Change:** modules touched; config diff (old -> new, with the reason for each new value)
- **Synthetic** (`python tools/synthetic_eval.py`, seeds, size): spellF1, macroF1, poss ratio, ± std
- **Public window** (correctness only): pass/fail, poss count vs 17, mean duration vs 14.2 s
- **Full-match plausibility** (human-run, aggregates only): possession count, mean duration,
  home share, D share of graded frames, transitions per minute, agreement with Smart Tagging
  set pieces
- **Bootstrap** (`--bootstrap 1000`, public window): interval, overlap with the previous version
- **Decision:** keep / drop / revisit, and why
- **Follow-ups:**
```

---

### E000 — Reproduce the reference baseline with our own code

- **Date / author / commit:** 2026-10-20, Hack Mavericks, b7292ff
- **Hypothesis:** our ATD reader, frame alignment (row = frame_id − 1), slot-team mapping and
  distance arithmetic match `starter/baseline.py`; if so, the four committed baseline CSVs are
  rebuilt frame for frame.
- **Change:** `possession/` pipeline with `model = "vote"`, `smooth_window = 50`.
- **Synthetic:** `features.nearest_team` equals the starter's nearest-player labels frame for
  frame, and `smoothing.majority_vote` equals `baseline.smooth` at 0/25/50/125
  (`tests/test_pipeline.py`).
- **Public window:** `hm_vote` scores exactly as `baseline_smooth50` (macroF1 0.6207, spellF1
  0.3256, 26 possessions), as it must.
- **Full-match check (human-run):** `make verify-baseline` → `IDENTICAL` for all four baselines.
  Our reader, alignment, slot-team mapping and distance arithmetic match the starter on the real
  file, frame for frame. Full match (vote): 584 possessions, mean 7.18 s, home share 0.566,
  D 34.0% of graded frames.
- **Decision:** keep as the floor (published spellF1 0.2433) and as the reference for E001.

### E001 — 3-state Viterbi decoder instead of the window vote

- **Date / author / commit:** 2026-10-20, Hack Mavericks, this PR
- **Hypothesis:** the vote has no notion of a possession as an interval: narrow windows fragment
  (raw baseline: 131 possessions where the public window has 17), wide ones swallow short
  possessions. A decoder that pays a fixed cost per state change keeps a change only when the
  evidence on both sides outweighs it, so short possessions with strong evidence survive and
  flicker does not. Adding dead-ball cues (ball untracked, outside the lines, still) should also
  lift the weak D class.
- **Change:** `possession/evidence.py`, `possession/decoder.py`, `config.py` (`model = "hmm"`
  available, default still `"vote"`). Values and reasons:
  - transitions: mean possession 12 s, 30% of possessions end in a stoppage, mean stoppage 15 s
    (football priors);
  - team cue: nearest player < 2 m → 85% right side, < 5 m → 65%, beyond → no cue;
  - ball untracked: 40% when alive (DATA.md: 41% of graded frames), 75% when dead (assumed);
  - ball outside the lines: 1% alive, 30% dead; ball still: 5% alive, 50% dead;
  - evidence weight 0.2: frames are strongly correlated, ~1 independent reading per 5 frames.
    Chosen before any result.
- **Synthetic** (5 full-size seeds, `tools/synthetic_eval.py`):

  | variant | spellF1 | macroF1 | poss ratio |
  |---|---|---|---|
  | vote (baseline) | 0.295 ± 0.010 | 0.512 ± 0.013 | 1.68 |
  | **hmm (w = 0.2)** | **0.551 ± 0.026** | **0.835 ± 0.010** | 0.73 |
  | hmm, w = 0.1 | 0.468 ± 0.027 | 0.826 ± 0.010 | 0.48 |
  | hmm, w = 0.4 | 0.386 ± 0.023 | 0.780 ± 0.014 | 1.34 |

  The decoder works as intended and the gain is stable across seeds. Two cautions:
  1. **The dead-ball cues mirror the generator's own mechanics** (ball goes out, waits at the
     spot, is lost more often when dead), so the macroF1 jump is an upper bound.
  2. **The weight is the sensitive parameter:** ×2 or ×½ costs 0.08–0.17 spellF1 here. It
     under-segments at 0.2 (0.73× the true possession count). Not retuned: on the real data
     the possession-count plausibility check decides whether it is in a sane range.
- **Pre-set criterion** (written before the real-data run): switch the default to `"hmm"` only
  if `make verify-baseline` passes and the full-match possession count and mean duration look
  like football (hundreds of possessions, ~10–20 s).
- **Full-match plausibility** (human-run, 2026-10-20, aggregates only):

  | | vote | hmm |
  |---|---|---|
  | possessions | 584 | 316 |
  | mean duration | 7.18 s | 11.0 s |
  | D share of graded frames | 34.0% | 45.3% |
  | home share | 0.566 | 0.611 |

- **Public window** (correctness only): hmm 18 possessions vs 17, mean 14.5 s vs 14.2 s;
  macroF1 0.6831 (D F1 0.48 vs 0.32 for the vote), spellF1 0.3771 vs 0.3256. Inside the public
  noise band: it shows the structure is right, not that hmm is better.
- **Bootstrap** (public window, 1000 draws): the scorer reports gaps below **~0.33 spellF1** as
  within noise. Every submission on the public board is inside that band, so the public window
  cannot order hmm, vote or any baseline. Full-match plausibility is the deciding evidence.
- **Decision:** keep. Both conditions of the pre-set criterion are met, so `model = "hmm"` is the
  default from PR #5.
- **Watch:**
  1. **D may be over-called:** 45% of graded frames, against a typical 35–45% stoppage share in
     elite football. On the public window, 385 A frames were called D.
  2. **Transitions are early:** mean latency −10 frames, median −11, on 6 matched transitions.
     A plausible cause is that the nearest player flips to the receiver while the ball is still
     travelling, before control. Not corrected from the public window; a hypothesis for a
     full-match test.
- **Follow-ups:** calibrate the dead-ball cues against reality (what the ball-missing, outside and
  still rates actually are when dead vs alive); Smart Tagging set pieces as restart evidence;
  a control-based team cue for the early-transition bias.

### E002 — Recalibrate the evidence cues against measured rates

- **Date / author / commit:** 2026-10-20, Hack Mavericks, this PR
- **Hypothesis:** E001's D over-call (45% of graded frames) comes from the untracked-ball cue.
  The tracker loses the ball in long bursts (41% of graded frames over the match), and each
  untracked frame was counted as independent evidence for D, so a few seconds of lost ball in
  open play outvoted the switching costs. Meanwhile the "still" cue never fires (jitter), and
  as configured every tracked frame pushed *against* D.
- **Measured** (`tools/inspect_cues.py`, public window, human-run; frame-level rates over
  thousands of frames, used to calibrate stated probabilities, not to compare scores):

  | cue | alive (assumed) | dead (assumed) |
  |---|---|---|
  | ball untracked | 0.065 (0.40) | 0.336 (0.75) |
  | outside lines, ball tracked | 0.000 (0.01) | 0.101 (0.30) |
  | still over 5 frames | 0.009 (0.05) | 0.009 (0.50) |

  Nearest-player team accuracy: < 2 m 0.873 (0.85 assumed), 2–5 m 0.647 (0.65), > 5 m 0.675
  (assumed uninformative).

  Smart Tagging, Team vs truth during the tag (1–5 tags per code, indicative only):
  - BUILD UP, PROGRESSION, LONG BALL, ATTACKING TRANSITION: Team = the side with the ball.
  - DEFENDING IN …, DEFENSIVE TRANSITION: Team = the side *without* it.
  - SET PIECES free kick and goal kick: Team = the restarting side.
  - BALL IN FINAL THIRD / IN THE BOX / PLAYERS IN THE BOX: ~1/3 each way; no team signal.
  - KICK OFF (START): 1 tag, 37 frames, pointing to the other side; undetermined.
- **Change** (`evidence.py`, `features.run_length`, `config.py`), with the reason for each value:
  - `missing_run_cap = 25`: a run of untracked frames counts as at most 1 s of evidence.
  - `missing_alive` 0.40 → 0.26, `missing_dead` 0.75 → 0.65: anchored on the documented
    full-match 41% (0.62 × 0.26 + 0.38 × 0.65 = 0.41, with a ~38% stoppage share and a 2.5×
    dead/alive ratio). Deliberately *not* the public-window rates: that window is unusually well
    tracked (6.5% untracked in play against 41% over the match).
  - `outside_alive` 0.01 → 0.002 (0 of 5,640 measured), `outside_dead` 0.30 → 0.10 (0.101).
  - `still_dead` 0.50 → 0.05: neutralised (equal to alive) until a definition that survives
    jitter is found; the tool now also reports a 1-second version.
  - `team_acc_far = 0.6` (new): 0.675 measured, shrunk towards 0.5 for a 923-frame sample.
- **Synthetic** (5 full seeds): hmm spellF1 0.551 → 0.585 ± 0.019, macroF1 0.835 → 0.868,
  poss ratio 0.73 → 0.65. Nothing broke. The generator's own rates differ from these, so this
  is not evidence for the values.
- **Pre-set criterion for the real run:** keep E002 if the full-match D share falls from 45.3%
  towards a typical 35–42% while the possession count stays in the hundreds with a 10–20 s mean.
  Revert if the possession count collapses below ~200 or the D share rises.
- **Full-match plausibility** (human-run, aggregates only):

  | | E001 | E002 | pre-set criterion |
  |---|---|---|---|
  | D share of graded frames | 45.3% | **8.4%** | towards 35–42% |
  | possessions | 316 | 251 | ≥ ~200 |
  | mean duration | 11.0 s | **23.2 s** | 10–20 s |
  | home share | 0.611 | 0.535 | |

- **Public window** (correctness only): macroF1 0.6831 → 0.5490, the D class collapsing;
  spellF1 0.3771 → 0.4121, inside the ±0.33 noise band and ignored as the rules require.
- **Why it failed:** the new diagnostics show untracked frames come in runs longer than 1 s
  in *both* states (alive 0.064 of 0.065; dead 0.336 of 0.336). The cap therefore removed
  nearly all D evidence along with the false D, leaving too little to pay the switching costs.
  The still cue measured over 1 s does not separate the states either (0.024 alive, 0.025
  dead). Lesson: untracked ball cannot carry dead-ball detection in either direction, and
  changing five values at once made the result hard to attribute.
- **Decision:** drop. E001's exact values restored in PR #9 (verified on synthetic data: the
  same labels as E001's code and config; on the real data the sha256 equals E001's
  `6501615e…`, confirmed by the human run). Measured rates stay noted in `config.py` for one-at-a-time changes.
- **Follow-ups:** E003, a players-movement cue (median player speed over ±1 s: players walk at
  stoppages and stay tracked when the ball does not), calibrated with the new
  `inspect_cues.py` rows before it is switched on. Then Smart Tagging set pieces as restart
  evidence, using the Team semantics above.

### E003 — Player-movement cue

- **Date / author / commit:** 2026-10-21, Hack Mavericks, PR #11. Design and criterion were fixed
  in PR #10, before the code; the code follows them with no value changed.
- **Hypothesis:** players stay tracked when the ball does not. In play they jog and sprint; at a
  stoppage they walk. A cue on the players' median speed should add dead-ball evidence where it
  belongs and, just as important, evidence *against* D during open play with the ball lost, which
  is where E001 over-calls D.
- **Measured** (`tools/inspect_cues.py`, public window, human-run; players' median speed, mean
  over ±1 s; share of frames below each threshold):

  | threshold | alive (n = 6031) | dead (n = 1469) | dead / alive |
  |---|---|---|---|
  | < 0.75 m/s | 0.007 | 0.000 | — |
  | < 1.00 m/s | 0.038 | 0.233 | 6.1 |
  | **< 1.25 m/s** | **0.104** | **0.624** | **6.0** |
  | < 1.50 m/s | 0.210 | 0.797 | 3.8 |
  | < 2.00 m/s | 0.442 | 0.863 | 2.0 |

  Same run, other diagnostics: untracked runs > 10 s are 0.000 of alive frames vs 0.249 of dead
  (a strong cue for a later, separate experiment); the 1-second still-ball cue does not separate
  the states (0.024 vs 0.025).
- **Design** (the only change against E001):
  - feature `features.players_median_speed(tracking, window=fps)`: already implemented and tested;
  - binary cue "players slow": median speed < `slow_speed = 1.25` m/s (best ratio with useful
    recall);
  - `slow_alive = 0.10`, `slow_dead = 0.62` (the measured rates, rounded); unknown where no
    player is tracked;
  - added to the alive/dead part of `evidence.log_emissions` like the other `_binary` cues;
    every other value stays at E001's.
- **Change** (`evidence.py`, `config.py`): new keys `slow_speed = 1.25`, `slow_alive = 0.10`,
  `slow_dead = 0.62`; `slow_speed = None` switches the cue off, giving E001's labels exactly
  (tested). The window is ±1 s (`spec.fps`), the one `inspect_cues.py` measured. Tests: walking
  players shift D − H by log(0.62/0.10), running players by log(0.38/0.90), the team margin is
  untouched, and frames with no tracked player get no slow-cue evidence. `synthetic_eval.py` gains an
  `hmm no-slow (E001)` variant.
- **Synthetic** (5 full seeds; correctness only):

  | variant | spellF1 | macroF1 | D predicted | D true |
  |---|---|---|---|---|
  | E001 (no slow cue) | 0.551 | 0.835 | 0.331 | 0.324 |
  | E003, real-data rates (0.10 / 0.62) | 0.458 | 0.688 | **0.112** | 0.324 |
  | slow cue, generator rates (0.01 / 0.18) | 0.573 | 0.855 | 0.307 | 0.324 |

  The drop comes from the generator, not the code. Synthetic players barely slow down at a
  stoppage: 18% of dead frames fall below 1.25 m/s, against 62% measured on the real window
  (alive: 0.1% vs 10%). With `slow_dead = 0.62`, the 82% of synthetic dead frames that are not
  slow each count as evidence against D. With the generator's own rates the cue does what it
  should: D share stays right and nothing breaks. The third row checks the mechanism and was
  not used to choose a value. Values stay at the measured real rates, as specified.
- **Risk this exposes for the real run:** the cue trusts that the public window's rates
  (0.624 dead) hold over the whole match. If stoppages over the match are much less often
  "walking" than in that window, D collapses as it does on the synthetic data, and the pre-set
  criterion's "D share below 30%" revert catches it.
- **Pre-set criterion for the real run:**
  - **keep** if the full-match D share moves from 45.3% into 30–42% while possessions stay
    ≥ 200 with a 10–20 s mean;
  - **revert** if the D share rises, falls below 30%, or possessions fall below 200;
  - the public window is reported for correctness only.
- **Commands for the human after merging:**
  `run_pipeline.py --out evaluation/results/hm_hmm_e003.csv`, then
  `run_evaluation.py --no-detail`; paste back the JSON summary and the board.
- **Full-match plausibility** (human-run, aggregates only):

  | | E001 | E003 | pre-set criterion |
  |---|---|---|---|
  | D share of graded frames | 45.3% | **36.7%** | into 30–42% |
  | possessions | 316 | 275 | ≥ 200 |
  | mean duration | 11.0 s | 14.6 s | 10–20 s |
  | home share | 0.611 | 0.599 | |
  | sha256 | `6501615e…` | `d6aee597…` | |

  D share = 58,323 D of 158,857 graded frames (no `X`).
- **Public window** (correctness only): passes. 17 possessions (truth 17), mean 16.5 s (truth
  14.2 s); macroF1 0.6831 -> 0.6825, unchanged; spellF1 0.3771 -> 0.4000, inside the ±0.33
  noise band and ignored as the rules require.
- **Decision:** keep. All three parts of the pre-set criterion are met: D share inside 30–42%,
  possessions ≥ 200, mean inside 10–20 s. E003 is the submitted model from PR #11. The fear
  raised by the synthetic run (D collapsing) did not happen on the real data, which supports
  reading that drop as a generator mismatch.
- **Follow-ups:** make the generator's players walk at stoppages (a `make_synthetic.py` change,
  kept out of this experiment so E003 stays a single change).

### E004 — Long untracked-ball runs as dead-ball evidence (dropped before implementation)

- **Status:** dropped by its pre-set rule after the full-match measurement (below). Only the
  diagnostic was implemented (`inspect_cues.py` table 4, PR #13); the model is unchanged (E003).
- **Hypothesis:** the ball goes untracked for long stretches mostly at long stoppages (goals,
  injuries, substitutions, balls into the stands), and those may be where the D share comes from
  after E003. A run's *length* says more than any single untracked frame, which by itself barely
  separates the states (E002).
- **Measured so far** (public window, E003 table): untracked runs > 10 s cover 0.000 of alive
  frames (n = 6031) and 0.249 of dead frames (n = 1469).
- **Why the public alive rate cannot be used:** that window loses the ball on 6.5% of in-play
  frames, against 41% over the match (E002). A 0.000 measured there would make the cue close to a
  veto over the full match, the same transfer error that sank E002. The alive rate has to come
  from the full match.
- **Measurement first** (`inspect_cues.py` table 4, human-run; aggregates only): over the decoded
  frames of the full match, the share inside untracked runs > 10 s, overall and split by E003's
  prediction (pred H/A, pred D).
- **Rule for the values, fixed now:**
  - `long_gap_dead = 0.25`: the public dead rate. Stoppages are less sensitive to how well the
    window is tracked than open play.
  - `long_gap_alive` = the larger of two full-match estimates, with a floor of 0.02:
    (a) table 4's pred H/A rate; (b) the mixture estimate (all − 0.367 × 0.25) / 0.633, using
    E003's D share of 36.7%. The larger one is the more cautious: a weaker cue.
  - If that gives `long_gap_alive` ≥ 0.15 (a dead/alive ratio under ~1.7, close to the
    untracked cue already in place), **drop E004 without implementing it**. It would add little
    beyond the untracked cue and would double-count the same frames.
- **Design** (if it goes ahead, the only change against E003): a binary cue "frame inside an
  untracked run longer than `long_gap_s = 10` s", added to the alive/dead part of
  `evidence.log_emissions` like the other `_binary` cues, known on every frame. It overlaps the
  untracked cue on purpose: that cue says "the ball is lost", this one says "lost for long".
- **Pre-set criterion for the real run:**
  - **keep** if the D share stays within 30–42%, possessions ≥ 200 with a 10–20 s mean, and the
    public window still passes (17 ± 3 possessions, macroF1 not below E003's 0.6825 by more than
    0.01);
  - **revert** if the D share leaves 30–42%, possessions fall below 200, or macroF1 drops more
    than 0.01.
  - Limitation, stated in advance: the full-match aggregates can show E004 is not broken, not
    that it is better. If it passes, it is kept on the strength of the reasoning above.
- **Commands for the human** (now, before any model change):
  `.venv/bin/python tools/inspect_cues.py`; paste back table 4 only.
- **Synthetic:** the generator never loses the ball for more than 10 s (table 4 on seed 0:
  0.000 in every column), so synthetic runs cannot exercise this cue. A follow-up for
  `make_synthetic.py`, with the walking-at-stoppages fix from E003.
- **Full-match measurement** (table 4, human-run, 155,923 decoded frames; aggregates only):

  | cue | all | pred H/A (100,534) | pred D (55,389) |
  |---|---|---|---|
  | ball not tracked | 0.411 | 0.172 | 0.844 |
  | untracked, run > 1 s | 0.411 | 0.172 | 0.844 |
  | untracked, run > 3 s | 0.411 | 0.172 | 0.844 |
  | untracked, run > 10 s | 0.351 | 0.120 | 0.770 |
  | untracked > 10 s & slow | 0.250 | 0.001 | 0.703 |

- **Applying the rule:** (a) pred H/A rate 0.120; (b) mixture (0.351 − 0.367 × 0.25) / 0.633 =
  0.410. The larger is 0.41 ≥ 0.15, so E004 is dropped. A cue at 0.25 dead vs 0.41 alive would
  point *away* from D.
- **What the rule's input hides:** estimate (b) trusts the public dead rate 0.25. If long runs
  covered far more of the match's real stoppages than that, the alive rate would be lower. The
  measurement cannot tell these apart without labels, and the rule was written to be cautious
  for exactly this reason. Not revisited.
- **What the measurement shows instead** (for later experiments):
  - On the full match, as on the public window, the ball is never lost briefly: every untracked
    frame is in a run over 3 s, and 85% of them in runs over 10 s. The tracker loses the ball in
    long blocks, not flickers.
  - E003's D is now essentially "ball lost for > 10 s while the players walk" (0.703 of pred D,
    0.001 of pred H/A). Movement, not the ball, decides D inside long gaps.
  - **17.2% of predicted-alive frames have no ball at all**, so no team cue: about 17,000 frames
    whose side comes only from the decoder carrying the last side through the gap. A possession
    change inside a gap is invisible today. That is approach-plan step 4 (a team model on player
    configuration for long gaps) and probably the largest remaining source of error.
- **Decision:** drop, by the pre-set rule. No code change to the model.

### E005 — Team inside ball gaps (E005a harness, E005b classifier, E005c decoder)

- **Status:** revised 2026-10-21 after the literature review. The first version (a 3-feature
  logistic shape model, PR #15) is superseded by E005b below. Carried over unchanged: the
  problem, the 0.60 drop gate, the real-run criterion, the home-share sanity check.
- **Problem** (E004 table 4): 17.2% of predicted-alive frames have no ball, so no team cue at all.
  There the decoder carries the last side through the gap, so a possession that changes hands
  inside a gap is lost. E003 predicts 275 possessions; the public window's rate (17 in 5 min)
  scaled to the graded time suggests roughly 350 (a rough figure from 17 possessions). Under-
  segmentation in gaps is the most likely cause.
- **Hypothesis:** players show who has the ball without the ball (Ball Radar, TranSPORTmer:
  ~81–86% from trajectories alone on cleaner data). The side in possession pushes up and spreads
  out; the side without it drops back, closes up and presses.
- **Real-run criterion for E005c** (unchanged from PR #15):
  - **keep** if possessions rise above 275 and stay ≤ 400, mean 10–20 s, D share within 30–42%,
    and the public window still passes (17 ± 3 possessions, macroF1 not below 0.6825 by more
    than 0.01);
  - **revert** otherwise;
  - secondary, not decisive: the home share should not move clearly away from ~0.65 (rules).

#### E005a — Masked-ball harness (design approved in PR #16; implemented)

- **Purpose:** a full-match measure of how well the pipeline survives losing the ball, with no
  ground truth. It is the yardstick for E005b and E005c, which change behaviour only inside gaps,
  exactly where the public window has too few frames (391) to say anything.
- **Method:**
  1. *Reference:* run the pipeline unmasked. Its labels are the reference on the frames we will
     mask: those frames had the ball seen, so this is the best label we have there.
  2. *Gap lengths:* the real untracked runs of the match, restricted to runs that are mostly
     (> 50% of frames) predicted alive by the reference. These are the gaps in play we want to
     imitate; the long gaps at stoppages are a different problem (E003, E004). Read at run time
     from the bundle and never printed beyond quantiles.
  3. *Placement:* draw a gap start among frames where the ball is seen and the reference is
     alive, then a length from step 2; hide the ball on the seen frames in that stretch. Start
     weights, mildly biased the way real gaps are: ×2 if 3+ players are within `mid_m` (5 m) of
     the ball (crowd), ×2 if within ±2 s of an H↔A change in the reference (contests, long
     balls), so ×4 if both. Repeat until the masked frames reach the share of untracked frames
     among the reference's alive frames (0.172 on the real match): the masked run then sees
     about twice the real gap density in play.
  4. *Masked run:* the same pipeline on the masked tracking. Only `tracking.ball` changes, so
     every ball-derived cue (nearest player, visible, speed, outside) sees the gap.
  5. *Seeds:* 5 fixed seeds (0–4); report mean ± std over them.
- **Outputs** (aggregates only):
  - *change recall:* of the reference's H↔A changes inside a masked stretch or within ±1 s of
    one, the share the masked run also has within ±1 s (the scorer's transition tolerance);
  - *possession count* masked vs reference, and *mean IoU* of each reference possession that
    touches a mask with its best-matching masked-run possession;
  - *spellF1-proxy:* the official `possession_metrics.evaluate` with the reference as truth;
  - *frame agreement* on the masked frames;
  - with `--gt` on the public window: the same numbers against the real truth, n stated
    (indicative only: 5 minutes).
- **Hyperparameters, each with its reason:** crowd = 3+ players within `mid_m` (existing
  config, 5 m); change neighbourhood ±2 s (contests and long balls around a turnover; Lang et al.:
  the seconds around a change matter most); weights ×2 (a mild bias, not a model of where gaps
  happen); masked share = 0.172 (measured, E004); 5 seeds. None is tuned on any result.
- **What it cannot say:** the reference is our own model, not the truth. The harness measures
  how much *worse* the pipeline gets when the ball goes missing, not how good it is. Real gaps
  sit where tracking is hardest, so the masked gaps are probably easier than real ones.
- **Code:** `possession/masking.py` (gap lengths, placement, masking; small and unit-tested),
  `tools/masked_harness.py` (runs reference + 5 masked runs, prints the table). Tests on synthetic
  data: masks only seen frames, reaches the target share, is deterministic per seed, recall is
  1.0 when nothing is masked, and the metric helpers on hand-built label arrays.
- **Synthetic first:** run on 5 synthetic seeds, where the real truth is also known, to check
  the proxy tracks the truth-based numbers in direction.
- **Implemented as designed**, one detail made concrete: the target hidden share is computed at run
  time from the bundle (untracked share of the reference's alive frames), not hard-coded as 0.172,
  so the same code is right on synthetic bundles. Placement constants live in
  `possession/masking.py`; the change tolerance is `spec.fps` (±1 s).
- **Synthetic** (5 full bundles × 5 masking seeds; correctness only). Table 1 = masked run vs our
  own unmasked run (the proxy); table 2 = against the synthetic truth, around the same frames:

  | bundle | proxy: change recall | proxy: spellF1 | truth: change recall unmasked → masked | truth: spellF1 unmasked → masked | possessions unmasked → masked |
  |---|---|---|---|---|---|
  | 0 | 0.123 | 0.527 | 0.236 → 0.044 | 0.458 → 0.208 | 280 → 143 |
  | 1 | 0.174 | 0.580 | 0.228 → 0.059 | 0.487 → 0.252 | 261 → 149 |
  | 2 | 0.118 | 0.536 | 0.259 → 0.043 | 0.445 → 0.214 | 278 → 146 |
  | 3 | 0.069 | 0.519 | 0.191 → 0.038 | 0.435 → 0.189 | 258 → 136 |
  | 4 | 0.073 | 0.521 | 0.225 → 0.038 | 0.467 → 0.210 | 274 → 138 |

  Seed-to-seed std is ≤ 0.025 on every rate, so 5 seeds suffice. On every bundle, the proxy and
  the truth-based scores fall together, so the proxy reads in the right direction. The size is not
  transferable: the generator loses the ball on ~42% of alive frames in short bursts (median
  0.7 s), against 17% in runs of 3 s and more on the real match, so the synthetic masking is far
  heavier. The pattern matches E005's hypothesis: without the ball, the decoder carries the last
  side through the gap, and about half the possessions disappear.
- **Command for the human** (after merge; ~1–2 minutes):
  `.venv/bin/python tools/masked_harness.py --gt evaluation/gt/BAR-ATH_possession_gt_public.csv`;
  paste back the whole output (context lines, table 1, table 2). Table 2 covers only the public
  5 minutes: indicative, n stated.
- **Real data** (human-run, E003 model, aggregates only). Context: 275 reference possessions;
  62 real gaps in play, length p50 7.2 s, p90 21.2 s; untracked share of alive frames 0.172,
  hidden 0.174.

  | vs unmasked reference (5 seeds) | value |
  |---|---|
  | change recall | 0.115 ± 0.052 (56 changes near hidden frames per seed) |
  | mean IoU of touched possessions | 0.504 ± 0.029 (116 touched) |
  | agreement on hidden frames | 0.581 ± 0.057 |
  | spellF1-proxy | 0.831 ± 0.017 |
  | possessions | 275 → 241 ± 4 |

  Public window vs truth (4 ± 2 changes, 7 touched possessions: not readable): spellF1
  0.400 → 0.367, possessions 17 → 14.
- **Reading:** with the ball gone, the pipeline misses almost 9 in 10 side changes and loses ~34
  possessions. Back-of-envelope for the real gaps: ~17,000 alive frames without a ball, at roughly
  one possession change per 11–12 s of play, hold some 60 changes; catching most of them would
  take the count from 275 towards the ~350 suggested by the public window. That is E005's target.
- **Not yet known:** whether the 42% disagreement on hidden frames is mostly "called dead" or
  "wrong side". Added `to_dead` and `wrong_side` to `compare()` (PR after #17); one rerun answers it,
  and E005b's go-ahead depends on it (below).
- **Decision:** none; a measuring tool. These numbers are the baseline E005c must beat.
- **Split rerun** (human-run, same masking, vs reference): `to_dead` 0.167 ± 0.036,
  `wrong_side` 0.264 ± 0.061 (together 0.431 = 1 − agreement 0.581, as they must be). Wrong side
  dominates, so E005b goes ahead by its pre-set condition. Called-dead-in-gaps is the next loss
  after E005, about 0.17 of hidden alive frames: a candidate after E005c.

#### E005b — Team-from-movement classifier (design approved in PR #18; implemented)

- **Go-ahead condition, fixed now:** proceed if the E005a rerun shows `wrong_side` ≥ `to_dead` on
  hidden frames (the loss is mostly the team, which E005b addresses). If `to_dead` dominates, stop
  and propose a dead/alive fix for gaps first (Borghesi et al.: dead/alive, then team).
- **Attack direction, label-free:** per half, the team whose players' mean x over the half is
  lower defends the low-x goal. All x and x-velocities are then flipped so each team attacks +x.
- **Per-team features at one frame** (outfield = all but the team's deepest player):
  1. centroid x; 2. centroid x-velocity; 3. spread: std x; 4. std y; 5. compactness: mean
  distance to the team centroid; 6. defensive-line height: x of the deepest outfield player;
  7. backward running: players with x-velocity below −4 m/s (14.4 km/h, the usual "running"
  band); 8. pressing distance: mean of the 3 smallest player-to-nearest-opponent distances.
  Velocities from positions over ±0.2 s. Missing players are skipped; a team with fewer than
  6 tracked players gives NaN (the model handles NaN natively).
- **Context:** both teams' 8 features at offsets 0, ±0.4, ±1, ±2, ±4, ±8 s (11 offsets) →
  176 features per frame. Acausal, since the decoder is offline.
- **Symmetry, so no home prior is learned:** each training frame is used twice, as is and with
  the teams swapped (label flipped); prediction averages p(x) and 1 − p(swap(x)), so swapping
  the teams flips the probability exactly.
- **Model:** scikit-learn `HistGradientBoostingClassifier` with library defaults (no tuning),
  early stopping off, `random_state=0`, fitted and predicted under `threadpool_limits(1)` so the
  result does not depend on the machine's thread count. New pinned dependency:
  `scikit-learn==1.5.2` (supports numpy 2 and Python 3.12). `make reproduce` must stay byte-identical.
- **Training labels (self-supervised, full match, no ground truth):** frames the E003 reference
  calls alive, ball seen, nearest player within `near_m` (2 m) → that player's side; every 5th
  such frame (neighbouring frames are near-duplicates).
- **Half-to-half cross-validation:** fit on half 1, predict half 2, and the reverse. Reported
  (new `tools/team_model_check.py`, aggregates only):
  - (a) accuracy on held-out training-type frames;
  - (b) accuracy on *all* held-out reference-alive frames, against the reference label: the
    harder, more relevant set, since gap frames are not close-control frames;
  - (c) accuracy on E005a's hidden frames (5 seeds), by gap length < 5 s, 5–15 s, > 15 s;
  - (d) public window vs truth on alive frames with the ball untracked (~391 frames; indicative);
  - share of frames with confident output (p ≥ 0.7 or ≤ 0.3) and accuracy within them.
- **Gate (carried over):** drop E005 if (b) < 0.60 or (d) < 0.5. Otherwise E005c is designed
  from these numbers.
- **Tests (synthetic):** direction inference on a hand-built half; each feature on hand-built
  positions; swap symmetry holds exactly; deterministic fit (two fits, identical predictions);
  cross-validation never trains on the predicted half.
- **Not doing:** a pre-trained model or weights in the repo (it is refit from the bundle on each
  run); feature selection or hyperparameter search.
- **Implemented as designed** (`possession/team_model.py`, `tools/team_model_check.py`,
  `tests/test_team_model.py`, 10 tests; 116 pass). The pipeline does not use it yet: E005c will.
  Details made concrete: "deepest player" is per frame (usually the keeper); pressing distance
  uses the team's outfield players against every opponent; the 0–5 / 5–15 / 15+ s bands in (c)
  use the length of each contiguous hidden stretch. Dependencies pinned in `requirements.txt`:
  scikit-learn 1.5.2, scipy 1.18.1, joblib 1.6.0, threadpoolctl 3.7.0.
- **Synthetic** (5 full bundles; correctness only):

  | bundle | (a) training-type | (b) all reference-alive | (d) truth-alive, untracked | confident share (b) |
  |---|---|---|---|---|
  | 0 | 0.720 | 0.666 | 0.705 | 0.56 |
  | 1 | 0.737 | 0.680 | 0.705 | 0.57 |
  | 2 | 0.743 | 0.693 | 0.702 | 0.61 |
  | 3 | 0.693 | 0.654 | 0.660 | 0.61 |
  | 4 | 0.719 | 0.667 | 0.687 | 0.51 |

  Stable across bundles; every bundle passes the gate. Confident calls (p ≥ 0.7 or ≤ 0.3) are
  right ~75–80% of the time. The synthetic players follow formation anchors that shift with the
  ball, so this shows the code learns a shape signal, not how strong the real one is.
- **Command for the human** (after merge; a few minutes, ~1 GB RAM):
  `.venv/bin/python tools/team_model_check.py --gt evaluation/gt/BAR-ATH_possession_gt_public.csv`;
  paste back the whole output (table and gate line).
- **Real data** (human-run; half-to-half; 8,913 training frames over both folds):

  | frames | accuracy | n | confident share | confident accuracy |
  |---|---|---|---|---|
  | (a) training-type | 0.836 | 44,559 | 0.848 | 0.882 |
  | (b) all reference-alive | **0.789** | 100,534 | 0.783 | 0.849 |
  | (c) hidden, 0–5 s | 0.712 | 5,071 | 0.709 | 0.749 |
  | (c) hidden, 5–15 s | 0.783 | 47,286 | 0.785 | 0.853 |
  | (c) hidden, 15+ s | 0.844 | 31,358 | 0.839 | 0.887 |
  | (d) public, truth-alive, untracked | **0.949** | 391 | 0.985 | 0.964 |
  | public, truth-alive, all | 0.864 | 6,031 | 0.890 | 0.918 |

  Gate: (b) 0.789 ≥ 0.60 and (d) 0.949 ≥ 0.5 → **PASS**. (d) is 391 frames from a handful of
  possessions: indicative only.
- **Leakage check** (a result this good must be checked): the classifier never sees the ball or
  any ground truth; each half is predicted by a model fitted on the other half's *pipeline*
  labels; the public window (first half) is predicted by the second-half model. Against the
  truth it is *better* than against the reference (0.864 vs 0.789), which fits the reference's own
  errors (it is itself ~86% right), not leakage. The level matches the literature (81–86%).
- **Decision:** pass; E005c is designed from these numbers.

#### E005c — Classifier probabilities as gap-only emissions (design approved in PR #20; implemented)

- **Change** (the only one against E003): a second team cue on frames where the ball is
  untracked, from E005b's classifier. Frames with the ball seen keep the nearest-player cue
  alone, so nothing is double-counted.
- **Two passes, deterministic:** pass 1 is E003 exactly. Its labels train the classifier
  half-to-half (`team_model.cross_half`, as validated: each half predicted by the other half's
  model). Pass 2 decodes again with the gap cue added. Adds ~1–2 min per run.
- **Emission**, in the existing team cue's form: with p = P(H) from the classifier, the side it
  favours is "agree". The alive states score log(acc) for agreeing and log(1 − acc) otherwise,
  and D scores log 0.5; then the usual `weight` (0.2).
  - confident band (p ≥ 0.7 or ≤ 0.3): `gap_acc_strong`; otherwise `gap_acc_weak`.
- **Value rule, fixed now:** the pooled accuracies on E005a's hidden frames (row (c), 3 bands,
  83,715 frames), used as measured, no shrinkage:
  - `gap_acc_strong = 0.86` (0.861, n = 67,024); `gap_acc_weak = 0.56` (0.564, n = 16,690).
  - Why no shrinkage (the PR #15 draft shrank halfway): (c) is measured on gap-shaped frames
    against the reference, which is itself wrong ~14% of the time, so it already understates
    accuracy (public window: 0.864 against the truth vs 0.789 against the reference). The
    evidence weight 0.2 already discounts correlated frames. Shrinking again would count the
    same caution twice.
  - What this means for the decoder: a confident frame adds 0.2 × log(0.86/0.14) = 0.36 to the
    favoured side; one switch costs 6.06; so a change of side inside a gap needs ~17 confident
    frames (0.7 s) against the carried side, and a flicker out and back ~34 (1.4 s). The
    classifier sees ±8 s of context, so its output is smooth over seconds.
  - `gap_acc_strong = None` switches the cue off (E003 exactly), tested.
- **Pre-set criterion for the real run** (E005's, plus E005a's baseline):
  - **keep** if possessions rise above 275 and stay ≤ 400, mean 10–20 s, D share within 30–42%,
    the public window still passes (17 ± 3 possessions, macroF1 ≥ 0.6725), **and** the E005a
    harness change recall rises above 0.167 (E003's 0.115 + 1 std);
  - **revert** otherwise;
  - secondary, not decisive: home share not moving clearly away from ~0.65; `to_dead` on hidden
    frames reported (the gap cue makes alive more likely than D where it is confident).
- **Tests (synthetic):** the cue is zero on frames with the ball seen; agreeing raises the
  favoured side by the stated amount; `None` gives E003's labels; the pipeline output is
  byte-identical across two runs.
- **Implemented as designed** (`evidence._add_gap_team`, `pipeline._hmm`, `config.py`): values 0.86 /
  0.56 / confidence 0.7. p = 0.5 exactly favours no side (no cue). 3 new tests; 119 pass.
  `synthetic_eval.py` now compares against `hmm no-gap (E003)`.
- **Synthetic** (5 full bundles; correctness only):

  | variant | spellF1 | macroF1 | poss ratio | wrong side (alive frames) |
  |---|---|---|---|---|
  | E003 (no gap cue) | 0.458 | 0.688 | 0.70 | 0.100 |
  | E005c, real-data values (0.86 / 0.56) | 0.429 | 0.650 | 0.96 | 0.120 |
  | E005c, generator rates (0.78 / 0.55) | 0.446 | 0.662 | 0.90 | 0.112 |

  Possessions move to the true count, but frame accuracy falls. The reason is measurable: on
  synthetic untracked alive frames, carrying the last side forward (E003) is right 0.80–0.92 of
  the time at every gap length (bundles 0–2: 0–1 s, 1–3 s, 3+ s), the classifier only ~0.70. The
  generator's gaps are short (median 0.7 s) and its shape signal weak, so a gap cue can only add
  errors there. The real match is the opposite regime: gaps in play have a median of 7.2 s, and
  on E005a's hidden frames carry-forward keeps the reference's side only ~0.57 of the time
  (1 − wrong_side 0.264 − to_dead 0.167), while the classifier gets 0.80 on the same kind of
  frames (E005b row (c)). Values stay as specified; the real run decides.
- **Commands for the human** (after merge; the harness now runs E005c six times, ~10–15 min):
  1. `.venv/bin/python run_pipeline.py --out evaluation/results/hm_hmm_e005c.csv`
  2. `.venv/bin/python evaluation/run_evaluation.py --no-detail`
  3. `.venv/bin/python tools/masked_harness.py --gt evaluation/gt/BAR-ATH_possession_gt_public.csv`
  4. `make reproduce`

  Paste back the JSON summary, the board, the harness output and the reproduce line.
- **Real data** (human-run, aggregates only):

  | | E003 | E005c | pre-set criterion |
  |---|---|---|---|
  | possessions | 275 | **308** | > 275 and ≤ 400 |
  | mean duration | 14.6 s | 13.3 s | 10–20 s |
  | D share of graded frames | 36.7% | 35.4% (56,166 / 158,857) | 30–42% |
  | public window: possessions / macroF1 | 17 / 0.6825 | 16 / 0.6980 | 17 ± 3 / ≥ 0.6725 |
  | harness change recall (vs own reference) | 0.115 ± 0.052 | **0.284 ± 0.044** | > 0.167 |
  | harness wrong_side / to_dead | 0.264 / 0.167 | 0.161 / 0.141 | (reported) |
  | harness agreement / possessions masked | 0.581 / 241 of 275 | 0.700 / 293 of 308 | (reported) |
  | home share (sanity) | 0.599 | 0.594 | not clearly away from ~0.65 |
  | sha256 | `d6aee597…` | `a8ee1b28…` | |

  `make reproduce`: byte-identical. Public spellF1 0.4000 → 0.4242, inside the noise band and
  ignored as the rules require.
- **Reading:** every part of the criterion is met. Hiding the ball now costs 15 possessions
  instead of 34, and 28% of side changes near a gap survive instead of 11.5%. One caveat: each
  harness number is measured against its own model's unmasked run, so the two recalls measure
  robustness to a lost ball, not accuracy. Public window, table 2 (3 ± 2 changes, 7 possessions,
  not readable): against the truth, the masked E005c run agrees more than the unmasked one
  (0.856 vs 0.715; wrong side 0.072 vs 0.206). It hints that the classifier can beat the
  nearest-player cue on some seen frames: a question for E006, not evidence.
- **Decision:** keep. E005c is the submitted model from PR #21 (`make submission` by the human).
- **Follow-ups:** `to_dead` 0.141 on hidden frames is now the largest single loss in gaps;
  E006 (visible-ball change rules) is next on the roadmap.

### E006 — Last-touch team on seen-ball frames (design approved in PR #23; E006a built)

- **Status:** E006a (measurement) implemented; E006b waits for the gate.
- **Problem:** on frames with the ball seen, the team cue is the side of the nearest player,
  trusted by distance (0.873 right within 2 m, 0.647 at 2–5 m, none beyond: E002 measurements).
  It is wrong in three recurring situations: a pass in flight passing close to an opponent, a
  duel where both sides are near, and a loose ball. The ground truth names an *owner*, which
  plausibly stays with the side of the last controlled touch while the ball travels.
  Public-window hint (E005c harness table 2, tiny): around hidden frames the classifier beat the
  nearest-player cue against the truth, so the seen-ball cue has room to improve.
- **Hypothesis:** a "last touch" signal fits the owner definition better than "nearest player":
  the side of the player who last touched the ball keeps it until a player touches it again
  (Vidal-Codina et al.; Link & Hoernig). A duel (both sides within reach) does not flip it.
- **Why measure first:** the papers use corrected or GPS tracking. Our ball is automatic, with
  position jitter of the order of 0.2 m (DATA.md: lost on 41% of graded frames). An acceleration
  threshold such as Link & Hoernig's 4 m/s² is far below what jitter alone produces when it is
  differentiated twice at 25 fps, so every threshold is set from our own data, by rules fixed here.
- **Touch detection, label-free:**
  - ball smoothed with a Savitzky–Golay filter, 7 frames, order 2 (Vidal-Codina et al.), only
    within runs of seen frames;
  - velocity change Δv(t) = |v(t + 3) − v(t − 3)| in m/s (0.12 s either side);
  - a *touch* is a local maximum of Δv above τ with a player within r = 1.0 m (the possession
    zone of Vidal-Codina et al.);
  - **τ by rule:** the 99th percentile of Δv on full-match seen frames where the nearest player is
    more than 5 m away (ball in free flight: what noise and bounces alone produce). No labels.
  - *duel:* players of both sides within r at the touch → no attribution (keep the previous side).
- **Last-touch team:** carried forward from each attributed touch while the ball stays seen;
  reset to unknown after any untracked run longer than 1 s (gaps belong to E005c's cue) and at
  every predicted stoppage (pass 1 labels).
- **E006a: measurement** (new `tools/inspect_touches.py`, human-run; aggregates only):
  - full match, label-free: τ; touches per minute of play (football: roughly 15–25 attributed
    touches per minute is plausible); share of touches in duels; share of seen alive frames with a
    last-touch side; agreement with the nearest-player side;
  - public window against the truth, on truth-alive seen frames: accuracy of the last-touch side
    vs the nearest-player side on the *same* frames, overall and split by nearest distance (< 2 m,
    2–5 m, > 5 m); n stated. This is a calibration of rates (as for every cue so far), not a
    version choice.
- **Gate to build E006b, fixed now:** last-touch accuracy on the public truth-alive seen frames at
  least 0.03 above the nearest-player accuracy on the same frames, and coverage ≥ 0.5 of seen
  alive frames. Otherwise drop E006 before it touches the model.
- **E006b, if the gate passes:** on seen frames with a last-touch side, it replaces the
  nearest-player cue (one team cue per frame, as now); elsewhere nothing changes. Its accuracy is
  the public measured rate, rounded, by the same rule as E003's values.
- **Pre-set criterion for E006b's real run:** keep if possessions stay within 290–400 with a mean
  of 10–20 s, D share within 30–42%, and the public window passes (16 ± 3 possessions, macroF1 ≥
  0.6880, i.e. E005c's 0.6980 − 0.01); revert otherwise. Stated limitation: there is no
  full-match accuracy check for seen frames, so a pass means "not broken and better calibrated",
  not "proven better".
- **Tests (synthetic + hand-built):** Savitzky–Golay coefficients; a hand-built kick (ball turns
  sharply next to a player) is a touch, a straight free flight is not; duel gives no attribution;
  carry-forward resets after a > 1 s gap and at stoppages.
- **Synthetic:** the generator's ball has a holder and passes, so touches exist; it checks the
  code, not the thresholds (the generator's ball jitter is its own).
- **E006a implemented as designed** (`possession/touches.py`, `tools/inspect_touches.py`,
  `tests/test_touches.py`, 8 tests; 127 pass). Detail made concrete: a touch is the frame where Δv
  peaks; alive frames and stoppages come from the current pipeline (E005c).
- **Synthetic** (5 full bundles; correctness only):

  | bundle | τ (m/s) | attributed touches / min | coverage | last touch | nearest player | gate |
  |---|---|---|---|---|---|---|
  | 0 | 16.5 | 6.4 | 0.491 | 0.878 | 0.782 | FAIL (coverage) |
  | 1 | 18.9 | 4.8 | 0.438 | 0.852 | 0.791 | FAIL (coverage) |
  | 2 | 18.5 | 4.3 | 0.384 | 0.838 | 0.790 | FAIL (coverage) |
  | 3 | 17.5 | 5.4 | 0.448 | 0.842 | 0.788 | FAIL (coverage) |
  | 4 | 17.6 | 5.6 | 0.473 | 0.851 | 0.792 | FAIL (coverage) |

  On the same frames, last touch beats nearest player by 0.05–0.10 on every bundle; the gain is in
  the 2–5 m and > 5 m bands (bundle 0: 0.835 vs 0.606 and 0.886 vs 0.591), while within 2 m the two
  agree (0.889 vs 0.900). Coverage fails because few touches are found: the generator's ball
  jitter puts the free-flight threshold at ~17 m/s, so only hard touches pass (4–6 per minute
  against a plausible 15–25), and its short gaps reset the side often. The real ball's jitter is
  its own, so the real run decides; the gate is not changed.
- **Command for the human** (after merge; a few minutes):
  `.venv/bin/python tools/inspect_touches.py --gt evaluation/gt/BAR-ATH_possession_gt_public.csv`;
  paste back the whole output.
- **Real data, first run** (human-run): τ 9.35 m/s; 0.015 touches per minute of play (about one
  in the match); coverage 0.004; no public frames with a last-touch side. Gate: FAIL.
- **Reading:** this is a broken detector, not evidence about the idea. A football match has
  thousands of touches; one detected touch means a rule is not firing on this tracking. Suspects,
  all label-free to check: (1) reach 1 m is too short for automatic body-centre positions; (2) the
  ball is lost at the feet, so the ±4 frames a touch needs are rarely all seen; (3) an irregular
  ball track (held positions, steps) makes τ meaningless.
- **Not done:** reading the pre-set gate as a verdict on E006. The gate assumed a detector that
  finds touches; it is kept unchanged for the calibrated comparison once the detector works.
- **Diagnostics added** (`inspect_touches.py` table 0, aggregates only): share of seen frames with
  Δv computable and in seen runs ≥ 9 frames; share of zero-movement steps and step length
  quantiles; share of seen frames and Δv quantiles by nearest-player distance (< 1, 1–2, 2–5,
  > 5 m); Δv peaks above τ per minute and where their nearest player sits.
- **Rule for any detector fix, fixed now:** a change must follow from table 0 alone (label-free),
  be stated here before the next run, and stay one change. Candidate rules, chosen by which
  suspect table 0 confirms:
  - (1) peaks above τ exist but their nearest player sits mostly beyond 1 m → reach = the median
    nearest-player distance at Δv peaks above τ, capped at 2 m;
  - (2) Δv is rarely computable near players → compute Δv over ±2 frames instead of ±3;
  - (3) many zero-movement steps → a held-position track; Δv-based touches are dropped and E006
    stops.
  The accuracy gate (last touch ≥ nearest + 0.03, coverage ≥ 0.5) is unchanged and still decides.
- **Command for the human** (after merge): the same command as before; paste back the whole
  output, now with table 0.
- **Diagnostics, real data** (table 0, human-run, 91,844 seen decoded frames; aggregates only):

  | | value |
  |---|---|
  | Δv computable / in seen runs ≥ 9 frames | 0.988 / 1.000 |
  | zero-movement steps; step p50 / p90 | 0.001; 0.203 / 0.567 m |
  | nearest < 1 m: share of seen; Δv p50 / p90 / p99 | 0.289; 1.2 / 3.7 / 6.1 m/s |
  | nearest 1–2 m | 0.232; 1.4 / 4.4 / 6.9 m/s |
  | nearest 2–5 m | 0.298; 1.6 / 4.8 / 8.2 m/s |
  | nearest > 5 m (free flight) | 0.181; 1.2 / 3.9 / 9.3 m/s |
  | Δv peaks above τ per minute; nearest at peak < 1 / 1–2 / 2–3 / 3–5 / > 5 m | 0.30; 0.03 / 0.13 / 0.23 / 0.23 / 0.39 |

- **Reading:** suspects (2) and (3) are ruled out (Δv is computable almost everywhere; the track
  does not hold positions). Suspect (1) formally fires (peaks sit mostly beyond 1 m), but its fix
  cannot help: there are only 0.3 peaks above τ per minute, so even a 2 m reach would give
  ~0.05 touches per minute. The real finding is beneath the rules: **near a player, the ball's
  velocity change is no larger than in free flight** (p99 6.1 m/s within 1 m vs 9.3 m/s beyond
  5 m). The ball track appears smoothed at source, so kicks do not stand out, while the largest
  jumps happen away from players (39% of peaks have nobody within 5 m: tracking jumps). The
  detector's premise, that a touch is a sharp velocity change next to a player, is false on
  this tracking.
- **Decision:** stop E006 (velocity-based touches). Not tuned further: any new threshold would be
  chosen after seeing this result. The code stays in the repo as a documented negative result;
  the model never used it (the submission is E005c).
- **What survives:** the synthetic result says "last side in control" can beat "nearest player"
  away from the ball (2–5 m, > 5 m), and on the real match 48% of seen frames have nobody within
  2 m. A proximity-based "last control" cue, needing no velocity, is a separate experiment
  if chosen (proposed in the next entry only on the human's go-ahead).

### E009 — Last-control side on seen-ball frames (design proposed; measure first)

- **Status:** design proposed for approval, chosen by the human after E006 stopped. Nothing
  implemented.
- **Why:** E006's synthetic result showed that "the side last in control" beats "the nearest
  side" where the nearest player is 2 m or more away (bundle 0: 0.835 vs 0.606 at 2–5 m, 0.886
  vs 0.591 beyond 5 m). On the real match, 48% of seen frames have nobody within 2 m (E006
  table 0), where the current cue is weak (0.647 at 2–5 m) or absent (> 5 m). E006 failed only on
  *detecting* touches from velocity, which this tracking cannot support. Proximity needs no
  velocity.
- **Control, by rule (no new fitted values):**
  - a frame is *held* by side S when S's nearest player is within `near_m` (2 m, existing config)
    and no opponent is within `near_m`. Both sides within 2 m is a duel: nobody holds it;
  - *control* needs at least K = 10 consecutive held frames (0.4 s) by the same side. Reason: a
    ball passing a player at pass speed (≥ 10 m/s, 0.4 m per frame) crosses a 2 m-radius disc in
    at most 10 frames even through its centre. So K = 10 excludes every fly-by at pass speed,
    while a reception or a dribble keeps the ball near a player for longer.
- **Last-control side:** set from the start of each controlled run, carried forward while the ball
  stays seen; reset after an untracked run longer than 1 s (gaps belong to E005c's cue) and on
  every stoppage (pass-1 labels). A duel keeps the previous side. Same carry-forward as E006
  (`touches.last_touch`), reused.
- **E009a: measurement** (`tools/inspect_touches.py` gains the control cue; aggregates only):
  - full match, label-free: controls per minute of play, share of held frames in duels,
    coverage of seen alive frames, agreement with the nearest-player side;
  - public window vs truth, truth-alive seen frames with a last-control side: accuracy of last
    control vs nearest player on the same frames, by nearest distance (< 2, 2–5, > 5 m).
- **Gate to build E009b, fixed now (E006's, unchanged):** last control ≥ nearest player + 0.03 on
  those public frames, and coverage ≥ 0.5 of full-match seen alive frames. Otherwise stop.
- **E009b, if the gate passes:** on seen frames with a last-control side, it replaces the
  nearest-player cue. Its accuracy is the public measured rate, rounded (E003's rule).
  Criterion for the real run (E006b's): possessions 290–400 with a 10–20 s mean, D share 30–42%,
  public window passes (16 ± 3 possessions, macroF1 ≥ 0.6880); revert otherwise. Limitation as
  before: no full-match accuracy check for seen frames.
- **Tests:** held, duel and control runs on hand-built frames (9 held frames are not control, 10
  are); a fly-by at 10 m/s through a player's disc is never control; carry-forward and resets as
  in E006.
