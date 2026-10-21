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

## Index

| ID | date | commit | idea | decision |
|---|---|---|---|---|
| E000 | 2026-10-20 | b7292ff | Reproduce the reference baseline with our own code | keep: IDENTICAL ×4 on real data |
| E001 | 2026-10-20 | PR #5 | 3-state Viterbi decoder instead of the window vote | keep: default model since PR #5 |
| E002 | 2026-10-20 | PR #8 | Recalibrate the evidence cues against measured rates | drop: D share 45% -> 8%; reverted in PR #9 |
| E003 | 2026-10-21 | this PR | Player-movement cue (players walk at stoppages) | pending: real-data run |

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

- **Date / author / commit:** 2026-10-21, Hack Mavericks, this PR. Design and criterion were fixed
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
- **Full-match plausibility** (human-run, aggregates only): pending.
- **Decision:** pending the real run; the pre-set criterion above decides.
- **Follow-ups:** make the generator's players walk at stoppages (a `make_synthetic.py` change,
  kept out of this experiment so E003 stays a single change).
