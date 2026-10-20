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
- **Bootstrap:** pending (`run_evaluation.py --bootstrap 1000`).
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
