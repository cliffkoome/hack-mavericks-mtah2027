# CLAUDE.md — More than a Hack 2027, Phase 1: Ball Possession

## What we are doing

Competition entry for More than a Hack 2027 (Metrica Sports / FC Barcelona / MWC Barcelona).
Task: for every frame of one match (Barcelona vs Athletic Club), output who has the ball.

Team name: **Hack Mavericks**. Deliverables: `hack_mavericks.csv` and `hack_mavericks.md` (lowercase,
no spaces; the technical document is Markdown because `.gitignore` excludes `*.pdf`). Code lives in the
`possession/` package.

Repos: `origin` = github.com/cliffkoome/hack-mavericks-mtah2027 (ours; private during development,
must be made public before submitting). `upstream` = the organisers' challenge repo, fetch-only
(push URL disabled). Never push to upstream; pull organiser fixes with `git fetch upstream`.

- Output: CSV `frame,label`, exactly 199,319 rows, frames 0..199318 in order.
- Labels: `H` home in possession, `A` away in possession, `D` dead ball, `X` no answer.
- Row i of the submission = frame i of the TACTICAL video. Never index the broadcast video (~164k rows = wrong).
- Ranked on `spellF1` (possession-level, private full-match ground truth). `macroF1` is the first tie-break.
- Baseline to beat: `baseline_smooth50`, spellF1 0.2433 on the full match.
- Deadline: 17 November 2026. Submit via Google Form: CSV + public repo URL + short technical doc.
- Name the files `<team_name>.csv` and `<team_name>.pdf` (or `.md`).

## HARD RULES — read first, never break

### 1. Match data confidentiality (competition terms)

Participants must NOT upload, disclose or make available to any third-party hosted LLM any videos,
datasets or other restricted materials from the competition, or extracts of them. Claude Code sends
file contents and command output to a hosted model, so:

- NEVER read, cat, head, tail, grep, print or open anything under `data/` (or `$MTAH_DATA_DIR`):
  `*.mp4`, `*_atd.txt`, `*_atd.xml`, `*_homography.csv`, `*_smart_tagging.xml`. Not even a few lines.
- NEVER print tracking values, ball positions, frame-level rows or video frames into the conversation.
- NEVER run a command whose output would contain rows or values from those files.
- Write code that processes the data, but the HUMAN runs data-touching scripts locally.
  The human may paste back only high-level, non-extractive results (e.g. spellF1/macroF1 scores,
  possession counts, error messages with no data values), and only if the team has confirmed this
  is acceptable with the organisers.
- To develop and test, use synthetic fake data that you generate yourself in the same format
  (see `tests/` and `tools/make_synthetic.py` once they exist). Never derive test fixtures from real data.
- Files that ARE safe to read: everything tracked in this git repo (code, docs, `evaluation/gt/`
  public ground truth, `evaluation/results/` baselines), because they are public. The only tracked
  file under `data/` is `data/README.md`; read it via `git show HEAD:data/README.md`, never by listing
  or globbing `data/`.
- Notebooks in `starter/` may carry saved outputs; do not read their outputs.
- `tools/check_format.py` (once it exists) is run by the human against the real files. It may only
  print PASS/FAIL per named structural check (field counts, column names, XML tag set), never values,
  and only that PASS/FAIL list may be pasted back.
- The authority for these rules is RULES.md, FAQ.md ("Can I use LLMs"), OVERVIEW.md and data/README.md:
  no data or extracts to any hosted LLM, no redistribution, no published derived data that would
  reconstruct it. Anything those docs do not prohibit is allowed; when near a line, ask the human.
- If unsure whether something is data, treat it as data and do not open it.

### 2. Competition rules

- Labels must be produced by an algorithm. No hand-labelling or hand-correcting output frame by frame.
- Do not use the private ground truth or anything derived from it. Only the public 5-minute GT
  (video frames 11185..18684) is allowed.
- The repo must regenerate the submitted CSV exactly (same file). Prefer deterministic CPU code.
- The repo is public: never commit data, frame-level tracking exports, clipped video or derived data
  that would reconstruct them. `data/` stays gitignored.

## Repo layout (from the challenge repo)

```
games.py                    constants, asset_paths(), frame alignment helpers
evaluation/
  possession_metrics.py     metric definitions (the specification; read it)
  run_evaluation.py         scorer (identical to the official one)
  validate_submission.py    format + sanity check; run before every submit
  gt/                       public 5-minute ground truth
  results/                  put candidate submission CSVs here (scored by run_evaluation)
starter/                    loaders (ATD, homography, smart tagging), baseline, notebooks
data/                       gated match data, gitignored (human only, see Hard Rules)

-- ours --
possession/                 the pipeline package: spec, atd, features, smoothing, evidence, decoder,
                            rules, submission, pipeline
config.py                   every hyperparameter, each with its reason; model = "vote" | "hmm"
run_pipeline.py             the one command: bundle -> submissions/hack_mavericks.csv (+ sha256)
EXPERIMENTS.md              the experiment log; read its rules before quoting any number
tools/make_synthetic.py     fake bundle in the real formats + synthetic GT (.synthetic/, gitignored)
tools/synthetic_eval.py     variants x synthetic seeds: correctness/robustness only, never a selector
tools/verify_baseline.py    real-data check: our reader must rebuild baseline_*.csv IDENTICALLY
tests/                      pytest; synthetic data and the public GT only
Makefile                    venv, test, synthetic, score, verify-baseline, submission, reproduce
```

Always get paths via `from games import asset_paths` (tactical is default). Do not hard-code filenames.

## Commands

```bash
pip install -r requirements.txt                       # numpy, pandas
python evaluation/run_evaluation.py                   # score everything in evaluation/results/
python evaluation/run_evaluation.py --bootstrap 1000  # confidence intervals
python evaluation/validate_submission.py <file>.csv   # ALWAYS before submitting
python starter/baseline.py                            # rebuild baselines (needs data; human runs it)
```

## Key facts about the data (documented, not looked up from raw files)

- Frame alignment: submission/homography frame = video frame; ATD: `video_frame = frame_id - 1`, nothing
  else (starter/load_atd.py, DATA.md). The per-half offsets in `games.py` (+1702057 / +1702058) map a
  separate IdFrame counter the GT was derived from; never apply them to the ATD. Smart Tagging seconds:
  `round(s*25)` (Python round, half-to-even, same as the starter loader).
- `games.py` kickoff anchors compute to 11186 and 113741, one frame off the published 11185 / 113742.
  They are playback anchors only; use the published numbers (DATA.md, FAQ.md).
- Graded frames: 11,185-90,651 (1st half) and 110,808-190,197 (2nd half). Everything else is never scored.
- Second half is graded from 110,808 but kicks off at 113,742: those 2,934 frames are all `D`.
- ATD: 46 anonymous track slots (not people), coords normalised 0-1 (pitch 105.3 x 68.0 m, scale both
  axes), speeds in m/s, NaN = missing. Ball missing on ~41% of graded frames.
- Smart Tagging: 638 automatic phases/set pieces, with `Team` (36 are `N/A`, including all throw-ins).
  Inferred, not ground truth. Phases are not possessions. Check what `Team` means per code
  (for defending phases it may be the defending team) against the public window before relying on it.
- Ground truth is mechanical: `D` where corrected tracking marks the ball dead, else the team the
  tracking names as owner. Dead-ball ownership is not asked for.
- Possession = maximal run of one team with ball alive. `H H D D H H` is two possessions.

## Metric facts that drive design

- `spellF1`: one-to-one matching of predicted vs true possessions by temporal IoU, averaged over
  thresholds 0.3-0.7. Over-fragmentation and missed short possessions both hurt; 1 frame late is free.
- `X` is forward-filled with the last non-`X` label in the half before spellF1. Abstaining inside a
  possession is free; abstaining across a hand-over is costly. `macroF1` counts every `X` as a miss.
- Public GT shape (correctness reference only): 17 possessions, median 7.6 s, 6 under 4 s, shortest
  19 frames; 12 team-to-team transitions vs 5 via D. Short possessions are real: never enforce a
  minimum possession length of seconds.
- Differences below ~0.04 spellF1 (full match) are noise. The public window (17 possessions) cannot rank
  versions: the smoothing window that wins there loses on the full match.

## Approach plan (work in this order)

1. Reproduce baseline and confirm the pipeline is correct on the public window (alignment, format).
2. Tracking-only possession model: ball-owner features (distance, ball speed, direction changes,
   sustained control over a few seconds) feeding a segment-level decoder (HMM/Viterbi with switching
   penalties) instead of a window majority vote.
3. Dead-ball detection: set-piece restarts from Smart Tagging, ball out of bounds, stationary ball,
   player regrouping, plus the 2nd-half pre-kickoff `D` block.
4. Missing ball (41%): bridge short gaps; for long gaps, a model on player-configuration features
   trained on frames where the ball is visible and the label is confident (self-supervised).
5. Smart Tagging as weak labels/priors across the whole match, handling `N/A` team.
6. Optional, only if 1-5 plateau: video-based ball recovery (GPU). Check the terms on where compute runs.

## Coding conventions

- Environment: `make venv` builds `.venv` from pyenv 3.12.7 (`.python-version`) with the pinned
  `requirements-dev.txt`; always run Python as `.venv/bin/python` (system python has no numpy).
  Synthetic bundles and caches go in gitignored folders.
- Python 3.9+, numpy/pandas first. Keep the scorer-compatible pieces plain. Pin versions in `requirements.txt`.
- Determinism: fixed seeds, no nondeterministic GPU ops in the final label path. One command must
  regenerate the final CSV from the data bundle (`make submission` or `python run_pipeline.py`).
- Few hyperparameters. Do NOT tune against the public 5 minutes to choose between versions; use it only
  to check correctness. Use `--bootstrap 1000` before believing a gap.
- Keep modules small and testable on synthetic data. Add unit tests for alignment, label rules and the decoder.
- Log every experiment (idea, change, full-match proxy checks, scores) in `EXPERIMENTS.md`.
- Never commit anything in `data/`, large CSV exports, or video.

## Before every submission

1. Re-run the full pipeline from scratch and confirm the CSV is byte-identical.
2. `python evaluation/validate_submission.py <team_name>.csv` passes (199,319 rows, header `frame,label`).
3. Sanity-check possession count and mean duration look like football (hundreds of possessions, ~10-20 s).
4. Confirm `git status` shows no data files, and the repo is public.
5. Update the technical document: approach, assets used and dropped, possession rules, limitations, reproduction.

## Working style for Claude Code

- Plan first; propose changes before large refactors. Ask before anything that would touch `data/`.
- Prefer simple, explainable logic over opaque models; the organisers must be able to inspect it.
- When a result looks too good, suspect an off-by-one in frame alignment or leakage from the public window.
- Workflow that works for this team: one change per branch and PR (`feat/...`, `fix/...`, `docs/...`);
  the human reviews and merges. Claude never runs data-touching commands; it hands the human the exact
  commands, and the human pastes back aggregate output only (scores, counts, rates, sha256).
- One change per experiment, with its keep/revert criterion written in EXPERIMENTS.md *before* the
  real-data run. The criterion decides, not the public-window score (its noise band is ±0.33).

## Current status (update at the end of every session)

Last updated: 2026-10-20.

- **Submitted model:** `hmm` (3-state Viterbi, `possession/decoder.py`) with E001's evidence values.
  Real data, full match: 316 possessions, mean 11.0 s, D 45.3% of graded frames, sha256 `6501615e…`.
  Public window (correctness only): spellF1 0.3771, macroF1 0.6831. Floor to beat: spellF1 0.2433.
- **Verified on the real data:** `make verify-baseline` IDENTICAL ×4 (reader, alignment, slot teams).
- **Known weakness:** D is over-called (45% vs a typical 35–42%), driven by untracked-ball frames
  counted as dead-ball evidence. E002 (cap + recalibration) over-corrected to 8% and was reverted.
- **Next step: E003, the player-movement cue** — fully specified in EXPERIMENTS.md (E003): cue
  definition, the measured rates it uses, and the pre-set keep/revert criterion. Implement it as the
  only change, then hand the human the run commands.
- **Queued after E003:** untracked-ball runs > 10 s as a dead-ball cue (0% alive vs 25% dead
  measured); Smart Tagging set pieces as restart evidence (Team semantics in E002); a control-based
  team cue for transitions ~10 frames early; `tools/check_format.py` is no longer needed (the real
  bundle parses and verify-baseline passes).
- **Deliverables still to do:** `hack_mavericks.md` technical document; make the repo public before
  17 November 2026; final `make reproduce` + `validate_submission.py` per "Before every submission".
