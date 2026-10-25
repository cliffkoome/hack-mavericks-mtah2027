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
pip install -r requirements.txt                       # numpy, pandas, scikit-learn
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
- Python 3.9+, numpy/pandas first (scikit-learn only for E005b's classifier). Keep the scorer-compatible pieces plain. Pin versions in `requirements.txt`.
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

Last updated: 2026-10-25. Deadline 17 November 2026: 23 days left.
**Submission = E005c. Technical document: `hack_mavericks.md`.** The freeze was lifted on
2026-10-25 for E011/E012 only, with a hard stop on 31 Oct; final freeze ~7 Nov. `autonomous-mode`
is merged (PR #33).

- **Submitted CSV:** E005c. `submissions/hack_mavericks.csv`, sha256 `a8ee1b28…`: 308 possessions,
  mean 13.3 s, D 35.4% of graded frames, home share 0.594. Public window (correctness only):
  16 possessions (truth 17), spellF1 0.4242, macroF1 0.6980. Floor to beat: 0.2433.
- **main's config = the submitted model (E005c)**, every later cue off. Confirmed on real data
  2026-10-25: `make submission` on `feat/e012-tag-cue` (cue off) gives `a8ee1b28…`.
- **The model (E005c):** 3-state Viterbi (`possession/decoder.py`). Evidence: nearest-player team
  cue and ball tracked/outside/still cues (E001 values), player-movement cue (E003), and a
  gap-only team cue from the team-from-movement classifier (`possession/team_model.py`,
  HistGradientBoosting, half-to-half, two-pass pipeline in `possession/pipeline.py`).
  Dependencies pinned in `requirements.txt` (numpy, pandas, scikit-learn 1.5.2, scipy, joblib,
  threadpoolctl). `make reproduce` byte-identical.
- **Experiments** (EXPERIMENTS.md has every number, the rules and the lessons): E001 keep; E002
  drop; E003 keep; E004 drop before code; E005a masked-ball harness (tool); E005b classifier gate
  PASS (0.789); E005c keep (possessions 275 → 308, gap change recall 0.115 → 0.284); E006 stop
  (velocity touches not detectable on this ball track); E009 stop (last-control: gate PASS, but no
  gain over E005c at any trust; code kept, off). 2026-10-22 (autonomous): E010a dead/alive
  classifier gate FAIL; E010c drop (classifier replaces slow cue in gaps: to_dead 0.141 → 0.045
  but D share 26.8%, change recall 0.209); E010d drop, narrowly (confident-alive beside slow
  cue: to_dead 0.100, D share 34.4%, change recall 0.249 vs limit 0.264); E007a stop (D already
  near 86% of set-piece starts); G2 synthetic generator v2. All new cues off (`gap_dead_alive`,
  `gap_alive_alive` = None). 2026-10-25: E011 (Diing's YOLO ball recovery, PR #34) closed
  unmerged: it injected 0-1 homography coordinates into the metre ball array, so every recovered
  ball sat in one corner. Synthetic: a perfect detector in metres would give +0.08 spellF1, but
  with 25% false detections the gain is ~0.01. Its next step is a detector-accuracy gate on
  seen-ball frames (Diing). E012 (Smart Tagging phase tags as a gap team cue, PR #35) stopped at
  its gate: agree 0.80 / 0.78, but right on only 0.29 / 0.32 of the frames where E005c is wrong
  (line 0.50). Code kept, off (`tag_acc_* = None`).
- **Tools the human runs** (print aggregates only, safe to paste back): `tools/inspect_cues.py`,
  `tools/masked_harness.py --gt …`, `tools/team_model_check.py --gt …`,
  `tools/inspect_touches.py --gt … [--cue touch]`, `tools/dead_model_check.py --gt …`,
  `tools/gap_dead_why.py`, `tools/inspect_restarts.py --gt …`, `tools/tag_gate.py`,
  `make verify-baseline`,
  `make submission`, `make reproduce`.
- **Known weaknesses:**
  - In gaps the model calls the ball dead on 14% of hidden in-play frames (harness `to_dead`
    0.141). Diagnosed 2026-10-22: 88% of those frames have the slow-players cue firing (slow play
    with the ball unseen). Every fix tried so far (E010c/d) costs side-change recall in gaps.
  - Synthetic v2 (G2) fixes walking at stoppages and gap lengths; the shape signal is still weak,
    and hmm macroF1 on v2 (0.54) is below the vote's (0.63), not yet diagnosed.
- **Lessons that bind new cues:** fix values, gates and criteria before the real run; gate a cue
  against the *decoded* labels on the same frames, not the raw cue it competes with; check that a
  carried cue's errors are not much longer than the ~5-frame correlation the weight assumes;
  gate a second cue on "right where the model is wrong" (E012), not only on raw agreement;
  one change per experiment; draft PRs are not for merging.

### Next session: start here

**Human, to submit:**
1. Merge PR #35 (E012, cue off: labels unchanged). Then on main: `make reproduce` (expect
   `byte-identical: OK`) and `make submission` (expect sha256 `a8ee1b28…`).
2. Make the repo public; `git status` must show no data files.
3. Google Form: upload `submissions/hack_mavericks.csv` (as `hack_mavericks.csv`) and
   `hack_mavericks.md`, plus the public repo URL. Validate the exact file you upload.
4. Repo hygiene when convenient: delete merged branches, including
   `revert-29-feat/e009b-control-cue`.

**Claude, at the start of the next session:** read this status and the end of EXPERIMENTS.md;
check that `config.py` has `control_acc = None` and `gap_acc_strong = 0.86`; then propose the
chosen step's design (values, gate, criterion) for the human's OK before any code.

**Next step candidates** (hard stop 31 Oct for anything that changes labels):
0. **E011 gate (Diing):** YOLO on frames where the tracking *has* the ball: detection share,
   median error in metres, share within 2 m, at confidences 0.05 / 0.25 / 0.5. Pass bar to fix
   before the run. Only then: the units fix, a cache in `cache/`, and a cue that is off by
   default. Broadcast-ATD sync feasibility is parked behind it.
1. **E010d revisit:** closest miss; only with more masking seeds and a criterion re-stated
   before the run, never by moving this one's limit. (E007b restarts: not built, E007a.)
2. If any model change lands, update `hack_mavericks.md` (sha, results, tables) with it.
3. **Diagnose v2 synthetic macroF1** (hmm 0.54 vs vote 0.63), then the shape signal in G2.
4. **E008 Metrica sample-data harness** (development aid only, never validation).

**Deliverables, with dates:**
- by ~7 Nov: model freeze; after it, no model changes.
- 7–12 Nov: write `hack_mavericks.md` (approach, assets used and dropped, possession rules,
  limitations, reproduction) from EXPERIMENTS.md; make the repo public; final `make reproduce` +
  `validate_submission.py`; submit via the Google Form (CSV + public repo URL + document).
- 13–17 Nov: buffer for submission problems only.
