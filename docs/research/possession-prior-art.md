# Prior art: possession, dead-ball and missing-ball methods

Date: 2026-10-29. Scope: published methods, papers and open-source code that could improve our
per-frame H/A/D labels from tracking data, especially where the ball is missing (~41% of graded
frames). Every claim was checked against its primary source (paper, DOI, arXiv or repo).
**[unverified]** marks claims we could not confirm. No match data was used or shared in this research.

**Bottom line:** two published works are close to our set-up. Lang et al. 2022 predicts in-play vs
stopped from player data only, and Borghesi et al. 2023 predicts dead/home/away per frame. Two open
models infer the possessor or ball path from players alone: BallRadar was trained on Metrica data,
and PathCRF ships pretrained weights. Several sources publish thresholds we can adopt without tuning.

## Fit with our experiments

Read this first. It maps the findings to EXPERIMENTS.md.

- **Lang et al. 2022 (3a)** targets our largest known weakness: in gaps, slow play is called dead
  (harness `to_dead` 0.141; 88% of those frames have the slow-players cue firing). Its key
  ingredient is long context (lags up to ±80 s) plus a 3.64 s median filter. Our E010a/c/d
  dead/alive attempts did not use that context, so this would be a new experiment (candidate E013),
  not a repeat. It changes labels, so it must land before the 31 Oct hard stop, which is tight.
- **BallRadar (2a)** is an alternative or second opinion for the E005c gap team classifier, for
  side changes inside gaps. Before using it, check two things: the repo has no licence file, and we
  must confirm RULES.md allows training on the public Metrica sample games.
- **PathCRF (2b)** masked transitions are close to what our Viterbi decoder already does with
  switching penalties. The new idea it adds is that a team change happens only via a kick or
  out-of-play.
- **Vidal-Codina set-piece layouts (1b)** overlap with E007a, which we stopped because D already
  covers ~86% of set-piece starts. Low expected gain.
- **Video detectors (5a–5d)** belong to the E011 track, behind Diing's detector-accuracy gate.
  WASB's multi-frame heatmaps are the most relevant answer to E011's false-detection problem.
- **Confidentiality:** all of these run locally. Nothing from `data/` goes to any hosted service.

## 1. Possession and ball ownership from tracking data

### 1a. Link & Hoernig, "Individual ball possession in soccer" (PLoS ONE 2017)
- Source: [DOI 10.1371/journal.pone.0179953](https://pmc.ncbi.nlm.nih.gov/articles/PMC5503225/);
  data on [figshare](https://figshare.com/articles/dataset/Individual_ball_possession_in_soccer/5190445).
  No code.
- What it does: detects individual ball action, possession and control, then derives team
  possession. Uses grid-searched thresholds plus a Bayesian network that grades control.
  F-score 0.88 for action and 0.83 for possession. Data: 60 Bundesliga matches at 25 Hz, x/y plus
  the provider's ball-in-play flag.
- Published parameters:
  - segments shorter than 1 s are dropped;
  - a kick needs a local peak in ball acceleration of at least 4 m/s²;
  - endpoint prediction looks 1 s ahead;
  - the control classifier counts opponents within 0.5–5 m.

  The final distance thresholds are not reported.
- Definition worth reusing: team possession lasts until the first opponent possession, and a ball
  in flight stays with the passer's team. Our ground truth probably follows the same logic.
- Fit: definitions only, because it needs the ball. Low effort.

### 1b. Vidal-Codina, Evans, El Fakir & Billingham, "Automatic event detection in football using tracking data" (Sports Engineering 2022)
- Source: [arXiv 2202.00804](https://arxiv.org/abs/2202.00804). No official code; databallpy
  re-implements its possession step (1c).
- What it does: a rule-based pipeline.
  - Possession: the ball is inside a player's possession zone.
  - Duel: two or more opponents within 1 m of the ball. Duels override possession.
  - A possession gain needs a change in ball direction or speed.
  - Set pieces come from player layouts during dead time: all players in their own half plus one at
    the centre spot is a kick-off; a player beyond the sideline is a throw-in; a player at a corner
    flag is a corner; anything else defaults to a free kick.
  - More than 90% detection in most event classes.
- Published parameters:
  - possession zone 0.5 m for one provider, 1 m for the others; duel zone 1 m;
  - ball smoothed with Savitzky-Golay, order 2, window 7 frames.

  Its speed, direction and displacement thresholds are in Online Resource 1 **[not seen]**. It
  treats missing ball data as dead ball, the same assumption that hurts us.
- Fit: the layout rules could confirm a real D versus slow play and suggest which team restarts.
  About 1 day. See the E007a overlap above.

### 1c. databallpy
- Source: [github.com/Alek050/databallpy](https://github.com/Alek050/databallpy),
  [docs](https://databallpy.readthedocs.io/en/latest/features/team_and_player_possession.html).
  Licence MIT **[unverified]**.
- What it does: loads Metrica data. `get_individual_player_possession` implements Vidal-Codina's
  possession step. Team possession needs event data.
- Default thresholds:
  - `pz_radius` 1.5 m;
  - `bv_threshold` 5 m/s;
  - `ba_threshold` 10°;
  - `min_frames_pz` 0.
- Fit: a cheap cross-check of our nearest-player cue on frames where the ball is tracked. A few
  hours.

### 1d. kloppy, floodlight, LaurieOnTracking, socceraction
- [kloppy](https://github.com/PySport/kloppy) loads Metrica data. Its frames carry `ball_state` and
  `ball_owning_team`, but we found no inference helpers **[unverified]**.
- [floodlight](https://www.theoj.org/joss-papers/joss.04588/10.21105.joss.04588.pdf) possession
  codes are filled by data parsers, not inferred.
- [LaurieOnTracking](https://github.com/Friends-of-Tracking-Data-FoTD/LaurieOnTracking) (MIT)
  covers Metrica I/O, velocities, pitch control and EPV. It does no possession inference.
- socceraction is event-based only.
- Possible use: pitch control around the likely ball location as an extra feature for the gap
  classifier.

### 1e. SkillCorner and StatsBomb definitions
SkillCorner exports a per-frame `ball_owning_player_id`
([fast-forward docs](https://fast-forward.readthedocs.io/en/latest/providers/skillcorner/)).
Neither provider publishes a rule in metres or frames **[not found]**.

## 2. Possession and ball location when the ball is missing

### 2a. BallRadar: Kim, Choi, Kim, Yoon & Ko (KDD 2023)
- Source: [arXiv 2306.08206](https://arxiv.org/abs/2306.08206),
  [github.com/pientist/ballradar](https://github.com/pientist/ballradar). **No licence file.**
- What it does: a Set Transformer plus hierarchical Bi-LSTM, in two stages: a per-frame possessor
  classifier, then a ball trajectory regressor conditioned on it.
  - Inputs: players grouped by team, with x/y, velocity, speed and acceleration.
  - Out of play is modelled as 4 pseudo-players at the touchline midpoints.
  - Trained on the 3 public Metrica matches at 10 Hz, in 10 s windows.
- Results: team possession accuracy 85.85%, player accuracy 64.7%, ball error about 3.66 m.
- Post-processing rules:
  - score each player as possession probability divided by distance to the predicted ball;
  - a score above 0.5 is a touch;
  - a score between 0.2 and 0.5 is a touch if it is a local maximum in time;
  - between touches, the ball is linearly interpolated.
- Fit: side changes in gaps. Its in-play-only training means it does not help with D.
  2–3 days.

### 2b. PathCRF: Kim, Lee, Seo, Ko, Yoon & Park (KDD 2026)
- Source: [arXiv 2602.12080](https://arxiv.org/abs/2602.12080),
  [github.com/hyunsungkim-ds/pathcrf](https://github.com/hyunsungkim-ds/pathcrf). Licence MPL-2.0;
  pretrained weights in `saved/`.
- What it does: models possession as one edge per time step in a graph of 22 players plus 4
  outside nodes.
  - A self-loop means a player controls the ball.
  - A directed edge means the ball is in flight from one player to another.
  - An outside node means out of play; it is absorbing.

  A CRF with masked transitions (forbidden moves scored −10⁴) is decoded with Viterbi, from players
  only, at 5 fps.
- Results: sender accuracy 80.4%, event F1 75.7%, home share 62.6% predicted vs 62.1% true.
  Trained on Sportec DFL open data.
- Fit: restrict team changes in our decoder to kick or out-of-play paths, which reduces
  fragmentation. 1 day for the structure alone; 3–5 days to use the model.

### 2c. TranSPORTmer: Capellera et al. (ACCV 2024)
- Source: [arXiv 2410.17785](https://arxiv.org/abs/2410.17785). No code found.
- What it does: a transformer that classifies each frame as pass, possession, uncontrolled or
  out-of-play, and infers the ball. With the ball fully hidden, ball error is 2.71 m (vs 3.89 m for
  BallRadar) and state accuracy is 80.8%. Data: LaLiga tracking.
- Fit: design reference only.

### 2d. MIDAS: Choi et al. (ECML-PKDD 2025)
[arXiv 2408.10878](https://arxiv.org/abs/2408.10878). Multi-agent trajectory imputation. No code
found, and whether it imputes the ball is unconfirmed. Low priority.

## 3. Dead ball and ball-in-play detection

### 3a. Lang, Wild, Isenko & Link, "Predicting the in-game status in soccer with machine learning using player tracking data" (Scientific Reports 2022)
- Source: [DOI 10.1038/s41598-022-19948-1](https://pmc.ncbi.nlm.nih.gov/articles/PMC9522646/).
  No code.
- What it does: classifies in-play vs stopped from player data only (x, y, speed, acceleration of
  all 22 players).
  - Each frame is stacked with copies of the same features at ±0.4, 0.8, 2, 4, 20, 40 and 80 s.
  - AdaBoost reaches 92% frame accuracy on 47 test matches.
  - Home and away are swapped as data augmentation.
  - Frame predictions are smoothed with a 3.64 s median filter (91 frames).
- Results on stoppages of at least 2 s: precision 81%, recall 79% at IoU ≥ 0.5.
- Fit: see "Fit with our experiments". About 1 day.

### 3b. Borghesi, Costa, Morra & Lamberti, "Using temporal convolutional networks to estimate ball possession in soccer games" (Expert Systems with Applications 2023)
- Source: [DOI 10.1016/j.eswa.2023.119780](https://iris.polito.it/handle/11583/2976470).
- What it does: per-frame dead/home/away, our label set minus X. A per-object TCN feeds a
  permutation-invariant pooling layer.
- Results: 86.2% possession accuracy (+7.3% over rules) and 89.2% dead/alive (+33.2%).
- Only the abstract was read. Whether the ball is an input, and whether code exists, are both
  **[unverified]**.

### 3c. Reusable structure
- Encode out of play as 4 boundary nodes (BallRadar, PathCRF).
- Vidal-Codina's set-piece layouts.
- Smart Tagging set pieces as weak labels for training a 3a-style classifier.

## 4. Event detection and synchronisation

- [ETSY](https://github.com/ML-KULeuven/ETSY) (KU Leuven), rule-based event-to-tracking alignment.
- [sync.soccer](https://github.com/huffyhenry/sync.soccer), Needleman-Wunsch alignment.
- ELASTIC, [arXiv 2508.09238](https://arxiv.org/abs/2508.09238): 88.4% exact alignment vs 21.1%
  for ETSY. Code link unconfirmed.

All three need a human event feed. We only have Smart Tagging, so their value is limited. The one
transferable idea is aligning Smart Tagging times to tracking with Needleman-Wunsch.
[Rana, arXiv 2301.10052](https://arxiv.org/abs/2301.10052), a master's thesis on GCN event
detection, is low priority.

## 5. Ball detection in video

- **WASB-SBDT** (BMVC 2023): [arXiv 2311.05237](https://arxiv.org/abs/2311.05237),
  [github.com/nttcom/WASB-SBDT](https://github.com/nttcom/WASB-SBDT), MIT.
  - A 3-frame heatmap detector; soccer F1 88.3 on an ISSIA-style fixed-camera dataset.
  - Its tracking step did not help on soccer.
  - Soccer weights are listed in MODEL_ZOO **[unverified]**.
  - Best suited to the tactical video.
- **FootAndBall** (VISAPP 2020): [arXiv 1912.05445](https://arxiv.org/abs/1912.05445),
  [github.com/jac99/footandball](https://github.com/jac99/footandball), MIT, pretrained weights
  for long-shot cameras.
- **TrackNetV3/V4**: [repo](https://github.com/qaz812345/TrackNetV3),
  [arXiv 2409.14543](https://arxiv.org/abs/2409.14543). Built for shuttlecock and tennis; would
  need fine-tuning.
- **roboflow/sports** ([MIT](https://github.com/roboflow/sports)): no false-positive filtering for
  the ball.
- **SoccerNet Team Ball Action Spotting 2025**: [arXiv 2508.19182](https://arxiv.org/abs/2508.19182).
  - Detects 12 ball actions in broadcast video and tags each with the left or right team; top
    Team-mAP@1 is 60.0.
  - Code: baseline in [sn-spotting](https://github.com/soccernet/sn-spotting); winner
    [int8/dude.k](https://github.com/int8/dude.k) (licence unverified).
  - Team-tagged passes are direct evidence of side changes in gaps. GPU-heavy.

Suggested false-detection filter (the research agent's own idea, not from a paper): project
candidates to the pitch with the homography. Keep only candidates within 2–3 m of a predicted or
BallRadar ball path that are consistent over at least 3 frames, then feed them to the decoder as a
soft observation.

## Thresholds that need no tuning

| Parameter | Value | Source |
|---|---|---|
| Possession zone | 0.5–1 m (paper); 1.5 m (databallpy) | Vidal-Codina 2022; databallpy |
| Duel zone (overrides possession) | 1 m, at least 2 opponents | Vidal-Codina 2022 |
| Ball speed / direction change for a gain | 5 m/s / 10° | databallpy |
| Ball smoothing | Savitzky-Golay, order 2, 7 frames | Vidal-Codina 2022 |
| Kick | ball acceleration peak ≥ 4 m/s² | Link & Hoernig 2017 |
| Drop segments shorter than | 1 s | Link & Hoernig 2017 |
| Dead/alive median filter | 3.64 s (91 frames at 25 fps) | Lang et al. 2022 |
| Dead/alive context lags | ±0.4, 0.8, 2, 4, 20, 40, 80 s | Lang et al. 2022 |
| Touch score (prob / dist) | > 0.5, or 0.2–0.5 if a local maximum | BallRadar |

## Ranked for this project

1. Lang et al. 2022 lagged features plus a median filter: targets slow play called dead. About 1 day.
2. BallRadar as the gap team cue: side changes in gaps. 2–3 days; check the licence first.
3. PathCRF transition structure: less fragmentation. 1 day for the structure alone.
4. Vidal-Codina set-piece layouts: confirm D and pick the restarting team. About 1 day; low gain
   given E007a.
5. Video (SoccerNet Team Ball Action Spotting, or WASB/FootAndBall with a homography gate): touches
   in gaps. 3–5 days; highest risk.
