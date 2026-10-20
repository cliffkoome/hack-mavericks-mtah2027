"""Every hyperparameter of the submitted pipeline, in one place.

Few on purpose (CLAUDE.md): each one is a way to overfit. Never choose a value by scoring the public
5 minutes -- record the reason for each in EXPERIMENTS.md instead.
"""

PARAMS = {
    # Which model labels the frames: "vote" (the reference baseline) or "hmm" (decoder.py).
    # "hmm" since E001 met its pre-set real-data criterion (EXPERIMENTS.md).
    "model": "hmm",

    # vote: majority-vote half-width in frames. 50 is the published baseline floor (EVALUATION.md).
    "smooth_window": 50,

    # hmm: transition rates (decoder.py). Football priors, not fitted.
    "decoder": {
        "mean_possession_s": 12.0,  # typical possession length in elite football
        "stoppage_share": 0.3,      # share of possessions ending in a stoppage, not a turnover
        "mean_dead_s": 15.0,        # typical stoppage length
    },

    # hmm: per-frame evidence (evidence.py). Each value is a probability with a stated meaning.
    # E002 recalibrated these against tools/inspect_cues.py on the public window (frame-level
    # rates over thousands of frames, not a score); see EXPERIMENTS.md for each change.
    "evidence": {
        "near_m": 2.0,              # nearest player this close: probably on the ball
        "mid_m": 5.0,               # this close: weaker cue
        "team_acc_near": 0.85,      # P(nearest side is the possessing side | near, alive); 0.873 measured
        "team_acc_mid": 0.65,       # same, mid band; 0.647 measured
        "team_acc_far": 0.6,        # same, beyond mid_m; 0.675 measured, shrunk towards 0.5 (E002)
        # Untracked ball. Anchored on the documented full-match rate (41% of graded frames), not
        # on the public window, which is unusually well tracked (6.5% alive / 33.6% dead there):
        # 0.62 * 0.26 + 0.38 * 0.65 = 0.41 with a ~38% stoppage share and a 2.5x dead/alive ratio.
        "missing_alive": 0.26,
        "missing_dead": 0.65,
        "missing_run_cap": 25,      # a run of untracked frames counts as at most 1 s of evidence
        "outside_margin_m": 0.5,    # metres beyond a line before a seen ball counts as outside
        "outside_alive": 0.002,     # P(seen ball outside | alive); 0 of 5640 measured
        "outside_dead": 0.1,        # P(seen ball outside | dead); 0.101 measured
        "speed_lag": 5,             # frames over which ball speed is measured
        "still_speed": 0.3,         # m/s below which a seen ball counts as still
        # Neutralised (equal under both states): measured 0.009 alive and 0.009 dead, because
        # tracking jitter makes a still ball look moving over 5 frames. Kept for a better
        # definition; see the 1 s diagnostic in tools/inspect_cues.py.
        "still_alive": 0.05,
        "still_dead": 0.05,
        "weight": 0.2,              # evidence tempering: ~1 independent reading per 5 frames
    },
}
