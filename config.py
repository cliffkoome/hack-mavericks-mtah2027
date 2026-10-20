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
    # These are E001's values exactly: E002 recalibrated five of them at once and failed its
    # pre-set criterion (D share 45% -> 8%), so it was reverted to change one thing at a time
    # (EXPERIMENTS.md). Measured rates from tools/inspect_cues.py are noted for the next steps.
    "evidence": {
        "near_m": 2.0,              # nearest player this close: probably on the ball
        "mid_m": 5.0,               # this close: weaker cue
        "team_acc_near": 0.85,      # P(nearest side is the possessing side | near, alive); 0.873 measured
        "team_acc_mid": 0.65,       # same, mid band; 0.647 measured
        "team_acc_far": None,       # beyond mid_m: None = no cue; 0.675 measured
        "missing_alive": 0.4,       # P(ball not tracked | alive); ~41% overall on graded frames
        "missing_dead": 0.75,       # P(ball not tracked | dead); assumed higher than alive
        "missing_run_cap": None,    # frames of evidence per untracked run; None = every frame counts
        "outside_margin_m": 0.5,    # metres beyond a line before a seen ball counts as outside
        "outside_alive": 0.01,      # P(seen ball outside | alive); 0 of 5640 measured
        "outside_dead": 0.3,        # P(seen ball outside | dead); 0.101 measured
        "speed_lag": 5,             # frames over which ball speed is measured
        "still_speed": 0.3,         # m/s below which a seen ball counts as still
        "still_alive": 0.05,        # P(still | alive); 0.009 measured
        "still_dead": 0.5,          # P(still | dead); 0.009 measured -- this cue does not work
        "weight": 0.2,              # evidence tempering: ~1 independent reading per 5 frames
    },
}
