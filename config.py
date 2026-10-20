"""Every hyperparameter of the submitted pipeline, in one place.

Few on purpose (CLAUDE.md): each one is a way to overfit. Never choose a value by scoring the public
5 minutes -- record the reason for each in EXPERIMENTS.md instead.
"""

PARAMS = {
    # Which model labels the frames: "vote" (the reference baseline) or "hmm" (decoder.py).
    # Stays "vote" until the decoder has been checked on the real data (EXPERIMENTS.md E001).
    "model": "vote",

    # vote: majority-vote half-width in frames. 50 is the published baseline floor (EVALUATION.md).
    "smooth_window": 50,

    # hmm: transition rates (decoder.py). Football priors, not fitted.
    "decoder": {
        "mean_possession_s": 12.0,  # typical possession length in elite football
        "stoppage_share": 0.3,      # share of possessions ending in a stoppage, not a turnover
        "mean_dead_s": 15.0,        # typical stoppage length
    },

    # hmm: per-frame evidence (evidence.py). Each value is a probability with a stated meaning.
    "evidence": {
        "near_m": 2.0,              # nearest player this close: probably on the ball
        "mid_m": 5.0,               # this close: weaker cue; beyond it, no team cue at all
        "team_acc_near": 0.85,      # P(nearest side is the possessing side | near, alive)
        "team_acc_mid": 0.65,       # same, mid band
        "missing_alive": 0.4,       # P(ball not tracked | alive); ~41% overall on graded frames
        "missing_dead": 0.75,       # P(ball not tracked | dead); assumed higher than alive
        "outside_margin_m": 0.5,    # metres beyond a line before a seen ball counts as outside
        "outside_alive": 0.01,      # P(seen ball outside | alive)
        "outside_dead": 0.3,        # P(seen ball outside | dead)
        "speed_lag": 5,             # frames over which ball speed is measured
        "still_speed": 0.3,         # m/s below which a seen ball counts as still
        "still_alive": 0.05,        # P(still | alive)
        "still_dead": 0.5,          # P(still | dead)
        "weight": 0.2,              # evidence tempering: ~1 independent reading per 5 frames
    },
}
