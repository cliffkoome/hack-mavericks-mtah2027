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
        # E016: mean seconds in the loose-ball state L (alive, neither side), through which every
        # turnover then goes; a duel or a ball in flight lasts about 1 s. None = no L (E005c).
        "loose_s": None,
        # E017: keep the direct H <-> A switch beside L (L is a second route), so a turnover
        # costs what it did without L. False = E016 (every turnover through L).
        "loose_keep_switch": False,
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
        # E003: players walk at a stoppage and stay tracked when the ball does not. Cue = the
        # players' median speed, mean over +/- 1 s, below slow_speed. Values measured with
        # tools/inspect_cues.py (EXPERIMENTS.md, E003); 1.25 has the best dead/alive ratio (6.0)
        # with useful recall.
        "slow_speed": 1.25,         # m/s; None = no cue (E001)
        "slow_alive": 0.10,         # P(players slow | alive); 0.104 measured
        "slow_dead": 0.62,          # P(players slow | dead); 0.624 measured
        # E005c: on frames with the ball untracked, the team-from-movement classifier
        # (team_model.py, E005b) as a second team cue. Accuracies measured on E005a's hidden
        # frames, used as measured (EXPERIMENTS.md, E005c: the value rule).
        "gap_conf": 0.7,            # p >= this or <= 1 - this is a confident call
        "gap_acc_strong": 0.86,     # P(classifier right | confident); 0.861; None = off (E003)
        "gap_acc_weak": 0.56,       # same, not confident; 0.564 measured
        # E009 (stopped, off): on seen-ball frames, the side last in control (touches.held/
        # controls) would replace the nearest-player cue. Its measured accuracy is 0.865 (0.87),
        # but at no trust setting did it beat E005c on synthetic data, so it stays off
        # (EXPERIMENTS.md, E009). None = off = E005c exactly.
        "control_acc": None,        # P(last-control side is right); 0.865 measured; None = off
        # E010c (off): on untracked frames, the dead/alive-from-movement classifier
        # (dead_model.py) replaces the slow-players cue. Rates measured in E010a row (3).
        "gap_dead_alive": None,     # P(classifier says dead | alive); 0.021 measured; None = off
        "gap_dead_dead": 0.64,      # P(classifier says dead | dead); 0.643 measured
        # E013: which recipe the dead/alive classifier uses (dead_model.RECIPES). "e010" = E010's;
        # "lang" = Lang et al.'s full recipe. Read only when a gap dead cue above is on.
        "gap_dead_recipe": "e010",
        # E014: the labels the dead/alive classifier learns from. "pass1" = the pass-1 labels
        # (E010/E013); "no_slow" = pass 1 decoded without the slow cue, so the classifier cannot
        # copy that cue's false alarms. Read only when a gap dead cue above is on.
        "gap_dead_teacher": "pass1",
        # E010d (off): on untracked frames, beside the slow cue, "confidently alive" =
        # p_dead <= 0.3 as a yes/no cue. Rates measured against the pass-1 reference.
        "gap_alive_alive": None,    # P(confidently alive | alive); 0.968 measured; None = off
        "gap_alive_dead": 0.30,     # P(confidently alive | dead); 0.300 measured
        # E012 (off): on untracked frames, beside the gap team cue, the side a Smart Tagging phase
        # says has the ball (tagging.phase_cue). Values fixed by reasoning before any run
        # (EXPERIMENTS.md, E012); the two accuracies come from its gate, used as measured.
        "tag_acc_attack": None,     # P(attacking tag's Team has the ball); None = group off
        "tag_acc_defend": None,     # P(defending tag's other side has it); both None = E005c
        "tag_trim_s": 1.0,          # seconds off each end of a tag: its times are automatic
        "tag_cap": 25,              # frames of evidence per run of one side (as missing_run_cap)
        "tag_unsure_only": False,   # True = only where the gap classifier is not confident
        "weight": 0.2,              # evidence tempering: ~1 independent reading per 5 frames
    },
}
