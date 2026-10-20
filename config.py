"""Every hyperparameter of the submitted pipeline, in one place.

Few on purpose (CLAUDE.md): each one is a way to overfit. Never choose a value by scoring the public
5 minutes -- record the reason for each in EXPERIMENTS.md instead.
"""

PARAMS = {
    # Majority-vote half-width in frames. 50 is the published baseline floor (EVALUATION.md);
    # it stays until a segment-level decoder replaces the vote.
    "smooth_window": 50,
}
