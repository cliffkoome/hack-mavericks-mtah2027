"""Hack Mavericks possession pipeline: tracking in, one H/A/D/X label per tactical video frame out.

Modules, in pipeline order:
    spec        frame layout of the match (length, graded blocks, kickoffs)
    atd         raw ATD -> arrays indexed by video frame
    features    per-frame evidence (nearest player to the ball, ...)
    smoothing   majority vote over a window (the reference baseline's step 2)
    rules       label rules every submission obeys (X off the graded blocks, pre-kickoff D)
    submission  write, validate and summarise a frame,label CSV
    pipeline    the whole chain, as run by run_pipeline.py
"""

import os
import sys

# games.py lives at the repository root and the scorer in evaluation/; neither is a package.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _path in (ROOT, os.path.join(ROOT, "evaluation")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

H, A, D, X = 0, 1, 2, 3
LABELS = "HADX"
