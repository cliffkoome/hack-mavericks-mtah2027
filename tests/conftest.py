"""Shared test setup: import paths and one synthetic bundle per session.

Tests never touch data/ or MTAH_DATA_DIR. Everything they read is generated here by
tools/make_synthetic.py into a pytest temporary directory.
"""

import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for sub in ("", "starter", "evaluation", "tools"):
    path = os.path.join(ROOT, sub)
    if path not in sys.path:
        sys.path.insert(0, path)

SYNTHETIC_SEED = 7


@pytest.fixture(scope="session")
def synthetic(tmp_path_factory):
    """(out_dir, spec) of a small synthetic match, generated once per test session."""
    import make_synthetic

    out = str(tmp_path_factory.mktemp("synthetic"))
    return out, make_synthetic.generate(out, seed=SYNTHETIC_SEED, size="small")
