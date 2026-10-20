# Hack Mavericks -- one entry point for every routine task.
#
#   make venv        create .venv and install pinned dependencies (dev included)
#   make test        run the test suite (synthetic data only, never the match data)
#   make synthetic   write a small fake match to .synthetic/small-seed0 (SIZE=full SEED=n)
#   make score       score evaluation/results/ against the public 5 minutes
#
# Targets that touch the match data (the pipeline itself) are run by a human, never by an
# assistant -- see CLAUDE.md. More targets (submission, reproduce) arrive with the code they run.

PYTHON_VERSION := $(shell cat .python-version)
VENV := .venv
PY := $(VENV)/bin/python
SIZE ?= small
SEED ?= 0

.PHONY: venv test synthetic score

test:
	$(PY) -m pytest tests -q

synthetic:
	$(PY) tools/make_synthetic.py --size $(SIZE) --seed $(SEED)

venv:
	pyenv exec python -c "import sys; assert sys.version.startswith('$(PYTHON_VERSION)'), sys.version"
	pyenv exec python -m venv $(VENV)
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements-dev.txt

score:
	$(PY) evaluation/run_evaluation.py
