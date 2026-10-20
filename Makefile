# Hack Mavericks -- one entry point for every routine task.
#
#   make venv        create .venv and install pinned dependencies (dev included)
#   make test        run the test suite (synthetic data only, never the match data)
#   make synthetic   write a small fake match to .synthetic/small-seed0 (SIZE=full SEED=n)
#   make score       score evaluation/results/ against the public 5 minutes
#
# These read the match data, so a human runs them (CLAUDE.md); their output is safe to share:
#   make verify-baseline   our reader rebuilds the 4 published baselines: expect IDENTICAL x4
#   make submission        data/ -> submissions/hack_mavericks.csv, validated, with sha256
#   make reproduce         build twice from scratch and require byte-identical files
# Point any of them at another bundle with DATA_DIR=..., e.g. DATA_DIR=.synthetic/small-seed0

PYTHON_VERSION := $(shell cat .python-version)
VENV := .venv
PY := $(VENV)/bin/python
SIZE ?= small
SEED ?= 0
DATA_DIR ?=
DATA_FLAG := $(if $(DATA_DIR),--data-dir $(DATA_DIR),)
SUBMISSION := submissions/hack_mavericks.csv

.PHONY: venv test synthetic score verify-baseline submission reproduce

verify-baseline:
	$(PY) tools/verify_baseline.py $(DATA_FLAG)

submission:
	$(PY) run_pipeline.py $(DATA_FLAG) --out $(SUBMISSION)
	$(PY) evaluation/validate_submission.py $(SUBMISSION)

reproduce:
	$(PY) run_pipeline.py $(DATA_FLAG) --out submissions/reproduce_a.csv > /dev/null
	$(PY) run_pipeline.py $(DATA_FLAG) --out submissions/reproduce_b.csv > /dev/null
	cmp submissions/reproduce_a.csv submissions/reproduce_b.csv && echo "byte-identical: OK"

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
