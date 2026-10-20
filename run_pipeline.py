"""Regenerate the Hack Mavericks submission from the data bundle. The one command.

    python run_pipeline.py                                   # data/ -> submissions/hack_mavericks.csv
    python run_pipeline.py --data-dir .synthetic/small-seed0 --out /tmp/synthetic.csv

Reads the match data, so on the real bundle a human runs it (CLAUDE.md). What it prints is safe to
share: label counts, possession count and duration, validation result and the file's sha256.
"""

import argparse
import json
import os
import sys

from config import PARAMS
from possession import pipeline, submission

ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT = os.path.join(ROOT, "submissions", "hack_mavericks.csv")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", help="bundle root (default: $MTAH_DATA_DIR or ./data)")
    parser.add_argument("--out", default=DEFAULT_OUT)
    args = parser.parse_args()

    codes, spec = pipeline.build(PARAMS, args.data_dir)
    digest = submission.write(args.out, codes)
    reason = submission.validate(args.out, spec)
    print(json.dumps({"out": os.path.relpath(args.out, ROOT), "params": PARAMS,
                      "valid": reason is None, "reason": reason, "sha256": digest,
                      **submission.summary(codes, spec)}, indent=2))
    return 0 if reason is None else 1


if __name__ == "__main__":
    sys.exit(main())
