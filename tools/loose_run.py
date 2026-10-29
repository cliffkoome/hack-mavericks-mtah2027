"""E016 in one run: the loose-ball decoder state, alone (E016a) and with E015's cue (E016b).

    python tools/loose_run.py --gt evaluation/gt/BAR-ATH_possession_gt_public.csv
    python tools/loose_run.py --keep-switch --gt …    # E017: the direct switch kept beside L
    python tools/loose_run.py --data-dir .synthetic/eval/full-seed0 \
        --gt .synthetic/eval/full-seed0/synthetic_possession_gt.csv

Each variant is scored by its own criterion, fixed in EXPERIMENTS.md (E016) before any run:
masked harness (5 seeds) against the unmasked run of the same config, full-match shares, and the
public-truth D floor. E016b uses E015's measured cue values as they are, not re-measured.

Everything printed is a rate, a share or a count over many frames. Safe to paste back.
"""

import argparse
import copy
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
from config import PARAMS  # noqa: E402
from possession import atd, spec as spec_module  # noqa: E402
import gap_alive_run as run  # noqa: E402
import possession_metrics as pm  # noqa: E402

# Params
LOOSE_S = 1.0                     # mean seconds in L: a duel or a ball in flight
E015_VALUES = (0.98, 0.31)        # gap_alive_alive, gap_alive_dead as E015 measured them
E016A = dict(to_dead_max=0.151, recall_min=0.314, d_share=(0.334, 0.374))
E016B = dict(to_dead_max=0.111, recall_min=0.264, d_share=(0.31, 0.45))


def variants(params, keep_switch=False):
    """{name: (params, criterion bounds)} for E016a and E016b (E017a/b with keep_switch)."""
    loose = copy.deepcopy(params)
    loose["decoder"].update(loose_s=LOOSE_S, loose_keep_switch=keep_switch)
    with_cue = copy.deepcopy(loose)
    with_cue["evidence"].update(gap_alive_alive=E015_VALUES[0], gap_alive_dead=E015_VALUES[1],
                                gap_dead_recipe="lang", gap_dead_teacher="pass1")
    tag = "E017" if keep_switch else "E016"
    return {tag + "a: L alone": (loose, E016A), tag + "b: L + E015 cue": (with_cue, E016B)}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir")
    parser.add_argument("--gt", required=True, help="public (or synthetic) ground truth")
    parser.add_argument("--keep-switch", action="store_true",
                        help="E017: keep the direct switch beside L")
    args = parser.parse_args()

    spec = spec_module.load(args.data_dir)
    tracking = atd.load(spec, args.data_dir)
    gt = pm.label_codes(pm.read_label_csv(args.gt))
    print("loose-ball decoder run, loose_s {}, keep switch {}".format(LOOSE_S, args.keep_switch))
    print("E005c (logged): to_dead 0.141, change_recall 0.284, D 35.4%, 308 poss, 13.3 s")
    for name, (params, bounds) in variants(PARAMS, args.keep_switch).items():
        print(name)
        s = run.score(tracking, spec, params, gt)
        run.print_score(s)
        run.report(run.criterion(s, **bounds))
    return 0


if __name__ == "__main__":
    sys.exit(main())
