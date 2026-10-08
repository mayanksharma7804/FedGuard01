"""Compare finished runs on the VALIDATION split (use this for every design decision; never the test set).

    python scripts/compare_runs.py b0_centralised_sel_cw_none_seed42 b0_centralised_sel_cw_sqrt_seed42 ...
    python scripts/compare_runs.py --glob "b0_centralised_sel_*"
"""
import argparse
import json

import pandas as pd

from fedguard.utils import REPO_ROOT

RUNS = REPO_ROOT / "results" / "runs"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="*")
    ap.add_argument("--glob")
    ap.add_argument("--split", default="val", choices=["val", "test"])
    args = ap.parse_args()
    dirs = sorted(RUNS.glob(args.glob)) if args.glob else [RUNS / r for r in args.runs]
    rows, per = {}, {}
    for d in dirs:
        f = d / f"metrics_{args.split}.json"
        if not f.exists():
            print(f"[missing] {d.name}"); continue
        m = json.load(open(f))
        rows[d.name] = {k: m.get(k) for k in ("macro_f1", "weighted_f1", "accuracy", "benign_fpr",
                                              "attack_detection_rate")}
        rows[d.name]["best"] = m.get("best_epoch", m.get("best_round"))          # B0: epochs, FL: rounds
        rows[d.name]["run_length"] = m.get("epochs_run", m.get("rounds_run"))
        per[d.name] = {c: v["recall"] for c, v in m["per_class"].items()}
    if not rows:
        return
    print(f"== {args.split} split ==")
    print(pd.DataFrame(rows).T.to_string(float_format=lambda v: f"{v:.4f}"))
    print(f"\nper-class recall ({args.split}):")
    print(pd.DataFrame(per).to_string(float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
