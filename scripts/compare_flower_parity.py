"""Engine-parity check (plan Phase 4 step 7): our fl_sim.py vs Flower must give the same FedAvg run.

Run scripts/windows/flower_parity.ps1 first (it writes results/flower_parity/flower_rounds.csv);
this script then runs the same config in our engine on the CPU and compares round by round.
"""
import json
import sys

import pandas as pd
import torch

from fedguard.config import load_config
from fedguard.data import load_processed
from fedguard.fl_sim import FLSimulation
from fedguard.partition import load_partition, partition_path
from fedguard.utils import REPO_ROOT, set_seed

OUT = REPO_ROOT / "results" / "flower_parity"


def main():
    meta = json.load(open(OUT / "flower_run.json"))
    flw = pd.read_csv(OUT / "flower_rounds.csv")
    cfg = load_config(REPO_ROOT / meta["config"], {"seed": meta["seed"], "train": {"device": "cpu"},
                                                   "fl": {"max_rounds": meta["rounds"], "min_rounds": 10**6}})
    set_seed(cfg["seed"])
    data = load_processed(cfg["dataset"])
    parts, _ = load_partition(partition_path(cfg["dataset"], cfg["fl"]["alpha"], cfg["fl"]["clients"], cfg["seed"]),
                              fingerprint=data.fingerprint)
    _, rounds, wlog, _ = FLSimulation(cfg, data, parts, torch.device("cpu"), log=lambda *a: None).run()
    eng = pd.DataFrame(rounds)[["round", "val_macro_f1", "val_accuracy"]]
    cmp_ = flw.merge(eng, on="round", suffixes=("_flower", "_engine"))
    cmp_["abs_diff_macro_f1"] = (cmp_["val_macro_f1_flower"] - cmp_["val_macro_f1_engine"]).abs()
    cmp_.to_csv(OUT / "parity_comparison.csv", index=False, float_format="%.8f")
    print(cmp_.to_string(index=False, float_format=lambda v: f"{v:.6f}"))
    worst = cmp_["abs_diff_macro_f1"].max()
    ok = len(cmp_) == meta["rounds"] and worst < 1e-6
    print(f"\nmax |difference| in validation macro-F1 over {len(cmp_)} rounds: {worst:.2e}  ->  "
          f"{'PARITY OK' if ok else 'MISMATCH'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
