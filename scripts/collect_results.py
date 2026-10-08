"""Collect every finished federated run into one table: results/master.csv (one row per run).

    python scripts/collect_results.py

Columns: run name, the config values that matter, then validation and test metrics. Every number in the
report must be traceable to a row of this file (plan section 14.3).
"""
import json

import pandas as pd
import yaml

from fedguard.utils import REPO_ROOT

RUNS = REPO_ROOT / "results" / "runs"
METRICS = ["macro_f1", "weighted_f1", "accuracy", "benign_fpr", "attack_detection_rate", "binary_f1"]


def row(run_dir):
    cfg = yaml.safe_load(open(run_dir / "config.yaml"))
    test = json.load(open(run_dir / "metrics_test.json"))
    val = json.load(open(run_dir / "metrics_val.json"))
    meta = json.load(open(run_dir / "meta.json")) if (run_dir / "meta.json").exists() else {}
    r = {"run": run_dir.name, "name": cfg["name"], "seed": cfg["seed"], "dataset": cfg["dataset"],
         "clients": cfg["fl"]["clients"], "alpha": cfg["fl"]["alpha"], "local_epochs": cfg["fl"]["local_epochs"],
         "aggregator": cfg["fl"]["aggregator"]["name"], "epsilon": cfg["dp"].get("epsilon"),
         "attack": cfg["attack"]["type"], "attack_fraction": cfg["attack"]["fraction"],
         "lr": cfg["train"]["lr"], "best_round": test.get("best_round"), "rounds_run": test.get("rounds_run"),
         "seconds_per_round": test.get("seconds_per_round_mean"), "git_commit": meta.get("git_commit"),
         # privacy (Phase 5): target epsilon covers fl.max_rounds; 'spent' = the rounds actually run
         "clip": cfg["dp"].get("clip") if cfg["dp"].get("epsilon") is not None else None,
         "max_rounds": cfg["fl"]["max_rounds"], "epsilon_spent_max": test.get("epsilon_spent_max"),
         "sigma_min": min(test["sigma_per_client"]) if test.get("sigma_per_client") else None,
         "sigma_max": max(test["sigma_per_client"]) if test.get("sigma_per_client") else None}
    r.update({f"test_{m}": test[m] for m in METRICS})
    r.update({f"val_{m}": val[m] for m in METRICS})
    r.update({f"test_recall_{c}": v["recall"] for c, v in test["per_class"].items()})
    return r


def main():
    rows = [row(d) for d in sorted(RUNS.iterdir())
            if (d / "metrics_test.json").exists() and (d / "rounds.csv").exists()   # federated runs only
            and not d.name.startswith("smoke")]                                        # smoke checks are not results
    df = pd.DataFrame(rows)
    out = REPO_ROOT / "results" / "master.csv"
    df.to_csv(out, index=False, float_format="%.5f")
    print(f"{len(df)} federated runs -> {out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
