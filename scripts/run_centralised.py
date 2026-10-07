"""Phase 3: centralised baseline B0 (one model trained on all training data).

    python scripts/run_centralised.py                                  # seeds 42 43 44, config b0_centralised.yaml
    python scripts/run_centralised.py --seeds 42 --class-weight sqrt --tag cw_sqrt
    python scripts/run_centralised.py --device cpu --seeds 42 --max-epochs 3 --tag cpu_timing

Each run writes results/runs/<name>_<tag>_seed<seed>/ (config, history, val/test metrics, model).
Finished runs are skipped unless --force. With several seeds, a summary goes to results/<name>/.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from fedguard.config import load_config, save_config
from fedguard.data import load_processed
from fedguard.model import build_model, count_parameters
from fedguard.train import evaluate, fit_centralised, get_device
from fedguard.utils import REPO_ROOT, set_seed

KEY_METRICS = ["macro_f1", "weighted_f1", "accuracy", "benign_fpr", "attack_detection_rate", "binary_f1"]


def run_one(cfg, run_dir: Path, force=False):
    if (run_dir / "metrics_test.json").exists() and not force:
        print(f"[skip] {run_dir.name} already finished")
        return json.load(open(run_dir / "metrics_test.json"))
    run_dir.mkdir(parents=True, exist_ok=True)
    live = open(run_dir / "train.log", "w", encoding="utf8", buffering=1)   # watch this file while training

    def log(msg):
        print(msg, flush=True)
        live.write(msg + "\n")

    set_seed(cfg["seed"])
    data = load_processed(cfg["dataset"])
    device = get_device(cfg["train"]["device"])
    model = build_model(data.n_features, data.n_classes, **cfg["model"])
    log(f"[run] {run_dir.name}  device={device}  params={count_parameters(model):,}  "
        f"train={len(data.y_tr):,} val={len(data.y_va):,} test={len(data.y_te):,}")
    model, history, info = fit_centralised(model, data, cfg, device, log=log)

    eb = cfg["train"]["eval_batch"]
    val = evaluate(model, data.X_va, data.y_va, data.classes, device, eb)
    test = evaluate(model, data.X_te, data.y_te, data.classes, device, eb)
    for d in (val, test):
        d.update(info, seed=cfg["seed"], data_fingerprint=data.fingerprint,
                 device=str(device), parameters=count_parameters(model))
    run_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, run_dir / "config.yaml")
    pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
    json.dump(val, open(run_dir / "metrics_val.json", "w"), indent=2)
    json.dump(test, open(run_dir / "metrics_test.json", "w"), indent=2)
    torch.save(model.state_dict(), run_dir / "model.pt")
    log(f"      test macro-F1 {test['macro_f1']:.4f}  acc {test['accuracy']:.4f}  "
        f"benign FPR {test['benign_fpr']:.4f}  detection {test['attack_detection_rate']:.4f}  "
        f"best epoch {info['best_epoch']}  {info['seconds_per_epoch_mean']:.1f}s/epoch")
    live.close()
    return test


def summarise(name, tag, results, run_dirs, classes):
    out = REPO_ROOT / "results" / (name if not tag else f"{name}_{tag}")
    fig_dir = REPO_ROOT / "results" / "figures"
    out.mkdir(parents=True, exist_ok=True)
    rows = [{"seed": r["seed"], **{k: r[k] for k in KEY_METRICS}, "best_epoch": r["best_epoch"],
             "epochs_run": r["epochs_run"], "seconds_per_epoch": r["seconds_per_epoch_mean"]} for r in results]
    df = pd.DataFrame(rows)
    agg = pd.DataFrame([{"seed": "mean", **df.mean(numeric_only=True).drop("seed").to_dict()},
                        {"seed": "std", **df.std(numeric_only=True).drop("seed").to_dict()}])
    summary = pd.concat([df, agg], ignore_index=True)
    summary.to_csv(out / "summary.csv", index=False, float_format="%.4f")

    per = {m: pd.DataFrame({r["seed"]: {c: r["per_class"][c][m] for c in classes} for r in results})
           for m in ("precision", "recall", "f1")}
    pc = pd.DataFrame({"support_test": [results[0]["per_class"][c]["support"] for c in classes]}, index=classes)
    for m, t in per.items():
        pc[f"{m}_mean"], pc[f"{m}_std"] = t.mean(axis=1), t.std(axis=1)
    pc.to_csv(out / "per_class.csv", float_format="%.4f")

    label = name if not tag else f"{name} ({tag})"
    # confusion matrix: counts summed over seeds, each row normalised to 1
    cm = np.sum([np.array(r["confusion_matrix"]) for r in results], axis=0).astype(float)
    cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(7.5, 6.3))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes))); ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(classes))); ax.set_yticklabels(classes, fontsize=8)
    for i in range(len(classes)):
        for j in range(len(classes)):
            if cmn[i, j] >= 0.01:
                ax.text(j, i, f"{cmn[i, j]:.2f}", ha="center", va="center", fontsize=6.5,
                        color="white" if cmn[i, j] > 0.6 else "black")
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.set_title(f"{label}: confusion matrix (test, {len(results)} seeds, row-normalised)", fontsize=9)
    fig.colorbar(im, fraction=0.04); fig.tight_layout()
    fig.savefig(fig_dir / f"{out.name}_confusion.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 3.6))
    x = np.arange(len(classes))
    ax.bar(x, pc["recall_mean"], yerr=pc["recall_std"], color="#2A9D8F", capsize=3)
    for i, (m, n) in enumerate(zip(pc["recall_mean"], pc["support_test"])):
        ax.text(i, min(m + 0.04, 1.02), f"n={n}", ha="center", fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels(classes, rotation=35, ha="right"); ax.set_ylim(0, 1.1)
    ax.set_ylabel("recall (test)"); ax.set_title(f"{label}: per-class recall, mean +/- std over seeds", fontsize=10)
    fig.tight_layout(); fig.savefig(fig_dir / f"{out.name}_per_class_recall.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 3.4))
    for d in run_dirs:
        h = pd.read_csv(d / "history.csv")
        ax.plot(h["epoch"], h["val_macro_f1"], label=f"seed {d.name.split('_seed')[-1]}")
    ax.set_xlabel("epoch"); ax.set_ylabel("validation macro-F1"); ax.grid(color="#eee")
    ax.legend(frameon=False, fontsize=8); ax.set_title(f"{label}: learning curves", fontsize=10)
    fig.tight_layout(); fig.savefig(fig_dir / f"{out.name}_learning_curves.png", dpi=150); plt.close(fig)

    print(f"\nsummary -> {out.relative_to(REPO_ROOT)}")
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\nper-class (test, mean over seeds):")
    print(pc[["support_test", "precision_mean", "recall_mean", "f1_mean"]].to_string(float_format=lambda v: f"{v:.3f}"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO_ROOT / "configs" / "b0_centralised.yaml"))
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--class-weight", choices=["none", "sqrt", "balanced"])
    ap.add_argument("--device")
    ap.add_argument("--max-epochs", type=int)
    ap.add_argument("--lr", type=float)
    ap.add_argument("--tag", default="", help="suffix for run folders, e.g. cw_sqrt")
    ap.add_argument("--force", action="store_true", help="re-run even if results exist")
    args = ap.parse_args()

    over = {"train": {}}
    if args.class_weight: over["train"]["class_weight"] = args.class_weight
    if args.device: over["train"]["device"] = args.device
    if args.max_epochs: over["train"]["max_epochs"] = args.max_epochs
    if args.lr: over["train"]["lr"] = args.lr

    results, dirs = [], []
    for s in args.seeds:
        cfg = load_config(args.config, {**over, "seed": s})
        d = REPO_ROOT / "results" / "runs" / f"{cfg['name']}{'_' + args.tag if args.tag else ''}_seed{s}"
        results.append(run_one(cfg, d, args.force)); dirs.append(d)
    if len(results) > 1:
        summarise(cfg["name"], args.tag, results, dirs, load_processed(cfg["dataset"]).classes)


if __name__ == "__main__":
    main()
