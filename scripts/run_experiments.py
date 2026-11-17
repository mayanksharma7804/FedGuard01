"""Run federated experiments from config files (plan Phase 4, section 9.1).

    python scripts/run_experiments.py --config configs/b1_fedavg.yaml                  # one run
    python scripts/run_experiments.py --config configs/b1_fedavg.yaml --set fl.alpha=0.1 seed=43
    python scripts/run_experiments.py --sweep configs/sweeps/b1_fedavg.yaml            # many runs
    python scripts/run_experiments.py --sweep configs/sweeps/b1_fedavg.yaml --shard 0/2  # half of them
    python scripts/run_experiments.py --sweep ... --dry-run                             # just list them

A sweep file has a base config and a grid of dotted keys; every combination is one run:
    base: configs/b1_fedavg.yaml
    grid:
      fl.alpha: [0.1, 0.5, 100]
      seed: [42, 43, 44]

Finished runs (metrics_test.json exists) are skipped, so a crashed or stopped sweep can simply be started
again (--force re-runs). Two shards (--shard 0/2 and --shard 1/2) in two terminals = 2 runs in parallel.
"""
import argparse
import itertools
import json
import os
import platform
import subprocess
import time
from pathlib import Path

import pandas as pd
import torch
import yaml

from fedguard.config import config_hash, deep_merge, get_dotted, load_config, save_config, set_dotted
from fedguard.data import load_processed
from fedguard.fl_sim import FLSimulation, PauseRequested
from fedguard.partition import load_partition, partition_path
from fedguard.train import evaluate, get_device
from fedguard.utils import REPO_ROOT, set_seed

RUNS = REPO_ROOT / "results" / "runs"
PAUSE = RUNS / "PAUSE"          # create this file to pause every running sweep at the next round boundary
PAUSE_AFTER_RUN = RUNS / "PAUSE_AFTER_RUN"   # ... or this one: finish the current run, then stop (clean boundary)


THROTTLE = RUNS / "THROTTLE"    # written by scripts/windows/thermal_guard.ps1 while the laptop is too hot


def pause_requested() -> bool:
    return PAUSE.exists()


def wait_while_hot(log=print):
    """Thermal guard: while results/runs/THROTTLE exists, wait before the next client (GPU and CPU idle).
    Waiting never changes a result - every client/round is seeded independently of wall time."""
    if not THROTTLE.exists():
        return
    t0 = time.time()
    log(f"  [thermal] laptop too hot - waiting to cool down ({time.strftime('%H:%M:%S')})")
    while THROTTLE.exists() and not PAUSE.exists():
        time.sleep(5)
    log(f"  [thermal] resumed after {time.time() - t0:.0f}s")


def parse_value(text):
    return yaml.safe_load(text)                     # "0.1" -> 0.1, "none" -> "none", "[1,2]" -> list


def run_name(cfg) -> str:
    f = cfg["fl"]
    parts = [cfg["name"], f"a{f['alpha']:g}", f"k{f['clients']}", f"e{f['local_epochs']}"]
    agg = f["aggregator"]
    if agg["name"] != "fedavg":                             # e.g. autogm-lam1, fedguard-modquadrature-noi3
        short = {"lam_scale": "lam", "k_mad": "k", "mode": "", "noise_tol_z": "r4z", "use_r_eff": "reff", "s_floor": "sf"}
        tags = [f"{short.get(k, k[:3])}{v:g}" if isinstance(v, (int, float)) and not isinstance(v, bool)
                else f"{short.get(k, k[:3])}{v}" for k, v in sorted(agg.items()) if k not in ("name", "outer")]
        parts.append("-".join([agg["name"]] + tags))
    eps = cfg["dp"].get("epsilon")
    if isinstance(eps, (list, tuple)):                      # heterogeneous privacy, e.g. eps1-3-8
        parts.append("eps" + "-".join(f"{e:g}" for e in eps) + f"_c{cfg['dp']['clip']:g}")
    elif eps is not None:
        parts.append(f"eps{eps:g}_c{cfg['dp']['clip']:g}")
    if cfg["attack"]["type"] != "none":
        parts.append(f"{cfg['attack']['type']}{cfg['attack']['fraction']:g}")
    parts.append(f"s{cfg['seed']}")
    return "_".join(parts) + "_" + config_hash(cfg)[:6]


def normalise(cfg):
    """Aggregator options set to null mean 'not used': drop them, so a config written with `noise_tol_z: null`
    is the same run (same hash) as one that leaves the key out."""
    agg = cfg["fl"]["aggregator"]
    cfg["fl"]["aggregator"] = {k: v for k, v in agg.items() if v is not None}
    return cfg


def expand_sweep(path):
    """base config + include files (deep-merged in order, e.g. the frozen FedGuard settings) + set + grid."""
    spec = yaml.safe_load(open(path))
    extra = {}
    for inc in spec.get("include", []):
        inc_path = REPO_ROOT / inc
        if not inc_path.exists():
            raise FileNotFoundError(f"{path} includes {inc}, which does not exist yet (FedGuard settings are frozen "
                                    f"on validation first - Findings F50)")
        extra = deep_merge(extra, yaml.safe_load(open(inc_path)) or {})
    base = load_config(REPO_ROOT / spec["base"], deep_merge(extra, spec.get("set") or {}))
    keys = list(spec.get("grid", {}))
    configs = []
    for combo in itertools.product(*[spec["grid"][k] for k in keys]):
        cfg = load_config(overrides=base)
        for k, v in zip(keys, combo):
            set_dotted(cfg, k, v)
        configs.append(normalise(cfg))
    return configs


def git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, capture_output=True,
                              text=True, timeout=10).stdout.strip()
    except Exception:
        return "unknown"


def run_one(cfg, force=False):
    run_dir = RUNS / run_name(cfg)
    if (run_dir / "metrics_test.json").exists() and not force:
        print(f"[skip] {run_dir.name} already finished")
        return run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt = run_dir / "checkpoint.pt"
    live = open(run_dir / "train.log", "a" if ckpt.exists() else "w", encoding="utf8", buffering=1)

    def log(msg):
        print(msg, flush=True)
        live.write(msg + "\n")

    set_seed(cfg["seed"])
    data = load_processed(cfg["dataset"])
    f = cfg["fl"]
    parts, _ = load_partition(partition_path(cfg["dataset"], f["alpha"], f["clients"], cfg["seed"]),
                              fingerprint=data.fingerprint)
    device = get_device(cfg["train"]["device"])
    save_config(cfg, run_dir / "config.yaml")
    log(f"[run] {run_dir.name}  device={device}  clients={f['clients']} alpha={f['alpha']:g} "
        f"local_epochs={f['local_epochs']}  aggregator={f['aggregator']['name']}  "
        f"client sizes={[len(p) for p in parts]}")

    sim = FLSimulation(cfg, data, parts, device, log=log)
    if sim.dp:
        dpc = cfg["dp"]
        log(f"  DP-SGD: target eps={dpc['epsilon']} over {f['max_rounds']} rounds, delta=1/N per client, C={dpc['clip']:g}")
        for c in sim.clients:
            log(f"    client {c.cid}: n={c.n:6d}  eps={c.meta['target_epsilon']:g}  T={c.meta['T']:4d}/round  sigma={c.meta['sigma']:.3f}  "
                f"delta={c.meta['delta']:.2e}  declared radius={sim.noise_radius(c.meta):.4f}")
    try:
        model, rounds, weights_log, info = sim.run(checkpoint=ckpt, should_pause=pause_requested,
                                                   before_client=lambda: wait_while_hot(log))
    except PauseRequested:
        live.close()
        raise
    eb = cfg["train"]["eval_batch"]
    val = evaluate(model, data.X_va, data.y_va, data.classes, device, eb)
    test = evaluate(model, data.X_te, data.y_te, data.classes, device, eb)
    for d in (val, test):
        d.update(info, seed=cfg["seed"], data_fingerprint=data.fingerprint, device=str(device))
    pd.DataFrame(rounds).to_csv(run_dir / "rounds.csv", index=False)
    pd.DataFrame(weights_log).to_csv(run_dir / "weights_log.csv", index=False)
    json.dump(val, open(run_dir / "metrics_val.json", "w"), indent=2)
    json.dump({"git_commit": git_commit(), "host": platform.node(), "torch": torch.__version__,
               "finished": time.strftime("%Y-%m-%d %H:%M:%S"), "config_hash": config_hash(cfg)},
              open(run_dir / "meta.json", "w"), indent=2)
    json.dump(test, open(run_dir / "metrics_test.json", "w"), indent=2)   # written last = "finished"
    ckpt.unlink(missing_ok=True)
    if sim.dp:
        log(f"      privacy: eps spent (max over clients) {info['epsilon_spent_max']:.3f} after {info['rounds_run']} rounds"
            f" (target {info['epsilon_target']:g} for {info['rounds_budget']})")
    log(f"      test macro-F1 {test['macro_f1']:.4f}  acc {test['accuracy']:.4f}  benign FPR {test['benign_fpr']:.4f}"
        f"  detection {test['attack_detection_rate']:.4f}  best round {info['best_round']}/{info['rounds_run']}"
        f"  {info['seconds_per_round_mean']:.1f}s/round")
    live.close()
    return run_dir


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--config")
    src.add_argument("--sweep", nargs="+", help="one or more sweep files: one queue, duplicates run once")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="dotted overrides, e.g. fl.alpha=0.1")
    ap.add_argument("--shard", default="0/1", help="i/n: run every n-th config starting at i")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--resume", action="store_true", help="skip finished runs (always on; kept for the plan's commands)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--threads", type=int, default=int(os.environ.get("FEDGUARD_THREADS", "2")),
                    help="CPU threads for torch per process (training is GPU-bound; fewer threads = cooler CPU)")
    args = ap.parse_args()
    torch.set_num_threads(max(1, args.threads))

    if args.sweep:
        configs, seen = [], set()
        for path in args.sweep:                           # several sweeps = one de-duplicated queue, in order
            for cfg in expand_sweep(path):
                if run_name(cfg) not in seen:
                    seen.add(run_name(cfg)); configs.append(cfg)
    else:
        configs = [load_config(args.config)]
    for kv in args.set:
        k, v = kv.split("=", 1)
        for cfg in configs:
            set_dotted(cfg, k, parse_value(v))
    i, n = map(int, args.shard.split("/"))
    configs = configs[i::n]
    print(f"{len(configs)} run(s) in shard {args.shard}")
    for cfg in configs:
        if args.dry_run:
            done = (RUNS / run_name(cfg) / "metrics_test.json").exists()
            print(("  [done] " if done else "  [todo] ") + run_name(cfg))
            continue
        if pause_requested() or PAUSE_AFTER_RUN.exists():
            print(f"PAUSED (before {run_name(cfg)}, every earlier run is complete): delete the PAUSE file and "
                  f"start the sweep again to continue", flush=True)
            return
        try:
            run_one(cfg, args.force)
        except PauseRequested:
            print(f"PAUSED inside {run_name(cfg)}: checkpoint saved. Delete {PAUSE} and start the sweep again "
                  f"to continue from the same round.", flush=True)
            return


if __name__ == "__main__":
    main()
