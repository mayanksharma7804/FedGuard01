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
import hashlib
import itertools
import json
import os
import platform
import subprocess
import threading
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


THERMAL_LOG = REPO_ROOT / "logs" / "thermal.log"
CPU_TEMP_PS = ("[math]::Round(((Get-Counter '\\Thermal Zone Information(*)\\Temperature').CounterSamples | "
               "Measure-Object CookedValue -Maximum).Maximum - 273.15, 1)")


class ThermalGuard(threading.Thread):
    """Keeps the laptop cool (Findings F49). Every `every` seconds it reads the CPU thermal zone (Windows) and the
    GPU temperature; at >= high C it sets `hot`, and the runner waits before the next client until the hottest
    sensor is back at <= low C. It lives INSIDE the training process, so it can never die and leave training
    stuck (that happened once with a separate guard process, 18 Nov) and never outlives the training either.
    Waiting never changes a result - every client and round is seeded independently of wall time."""

    def __init__(self, high, low, every):
        super().__init__(daemon=True, name="thermal-guard")
        self.high, self.low, self.every = high, low, every
        self.hot = threading.Event()
        self.flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)

    def _read(self, cmd):
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=20, creationflags=self.flags).stdout
            return float(out.strip().splitlines()[0].replace(",", "."))
        except Exception:
            return None

    def temps(self):
        cpu = self._read(["powershell", "-NoProfile", "-Command", CPU_TEMP_PS]) if os.name == "nt" else None
        gpu = self._read(["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"])
        return cpu, gpu

    def write(self, msg):
        try:
            THERMAL_LOG.parent.mkdir(parents=True, exist_ok=True)
            with open(THERMAL_LOG, "a", encoding="utf-8") as f:
                f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {msg}\n")
        except OSError:
            pass

    def run(self):
        self.write(f"thermal guard started in the training process (high {self.high:g} C, low {self.low:g} C, "
                   f"every {self.every:g} s)")
        while True:
            cpu, gpu = self.temps()
            vals = [t for t in (cpu, gpu) if t is not None]
            if vals:
                t = max(vals)
                if t >= self.high and not self.hot.is_set():
                    self.hot.set(); self.write(f"HOT  cpu {cpu} C  gpu {gpu} C -> cooling pause on")
                elif t <= self.low and self.hot.is_set():
                    self.hot.clear(); self.write(f"OK   cpu {cpu} C  gpu {gpu} C -> cooling pause off")
            self.write(f"temps cpu {cpu} C  gpu {gpu} C  cooling {self.hot.is_set()}")
            time.sleep(self.every)


GUARD = None                    # set in main() when --thermal-high is given


def disable_power_throttling():
    """Windows 11 puts processes it sees as 'background' (no visible window, e.g. started through WMI by
    queue_start.ps1) into EcoQoS power throttling: slow clocks / efficiency cores. Training is bound by one CPU
    core launching GPU kernels, and throttling made rounds ~3.7x slower (139 s instead of 37 s, 18 Nov, F51).
    Opt this process out (PROCESS_POWER_THROTTLING_EXECUTION_SPEED with state 0 = never throttle)."""
    if os.name != "nt":
        return
    try:
        import ctypes
        from ctypes import wintypes

        class PowerThrottlingState(ctypes.Structure):
            _fields_ = [("Version", wintypes.ULONG), ("ControlMask", wintypes.ULONG), ("StateMask", wintypes.ULONG)]

        state = PowerThrottlingState(1, 0x1 | 0x4, 0)   # EXECUTION_SPEED | IGNORE_TIMER_RESOLUTION, both off
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.GetCurrentProcess.restype = wintypes.HANDLE      # pointer-sized handle (64-bit)
        k32.SetProcessInformation.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
        k32.SetProcessInformation.restype = wintypes.BOOL
        ok = k32.SetProcessInformation(k32.GetCurrentProcess(), 4, ctypes.byref(state), ctypes.sizeof(state))
        print(f"power throttling (EcoQoS) disabled for this process: {bool(ok)}", flush=True)
    except Exception as e:                                  # never let this stop training
        print(f"could not disable power throttling: {e}", flush=True)


def pause_requested() -> bool:
    return PAUSE.exists()


def wait_while_hot(log=print):
    """Before each client: wait while the thermal guard says the laptop is too hot (GPU and CPU idle)."""
    if GUARD is None or not GUARD.hot.is_set():
        return
    t0 = time.time()
    log(f"  [thermal] laptop too hot - waiting to cool down ({time.strftime('%H:%M:%S')})")
    while GUARD.hot.is_set() and not PAUSE.exists():
        time.sleep(2)
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
    ap.add_argument("--skip-file", help="text file of run names finished on ANOTHER machine (one per line, as written "
                                        "by scripts/export_runs.py); they are treated as done")
    ap.add_argument("--thermal-high", type=float, help="thermal guard: pause before the next client at this CPU/GPU "
                                                       "temperature (C); off when not given (queue_start.ps1 passes 75)")
    ap.add_argument("--thermal-low", type=float, default=68.0, help="thermal guard: resume at this temperature (C)")
    ap.add_argument("--thermal-every", type=float, default=25.0, help="thermal guard: seconds between readings")
    ap.add_argument("--max-hours", type=float, help="do not START a new run after this many hours (e.g. 11 on Kaggle, "
                                                    "whose sessions end at 12 h); finished runs stay complete")
    args = ap.parse_args()
    torch.set_num_threads(max(1, args.threads))
    if not args.dry_run:
        disable_power_throttling()
    global GUARD
    if args.thermal_high is not None and not args.dry_run:
        GUARD = ThermalGuard(args.thermal_high, args.thermal_low, args.thermal_every)
        GUARD.start()
    t_start = time.time()
    elsewhere = set()
    if args.skip_file and Path(args.skip_file).exists():
        elsewhere = {ln.strip() for ln in open(args.skip_file, encoding="utf-8") if ln.strip() and not ln.startswith("#")}

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
    queue_id = hashlib.sha256("|".join(run_name(c) for c in configs).encode()).hexdigest()[:8]
    i, n = map(int, args.shard.split("/"))
    total = len(configs)
    configs = configs[i::n]
    print(f"queue id {queue_id} ({total} runs in the whole queue) - {len(configs)} run(s) in shard {args.shard}"
          + (f", {len(elsewhere)} run names finished elsewhere" if elsewhere else ""), flush=True)
    for cfg in configs:
        name = run_name(cfg)
        if args.dry_run:
            done = (RUNS / name / "metrics_test.json").exists()
            print(("  [done] " if done else "  [done-elsewhere] " if name in elsewhere else "  [todo] ") + name)
            continue
        if name in elsewhere and not args.force and not (RUNS / name / "metrics_test.json").exists():
            print(f"[skip] {name} finished on another machine")
            continue
        if args.max_hours is not None and (time.time() - t_start) / 3600 >= args.max_hours:
            print(f"TIME LIMIT ({args.max_hours:g} h) reached before {name}: every earlier run is complete. "
                  f"Export the results now.", flush=True)
            return
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
