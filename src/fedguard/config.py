"""YAML experiment configs: defaults + file + command-line overrides, and a stable run id."""
import copy
import hashlib
import json
from pathlib import Path

import yaml

DEFAULTS = {
    "name": "run",
    "dataset": "nf_unsw",
    "seed": 42,
    "model": {"conv1": 32, "conv2": 64, "hidden": 64, "dropout": 0.3},
    "train": {
        "lr": 0.05,                 # plain SGD, momentum 0, no weight decay (needed by the DP radius formula)
        "batch_size": 256,
        "max_epochs": 200,
        "patience": 20,             # early stopping on validation macro-F1 ...
        "min_epochs": 20,           # ... but never before this epoch (plain SGD starts slow and noisy)
        "class_weight": "none",     # none | sqrt | balanced  (same choice for every baseline)
        "eval_batch": 4096,
        "device": "auto",           # auto | cuda | cpu
    },
    # federated settings (Phase 4+); ignored by the centralised B0 script
    "fl": {
        "clients": 5,
        "alpha": 0.5,               # Dirichlet non-IID level of the partition file
        "local_epochs": 1,
        "max_rounds": 200,
        "patience": 30,             # early stopping on validation macro-F1, in rounds
        "min_rounds": 20,
        "aggregator": {"name": "fedavg"},
    },
    "dp": {"epsilon": None},        # None = no DP (Phase 5 adds DP-SGD)
    "attack": {"type": "none", "fraction": 0.0},   # Phase 6
}


def get_dotted(cfg: dict, key: str):
    for part in key.split("."):
        cfg = cfg[part]
    return cfg


def set_dotted(cfg: dict, key: str, value) -> None:
    parts = key.split(".")
    for part in parts[:-1]:
        cfg = cfg.setdefault(part, {})
    cfg[parts[-1]] = value


def deep_merge(base: dict, extra: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (extra or {}).items():
        out[k] = deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_config(path=None, overrides=None) -> dict:
    cfg = copy.deepcopy(DEFAULTS)
    if path:
        with open(path) as f:
            cfg = deep_merge(cfg, yaml.safe_load(f) or {})
    return deep_merge(cfg, overrides or {})


def config_hash(cfg: dict) -> str:
    """Same config -> same 10-character id (used to skip runs that are already finished)."""
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:10]


def save_config(cfg: dict, path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
