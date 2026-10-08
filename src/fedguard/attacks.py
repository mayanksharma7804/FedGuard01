"""Byzantine attacks (plan Phase 6, paper O4). Attackers are SMART: they also run DP-SGD when DP is on and
declare a normal-looking sigma, so they cannot be caught just by their declared noise level.

  label_flip       DATA poisoning: every attack row of the attacker's data is relabelled 'benign'
                   (the most dangerous flip for an IDS: attacks become invisible)
  sign_flip        MODEL poisoning: send -scale x the honest update
  gaussian         MODEL poisoning: send random noise instead of an update; its length is
                   scale x the length of the honest update the attacker computed

config:  attack: {type: none | label_flip | sign_flip | gaussian, fraction: 0.3, scale: <optional>}
"""
import zlib

import numpy as np

ATTACKS = ("none", "label_flip", "sign_flip", "gaussian")
DEFAULT_SCALE = {"sign_flip": 1.0, "gaussian": 10.0, "label_flip": 1.0, "none": 1.0}


def label_flip(y, benign=0):
    y = np.array(y, copy=True)
    y[y != benign] = benign
    return y


def sign_flip(delta, scale=1.0):
    return -scale * delta


def gaussian_update(delta, std, rng):
    return rng.normal(0.0, std, size=delta.shape)


def attack_scale(attack_cfg) -> float:
    s = attack_cfg.get("scale")
    return DEFAULT_SCALE[attack_cfg["type"]] if s is None else float(s)


def choose_attackers(n_clients: int, fraction: float, seed: int) -> np.ndarray:
    """Which clients are attackers: round(fraction * K) of them, drawn once per run from the seed (fixed for
    the whole run). Returns a boolean mask."""
    if fraction < 0 or fraction >= 1:
        raise ValueError("attack fraction must be in [0, 1)")
    k = int(round(fraction * n_clients))
    rng = np.random.default_rng(zlib.crc32(f"attackers-{seed}".encode()))
    mask = np.zeros(n_clients, dtype=bool)
    mask[rng.choice(n_clients, size=k, replace=False)] = True
    return mask


def validate(attack_cfg):
    if attack_cfg["type"] not in ATTACKS:
        raise ValueError(f"unknown attack '{attack_cfg['type']}'; available: {ATTACKS}")
