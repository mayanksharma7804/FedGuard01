"""Differential privacy for federated clients (plan Phase 5): DP-SGD with Opacus, the privacy budget and
the declared noise radius used by FedGuard.

DP-SGD in three steps: (1) clip every example's gradient to length <= C, (2) add Gaussian noise of std
sigma*C to the summed batch gradient, (3) an accountant adds up the privacy cost -> epsilon at delta.

Budget rule (plan 10.1, step 1): sigma is chosen ONCE per client for the whole training,
T_round * R steps, where R = fl.max_rounds (the round budget). A new PrivacyEngine is created every round,
so its own get_epsilon() would only cover that round - never use it; use `epsilon_spent` instead.
"""
import math
import warnings

import torch
from opacus import PrivacyEngine
from opacus.accountants import RDPAccountant
from opacus.accountants.utils import get_noise_multiplier

ACCOUNTANT = "rdp"
# two harmless Opacus warnings that would otherwise flood every run log
warnings.filterwarnings("ignore", message="Secure RNG turned off")             # fine for research runs
warnings.filterwarnings("ignore", message="Full backward hook is firing")      # first layer's input has no grad


def is_dp(cfg) -> bool:
    eps = cfg.get("dp", {}).get("epsilon")
    return eps is not None and math.isfinite(float(eps))


def client_delta(n: int, cfg) -> float:
    """delta per client: 1/N by default (plan 10.1, step 3)."""
    d = cfg["dp"].get("delta", "auto")
    return 1.0 / n if d in (None, "auto") else float(d)


def steps_per_round(n: int, batch_size: int, local_epochs: int) -> int:
    """Opacus' Poisson loader has int(1/q) = int(n/B) batches per epoch."""
    return local_epochs * max(1, int(n / batch_size))


def calibrate_sigma(target_epsilon: float, delta: float, sample_rate: float, steps: int) -> float:
    """Smallest noise multiplier that keeps the whole training at (target_epsilon, delta)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(get_noise_multiplier(target_epsilon=target_epsilon, target_delta=delta,
                                          sample_rate=sample_rate, steps=steps, accountant=ACCOUNTANT))


def epsilon_spent(sigma: float, sample_rate: float, steps: int, delta: float) -> float:
    """Epsilon actually spent after `steps` DP-SGD steps (RDP accountant). inf without noise."""
    if sigma <= 0:
        return math.inf
    if steps <= 0:
        return 0.0
    acc = RDPAccountant()
    acc.history = [(sigma, sample_rate, steps)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(acc.get_epsilon(delta))


def noise_radius(meta: dict, d: int) -> float:
    """Expected length of the DP noise inside one client's round update (first-order):
        r = lr * sigma * C * sqrt(T * d) / B
    lr = SGD step size, sigma = noise multiplier, C = clipping norm, T = local steps, d = number of
    parameters, B = (expected) batch size. Each step adds lr * N(0, (sigma*C/B)^2 I_d); T independent steps
    add up like a random walk -> length grows with sqrt(T)."""
    if meta.get("sigma", 0) <= 0:
        return 0.0
    return meta["lr"] * meta["sigma"] * meta["C"] * math.sqrt(meta["T"] * d) / meta["B"]


def client_privacy_plan(n: int, cfg) -> dict:
    """sigma, q, delta and steps for one client of size n under cfg (sigma = 0 without DP)."""
    t, f, dp = cfg["train"], cfg["fl"], cfg["dp"]
    B = t["batch_size"]
    T = steps_per_round(n, B, f["local_epochs"])
    plan = {"n": n, "B": B, "T": T, "lr": t["lr"], "sigma": 0.0, "C": 0.0, "q": min(1.0, B / n),
            "delta": None, "target_epsilon": None, "rounds_budget": f["max_rounds"]}
    if is_dp(cfg):
        plan["C"] = float(dp["clip"])
        plan["delta"] = client_delta(n, cfg)
        plan["target_epsilon"] = float(dp["epsilon"])
        plan["sigma"] = calibrate_sigma(plan["target_epsilon"], plan["delta"], plan["q"], T * f["max_rounds"])
    return plan


def make_private(model, optimizer, loader, sigma: float, clip: float, noise_seed: int, device):
    """Wrap model/optimizer/loader for DP-SGD with Poisson sampling.

    The noise has its OWN generator (seeded per client/round): batch sampling uses the loader's generator and
    dropout uses the global torch RNG, so a run with sigma and the same run with sigma = 0 see exactly the
    same batches and dropout masks (needed by the E0 calibration and for determinism)."""
    model.train()                          # the global copy may still be in eval mode after evaluation
    gen = torch.Generator(device=device)
    gen.manual_seed(noise_seed)
    engine = PrivacyEngine(accountant=ACCOUNTANT)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return engine.make_private(module=model, optimizer=optimizer, data_loader=loader,
                                   noise_multiplier=sigma, max_grad_norm=clip, poisson_sampling=True,
                                   noise_generator=gen)
