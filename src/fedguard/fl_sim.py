"""Our federated-learning engine (plan Phase 4, Figure 9.1). One Windows process, no Ray, deterministic.

Each round:
  1. every client starts from the current global weights
  2. trains `local_epochs` epochs on its own partition (plain SGD, or DP-SGD with Opacus when dp.epsilon is set)
  3. sends delta = local - global plus declared metadata (n, lr, sigma, C, B, T)
  4. the server aggregates the deltas (FedAvg / AutoGM / FedGuard) and adds the result to the global model
  5. the global model is evaluated on the VALIDATION split; the best round is kept (early stopping)
At the end the best global model is evaluated once on the TEST split - same protocol as B0.
"""
import copy
import time
import zlib
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import parameters_to_vector

from fedguard import attacks, dp
from fedguard.aggregators import get_aggregator
from fedguard.metrics import attacker_weight_share, honest_rejection_rate
from fedguard.model import build_model
from fedguard.train import class_weights, evaluate, make_loader, make_optimizer, train_epoch


@dataclass
class Client:
    cid: int
    idx: np.ndarray                     # row indices into the training split
    is_attacker: bool = False
    meta: dict = field(default_factory=dict)

    @property
    def n(self):
        return len(self.idx)


def flat(model) -> torch.Tensor:
    return parameters_to_vector([p.detach() for p in model.parameters()]).clone()


def set_flat(model, vec: torch.Tensor) -> None:
    """COPY `vec` into the model's parameters.

    Do NOT use torch's vector_to_parameters here: it makes the parameters *views* of `vec`, so training
    the model would silently modify `vec` (the global model) in place. That bug turned FedAvg into
    sequential training across clients and was caught by the Flower parity check (Findings F24).
    """
    i = 0
    with torch.no_grad():
        for p in model.parameters():
            k = p.numel()
            p.copy_(vec[i:i + k].view_as(p))
            i += k
    if i != vec.numel():
        raise ValueError(f"vector has {vec.numel()} values but the model has {i} parameters")


def round_seed(seed: int, rnd: int, cid: int) -> int:
    """Deterministic, distinct seed for every (run seed, round, client)."""
    return zlib.crc32(f"{seed}-{rnd}-{cid}".encode()) & 0x7FFFFFFF


class FLSimulation:
    def __init__(self, cfg, data, parts, device, log=print):
        self.cfg, self.data, self.device, self.log = cfg, data, device, log
        self.clients = [Client(k, np.asarray(p)) for k, p in enumerate(parts)]
        attacks.validate(cfg["attack"])
        self.attack = cfg["attack"]["type"]
        if self.attack != "none":                    # fixed set of attackers for the whole run
            mask = attacks.choose_attackers(len(self.clients), cfg["attack"]["fraction"], cfg["seed"])
            for c, m in zip(self.clients, mask):
                c.is_attacker = bool(m)
        self.is_attacker = np.array([c.is_attacker for c in self.clients])
        self.dp = dp.is_dp(cfg)
        for c in self.clients:                       # sigma fixed once per client for the whole budget
            c.meta = dp.client_privacy_plan(c.n, cfg, c.cid)
        self.aggregate = get_aggregator(cfg["fl"]["aggregator"]["name"])
        w = class_weights(data.y_tr, data.n_classes, cfg["train"]["class_weight"])
        self.loss_fn = nn.CrossEntropyLoss(weight=None if w is None else w.to(device))
        torch.manual_seed(cfg["seed"])
        self.global_model = build_model(data.n_features, data.n_classes, **cfg["model"]).to(device)
        self.d = sum(p.numel() for p in self.global_model.parameters())

    # ------------------------------------------------------------------ client side
    def local_train(self, client: Client, global_vec: torch.Tensor, rnd: int) -> dict:
        t, f = self.cfg["train"], self.cfg["fl"]
        model = copy.deepcopy(self.global_model)
        set_flat(model, global_vec)
        rs = round_seed(self.cfg["seed"], rnd, client.cid)
        torch.manual_seed(rs)                                   # dropout masks of this client/round
        y = self.data.y_tr[client.idx]
        if client.is_attacker and self.attack == "label_flip":
            y = attacks.label_flip(y)                         # data poisoning: attacks relabelled benign
        loader = make_loader(self.data.X_tr[client.idx], y, t["batch_size"], shuffle=True, seed=rs)
        opt = make_optimizer(model, t["lr"])
        if self.dp:
            p = client.meta
            model, opt, loader = dp.make_private(model, opt, loader, p["sigma"], p["C"],
                                                 noise_seed=rs ^ 0x5EED, device=self.device)
            steps, batch = p["T"], p["B"]                     # Opacus' Poisson batches / expected batch size
        else:
            steps, batch = len(loader) * f["local_epochs"], t["batch_size"]
        losses = [train_epoch(model, loader, opt, self.loss_fn, self.device) for _ in range(f["local_epochs"])]
        delta = flat(model) - global_vec
        if client.is_attacker and self.attack in ("sign_flip", "gaussian"):
            delta = self.poison(delta, rs)                    # model poisoning; declared meta stays honest-looking
        meta = {"n": client.n, "lr": t["lr"], "sigma": client.meta["sigma"], "C": client.meta["C"],
                "B": batch, "T": steps}
        return {"delta": delta, "loss": float(np.mean(losses)), "meta": meta}

    def poison(self, delta: torch.Tensor, rs: int) -> torch.Tensor:
        scale = attacks.attack_scale(self.cfg["attack"])
        if self.attack == "sign_flip":
            return attacks.sign_flip(delta, scale)
        gen = torch.Generator(device=delta.device)
        gen.manual_seed(rs ^ 0xA77AC)
        std = scale * float(delta.norm()) / np.sqrt(delta.numel())     # length = scale x honest length
        return std * torch.randn(delta.shape, generator=gen, device=delta.device, dtype=delta.dtype)

    # ------------------------------------------------------------------ server side
    def noise_radius(self, meta) -> float:
        """Declared expected DP-noise length in the update: lr*sigma*C*sqrt(T*d)/B (0 without DP)."""
        return dp.noise_radius(meta, self.d)

    def privacy_report(self, rounds_run: int) -> dict:
        """epsilon per client for the rounds actually run (<= the target, which covers fl.max_rounds)."""
        if not self.dp:
            return {"epsilon_target": None, "epsilon_spent_max": None}
        spent = [dp.epsilon_spent(c.meta["sigma"], c.meta["q"], c.meta["T"] * rounds_run, c.meta["delta"])
                 for c in self.clients]
        return {"epsilon_target": max(c.meta["target_epsilon"] for c in self.clients),
                "epsilon_target_per_client": [c.meta["target_epsilon"] for c in self.clients],
                "epsilon_spent_max": max(spent),
                "epsilon_spent_per_client": spent, "delta_per_client": [c.meta["delta"] for c in self.clients],
                "sigma_per_client": [c.meta["sigma"] for c in self.clients], "clip": self.clients[0].meta["C"],
                "rounds_budget": self.cfg["fl"]["max_rounds"]}

    def run(self):
        f, t = self.cfg["fl"], self.cfg["train"]
        g = flat(self.global_model)
        best = {"f1": -1.0, "round": 0, "vec": g.clone()}
        bad, rounds, weights_log = 0, [], []
        t_start = time.perf_counter()
        for rnd in range(1, f["max_rounds"] + 1):
            t0 = time.perf_counter()
            outs = [self.local_train(c, g, rnd) for c in self.clients]
            X = torch.stack([o["delta"] for o in outs]).cpu().numpy()
            n = np.array([o["meta"]["n"] for o in outs], dtype=float)
            r = np.array([self.noise_radius(o["meta"]) for o in outs])
            z, w, info = self.aggregate(X, n, r, f["aggregator"])
            g = g + torch.as_tensor(z, dtype=g.dtype, device=g.device)
            set_flat(self.global_model, g)
            train_s = time.perf_counter() - t0

            val = evaluate(self.global_model, self.data.X_va, self.data.y_va, self.data.classes,
                           self.device, t["eval_batch"])
            improved = val["macro_f1"] > best["f1"] + 1e-4
            if improved:
                best = {"f1": val["macro_f1"], "round": rnd, "vec": g.clone()}
                bad = 0
            else:
                bad += 1
            rounds.append({"round": rnd, "val_macro_f1": val["macro_f1"], "val_accuracy": val["accuracy"],
                           "val_benign_fpr": val["benign_fpr"], "client_loss_mean": float(np.mean([o["loss"] for o in outs])),
                           "update_norm": float(np.linalg.norm(z)), "round_seconds": train_s,
                           "hrr": honest_rejection_rate(w, self.is_attacker),
                           "attacker_share": attacker_weight_share(w, self.is_attacker)})
            for c, o, wi, xi in zip(self.clients, outs, w, X):
                weights_log.append({"round": rnd, "client": c.cid, "weight": float(wi), "n": c.n,
                                    "is_attacker": c.is_attacker, "delta_norm": float(np.linalg.norm(xi)),
                                    "client_loss": o["loss"], "declared_radius": float(r[c.cid])})
            self.log(f"  round {rnd:3d}/{f['max_rounds']}  client loss {rounds[-1]['client_loss_mean']:.4f}  "
                     f"val macro-F1 {val['macro_f1']:.4f}  best {best['f1']:.4f} (round {best['round']})  "
                     f"patience {bad}/{f['patience']}{'  *new best' if improved else ''}  ({train_s:.1f}s)"
                     + (f"  weights max {w.max():.2f} HRR {rounds[-1]['hrr']:.2f}" if f["aggregator"]["name"] != "fedavg" else "")
                     + (f" attacker share {rounds[-1]['attacker_share']:.2f}" if self.attack != "none" else ""))
            if bad >= f["patience"] and rnd >= f["min_rounds"]:
                self.log(f"  early stop at round {rnd} (best round {best['round']}, val macro-F1 {best['f1']:.4f})")
                break

        set_flat(self.global_model, best["vec"])
        info = {"best_round": best["round"], "rounds_run": len(rounds), "best_val_macro_f1": best["f1"],
                "train_seconds_total": time.perf_counter() - t_start,
                "seconds_per_round_mean": float(np.mean([x["round_seconds"] for x in rounds])),
                "n_params": self.d, "client_sizes": [c.n for c in self.clients],
                "attackers": [c.cid for c in self.clients if c.is_attacker],
                "hrr_mean": float(np.mean([x["hrr"] for x in rounds])),
                "attacker_share_mean": float(np.mean([x["attacker_share"] for x in rounds])),
                "declared_radius_per_client": [self.noise_radius({**c.meta}) for c in self.clients],
                **self.privacy_report(len(rounds))}
        return self.global_model, rounds, weights_log, info
