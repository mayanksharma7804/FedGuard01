"""Flower integration (plan section 12.2 step 6). Used for the engine-parity check and the live demo.

`AggregatorStrategy` is Flower's FedAvg with ONE change: `aggregate_train` turns the clients' returned
weights into deltas, calls the SAME NumPy aggregator our engine uses (fedavg / autogm / fedguard), and
adds the result to the current global model. Client-side training reuses the engine's code path, so
Flower and fl_sim.py should produce the same numbers.
"""
import copy

import numpy as np
import torch
from flwr.app import ArrayRecord, MetricRecord
from flwr.serverapp.strategy import FedAvg

from fedguard import attacks, dp
from fedguard.aggregators import get_aggregator
from fedguard.fl_sim import flat, poison_delta, round_seed, set_flat
from fedguard.train import make_loader, make_optimizer, train_epoch


def write_csv_atomic(rows, path):
    """Write rows (list of dicts) to CSV via a temporary file, so a reader never sees half a file."""
    import os
    import pandas as pd
    tmp = f"{path}.tmp"
    pd.DataFrame(rows).to_csv(tmp, index=False)
    os.replace(tmp, path)


def vec_from_arrays(template_model, arrays: ArrayRecord) -> torch.Tensor:
    m = copy.deepcopy(template_model)
    m.load_state_dict(arrays.to_torch_state_dict())
    return flat(m).cpu()


def arrays_from_vec(template_model, vec: torch.Tensor) -> ArrayRecord:
    m = copy.deepcopy(template_model)
    set_flat(m, vec.to(next(m.parameters()).device))
    return ArrayRecord(m.state_dict())


def is_attacker(cfg, cid) -> bool:
    """Same attacker choice as FLSimulation (fixed per run from the seed)."""
    a = cfg.get("attack") or {"type": "none", "fraction": 0.0}
    if a["type"] == "none":
        return False
    return bool(attacks.choose_attackers(cfg["fl"]["clients"], a["fraction"], cfg["seed"])[cid])


def client_round(model, X, y, cfg, cid, rnd, loss_fn, device):
    """Exactly what FLSimulation.local_train does for one client in one round (DP-SGD if dp.epsilon is set,
    and the configured attack if this client is an attacker; its declared metadata stays honest-looking).
    `model` is trained in place; returns (mean loss, declared metadata)."""
    rs = round_seed(cfg["seed"], rnd, cid)
    torch.manual_seed(rs)
    attack = (cfg.get("attack") or {}).get("type", "none")
    bad = is_attacker(cfg, cid)
    if bad and attack == "label_flip":
        y = attacks.label_flip(y)
    g0 = flat(model)
    loader = make_loader(X, y, cfg["train"]["batch_size"], shuffle=True, seed=rs, device=device)
    opt = make_optimizer(model, cfg["train"]["lr"])
    plan = dp.client_privacy_plan(len(y), cfg, cid)
    net = model
    if dp.is_dp(cfg):
        net, opt, loader = dp.make_private(model, opt, loader, plan["sigma"], plan["C"], noise_seed=rs ^ 0x5EED,
                                           device=device)
        steps, batch = plan["T"], plan["B"]
    else:
        steps, batch = len(loader) * cfg["fl"]["local_epochs"], cfg["train"]["batch_size"]
    losses = [train_epoch(net, loader, opt, loss_fn, device) for _ in range(cfg["fl"]["local_epochs"])]
    if bad and attack in ("sign_flip", "gaussian"):
        set_flat(model, g0 + poison_delta(flat(model) - g0, cfg["attack"], rs))
    meta = {"n": len(y), "lr": cfg["train"]["lr"], "sigma": plan["sigma"], "C": plan["C"],
            "B": batch, "T": steps}
    return float(np.mean(losses)), meta


class AggregatorStrategy(FedAvg):
    """FedAvg message flow + our aggregator. Records per-round client weights for the demo dashboard."""

    def __init__(self, template_model, aggregator: str, params: dict, radius_fn=None, live_csv=None, **kwargs):
        """live_csv: if given, the weights log is rewritten there after every round (live demo dashboard)."""
        super().__init__(**kwargs)
        self.live_csv = live_csv
        self.template = template_model
        self.aggregate_fn = get_aggregator(aggregator)
        self.params = params
        d = sum(p.numel() for p in template_model.parameters())
        self.radius_fn = radius_fn or (lambda meta: dp.noise_radius(meta, d))     # declared noise radius
        self.current = None
        self.weights_log = []

    def configure_train(self, server_round, arrays, config, grid):
        self.current = vec_from_arrays(self.template, arrays)      # remember the global model of this round
        return super().configure_train(server_round, arrays, config, grid)

    def aggregate_train(self, server_round, replies):
        valid, _ = self._check_and_log_replies(replies, is_train=True)
        if not valid:
            return None, None
        valid = sorted(valid, key=lambda m: int(m.content["metrics"]["cid"]))   # stable client order
        g = self.current
        X = np.stack([(vec_from_arrays(self.template, m.content["arrays"]) - g).numpy() for m in valid])
        metas = [dict(m.content["metrics"]) for m in valid]
        n = np.array([mt["num-examples"] for mt in metas], dtype=float)
        r = np.array([self.radius_fn(mt) for mt in metas])
        z, w, info = self.aggregate_fn(X.astype(np.float64), n, r, self.params)
        new = g + torch.as_tensor(z, dtype=g.dtype)
        for k, (mt, wi) in enumerate(zip(metas, w)):
            row = {"round": server_round, "client": int(mt["cid"]), "weight": float(wi),
                   "declared_radius": float(r[k]), "delta_norm": float(np.linalg.norm(X[k]))}
            for key in ("dist", "residual", "excess"):
                if key in info:
                    row[key] = float(info[key][k])
            self.weights_log.append(row)
        if self.live_csv is not None:
            write_csv_atomic(self.weights_log, self.live_csv)
        loss = float(np.average([mt["train_loss"] for mt in metas], weights=n))
        return arrays_from_vec(self.template, new), MetricRecord({"train_loss": loss})
