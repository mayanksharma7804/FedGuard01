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

from fedguard import dp
from fedguard.aggregators import get_aggregator
from fedguard.fl_sim import flat, round_seed, set_flat
from fedguard.train import make_loader, make_optimizer, train_epoch


def vec_from_arrays(template_model, arrays: ArrayRecord) -> torch.Tensor:
    m = copy.deepcopy(template_model)
    m.load_state_dict(arrays.to_torch_state_dict())
    return flat(m).cpu()


def arrays_from_vec(template_model, vec: torch.Tensor) -> ArrayRecord:
    m = copy.deepcopy(template_model)
    set_flat(m, vec.to(next(m.parameters()).device))
    return ArrayRecord(m.state_dict())


def client_round(model, X, y, cfg, cid, rnd, loss_fn, device):
    """Exactly what FLSimulation.local_train does for one client in one round (DP-SGD if dp.epsilon is set).
    `model` is trained in place; returns (mean loss, declared metadata)."""
    rs = round_seed(cfg["seed"], rnd, cid)
    torch.manual_seed(rs)
    loader = make_loader(X, y, cfg["train"]["batch_size"], shuffle=True, seed=rs)
    opt = make_optimizer(model, cfg["train"]["lr"])
    plan = dp.client_privacy_plan(len(y), cfg)
    net = model
    if dp.is_dp(cfg):
        net, opt, loader = dp.make_private(model, opt, loader, plan["sigma"], plan["C"], noise_seed=rs ^ 0x5EED,
                                           device=device)
        steps = plan["T"]
    else:
        steps = len(loader) * cfg["fl"]["local_epochs"]
    losses = [train_epoch(net, loader, opt, loss_fn, device) for _ in range(cfg["fl"]["local_epochs"])]
    meta = {"n": len(y), "lr": cfg["train"]["lr"], "sigma": plan["sigma"], "C": plan["C"],
            "B": cfg["train"]["batch_size"], "T": steps}
    return float(np.mean(losses)), meta


class AggregatorStrategy(FedAvg):
    """FedAvg message flow + our aggregator. Records per-round client weights for the demo dashboard."""

    def __init__(self, template_model, aggregator: str, params: dict, radius_fn=None, **kwargs):
        super().__init__(**kwargs)
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
        z, w, _ = self.aggregate_fn(X.astype(np.float64), n, r, self.params)
        new = g + torch.as_tensor(z, dtype=g.dtype)
        for mt, wi in zip(metas, w):
            self.weights_log.append({"round": server_round, "client": int(mt["cid"]), "weight": float(wi),
                                     "declared_radius": float(self.radius_fn(mt))})
        loss = float(np.average([mt["train_loss"] for mt in metas], weights=n))
        return arrays_from_vec(self.template, new), MetricRecord({"train_loss": loss})
