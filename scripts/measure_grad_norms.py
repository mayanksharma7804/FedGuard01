"""Phase 5: per-example gradient norms of the CNN-LSTM on the real training data, to choose the DP clipping
norm C (common rule: C near the median per-example norm, so clipping neither destroys nor ignores the signal).

    python scripts/measure_grad_norms.py

Trains the B0 model (no DP) and, at a few checkpoints, measures the per-example gradient norm of 4096
random training rows with Opacus' GradSampleModule. Writes results/dp/grad_norms.csv.
"""
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from opacus import GradSampleModule

from fedguard.data import load_processed
from fedguard.model import build_model
from fedguard.train import get_device, make_loader, make_optimizer, train_epoch
from fedguard.utils import REPO_ROOT, set_seed

warnings.filterwarnings("ignore")
CHECKPOINTS = [0, 1, 3, 10, 30]          # epochs of plain training (B0 settings: SGD lr 0.2, batch 256)


def per_example_norms(model, X, y, device):
    gsm = GradSampleModule(model)
    gsm.train()
    norms = []
    for i in range(0, len(X), 512):
        xb = torch.as_tensor(X[i:i + 512], device=device)
        yb = torch.as_tensor(y[i:i + 512], device=device)
        gsm.zero_grad(set_to_none=True)
        nn.CrossEntropyLoss()(gsm(xb), yb).backward()       # MEAN loss: GradSampleModule assumes loss_reduction="mean"
                                                            # (with "sum" every norm comes out batch-size times too big)
        sq = sum(p.grad_sample.reshape(len(yb), -1).pow(2).sum(1) for p in gsm.parameters() if p.requires_grad)
        norms.append(sq.sqrt().detach().cpu().numpy())
    gsm._close()
    return np.concatenate(norms)


def main():
    set_seed(42)
    device = get_device()
    data = load_processed("nf_unsw")
    rng = np.random.default_rng(0)
    pick = rng.choice(len(data.y_tr), 4096, replace=False)
    Xs, ys = data.X_tr[pick], data.y_tr[pick]
    model = build_model(data.n_features, data.n_classes).to(device)
    loader = make_loader(data.X_tr, data.y_tr, 256, seed=42)
    opt = make_optimizer(model, 0.2)
    rows, epoch = [], 0
    for ck in CHECKPOINTS:
        while epoch < ck:
            train_epoch(model, loader, opt, nn.CrossEntropyLoss(), device); epoch += 1
        n = per_example_norms(model, Xs, ys, device)
        row = {"epoch": ck, "mean": n.mean(), "p10": np.quantile(n, 0.1), "p25": np.quantile(n, 0.25),
               "median": np.median(n), "p75": np.quantile(n, 0.75), "p90": np.quantile(n, 0.9), "max": n.max()}
        for c in (0.5, 1.0, 2.0, 5.0):
            row[f"share_clipped_C{c:g}"] = float((n > c).mean())
        rows.append(row)
        print({k: round(float(v), 3) for k, v in row.items()}, flush=True)
    out = REPO_ROOT / "results" / "dp"; out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "grad_norms.csv", index=False, float_format="%.4f")
    print("-> results/dp/grad_norms.csv")


if __name__ == "__main__":
    main()
