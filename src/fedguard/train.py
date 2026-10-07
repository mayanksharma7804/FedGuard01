"""Training and evaluation shared by every baseline and every federated client.

Rules from the plan (section 8.1): plain SGD (momentum 0, no weight decay), CrossEntropy, batch 256,
early stopping on VALIDATION macro-F1, seeds fixed, DataLoader num_workers=0 on Windows.
"""
import copy
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from fedguard.metrics import classification_metrics


def get_device(pref: str = "auto") -> torch.device:
    if pref == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(pref)


def make_loader(X, y, batch_size, shuffle=True, seed=0) -> DataLoader:
    ds = TensorDataset(torch.as_tensor(X, dtype=torch.float32), torch.as_tensor(y, dtype=torch.long))
    gen = torch.Generator().manual_seed(seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, generator=gen, num_workers=0)


def class_weights(y, n_classes, mode="none"):
    """Loss weights per class. 'balanced' = n / (K * count); 'sqrt' = its square root (milder)."""
    if mode == "none":
        return None
    counts = np.bincount(y, minlength=n_classes).astype(float)
    w = len(y) / (n_classes * np.maximum(counts, 1))
    if mode == "sqrt":
        w = np.sqrt(w)
    elif mode != "balanced":
        raise ValueError(f"unknown class_weight '{mode}'")
    w = w / w[counts > 0].mean()                       # keep the average weight at 1
    return torch.tensor(w, dtype=torch.float32)


def make_optimizer(model, lr):
    return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.0, weight_decay=0.0)


def train_epoch(model, loader, optimizer, loss_fn, device) -> float:
    model.train()
    total, n = 0.0, 0
    for xb, yb in loader:
        if len(yb) == 0:                               # Opacus Poisson sampling can give empty batches
            continue
        xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(xb), yb)
        loss.backward()
        optimizer.step()
        total += loss.item() * len(yb); n += len(yb)
    return total / max(n, 1)


@torch.no_grad()
def predict(model, X, device, batch=4096) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(X), batch):
        xb = torch.as_tensor(X[i:i + batch], dtype=torch.float32, device=device)
        out.append(model(xb).argmax(1).cpu().numpy())
    return np.concatenate(out) if out else np.empty(0, dtype=np.int64)


def evaluate(model, X, y, classes, device, batch=4096) -> dict:
    return classification_metrics(y, predict(model, X, device, batch), classes)


def fit_centralised(model, data, cfg, device, log=print):
    """Train on ALL training data (baseline B0). Returns (best model, history list, info dict)."""
    t = cfg["train"]
    model.to(device)
    loader = make_loader(data.X_tr, data.y_tr, t["batch_size"], shuffle=True, seed=cfg["seed"])
    w = class_weights(data.y_tr, data.n_classes, t["class_weight"])
    loss_fn = nn.CrossEntropyLoss(weight=None if w is None else w.to(device))
    opt = make_optimizer(model, t["lr"])

    best_f1, best_state, best_epoch, bad, history = -1.0, None, 0, 0, []
    t_start = time.perf_counter()
    for epoch in range(1, t["max_epochs"] + 1):
        t0 = time.perf_counter()
        loss = train_epoch(model, loader, opt, loss_fn, device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        train_s = time.perf_counter() - t0
        val = evaluate(model, data.X_va, data.y_va, data.classes, device, t["eval_batch"])
        history.append({"epoch": epoch, "train_loss": loss, "val_macro_f1": val["macro_f1"],
                        "val_accuracy": val["accuracy"], "val_benign_fpr": val["benign_fpr"],
                        "epoch_train_seconds": train_s})
        improved = val["macro_f1"] > best_f1 + 1e-4
        if improved:
            best_f1, best_epoch, bad = val["macro_f1"], epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad += 1
        log(f"  epoch {epoch:3d}/{t['max_epochs']}  loss {loss:.4f}  val macro-F1 {val['macro_f1']:.4f}"
            f"  best {best_f1:.4f} (epoch {best_epoch})  patience {bad}/{t['patience']}"
            f"{'  *new best' if improved else ''}  ({train_s:.1f}s)")
        if bad >= t["patience"] and epoch >= t.get("min_epochs", 0):
            log(f"  early stop at epoch {epoch} (best epoch {best_epoch}, val macro-F1 {best_f1:.4f})")
            break
    model.load_state_dict(best_state)
    info = {"best_epoch": best_epoch, "epochs_run": len(history), "best_val_macro_f1": best_f1,
            "train_seconds_total": time.perf_counter() - t_start,
            "seconds_per_epoch_mean": float(np.mean([h["epoch_train_seconds"] for h in history]))}
    return model, history, info
