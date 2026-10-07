"""Phase 1 check: CNN-LSTM (with Opacus DPLSTM) trains with and without DP-SGD on CPU and GPU.

Prints seconds per training step for each case:
    python scripts/check_gpu_dp.py
The numbers feed the Phase 8 compute budget (runs x time per run).
"""
import time
import warnings

import torch
import torch.nn as nn
from opacus import PrivacyEngine
from opacus.layers import DPLSTM
from opacus.validators import ModuleValidator
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")
N_FEATURES, N_CLASSES, N_ROWS, BATCH, STEPS = 39, 10, 256 * 40, 256, 30


class CNNLSTM(nn.Module):
    """Same architecture as the plan (Figure 8.1)."""

    def __init__(self, n_features, n_classes):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, 3, padding=1), nn.ReLU(),
            nn.Conv1d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool1d(2))
        self.lstm = DPLSTM(64, 64, batch_first=True)
        self.drop = nn.Dropout(0.3)
        self.head = nn.Linear(64, n_classes)

    def forward(self, x):
        h = self.conv(x).permute(0, 2, 1)
        out, _ = self.lstm(h)
        return self.head(self.drop(out[:, -1]))


def time_steps(device, dp):
    torch.manual_seed(0)
    x = torch.randn(N_ROWS, 1, N_FEATURES)
    y = torch.randint(0, N_CLASSES, (N_ROWS,))
    loader = DataLoader(TensorDataset(x, y), batch_size=BATCH, shuffle=True, num_workers=0)
    model = CNNLSTM(N_FEATURES, N_CLASSES).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.0)
    if dp:
        model, opt, loader = PrivacyEngine(accountant="rdp").make_private(
            module=model, optimizer=opt, data_loader=loader, noise_multiplier=1.0, max_grad_norm=1.0)
    loss_fn = nn.CrossEntropyLoss()
    done, t0 = 0, None
    while done < STEPS:
        for xb, yb in loader:
            if len(yb) == 0:
                continue
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()
            done += 1
            if done == 5:                       # skip warm-up steps
                if device == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
            if done >= STEPS:
                break
    if device == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / (STEPS - 5)


def main():
    model = CNNLSTM(N_FEATURES, N_CLASSES)
    errors = ModuleValidator.validate(model, strict=False)
    print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("Opacus validator:", "OK" if not errors else errors)
    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
    print(f"\n{'device':6s} {'DP':5s} {'sec/step':>9s} {'est. min per 40-round run*':>28s}")
    for dev in devices:
        for dp in (False, True):
            s = time_steps(dev, dp)
            # ~500 steps per round across all clients (about 128k capped training rows / 256)
            print(f"{dev:6s} {str(dp):5s} {s:9.4f} {s * 500 * 40 / 60:28.1f}")
    print("\n* rough: 500 steps/round x 40 rounds, training only (no evaluation, no overhead).")


if __name__ == "__main__":
    main()
