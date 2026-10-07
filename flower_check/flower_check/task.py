"""Tiny model + synthetic data shared by the client and server apps."""
import numpy as np
import torch
import torch.nn as nn

N_FEATURES, N_CLASSES = 20, 3


def make_model():
    return nn.Sequential(nn.Linear(N_FEATURES, 32), nn.ReLU(), nn.Linear(32, N_CLASSES))


def make_data(partition_id, n=600, seed=0):
    """Same class centres for every client, different samples per client (seeded by partition id)."""
    centres = np.random.default_rng(seed).normal(scale=2.0, size=(N_CLASSES, N_FEATURES))
    rng = np.random.default_rng(seed + 1 + partition_id)
    y = rng.integers(0, N_CLASSES, size=n)
    x = centres[y] + rng.normal(size=(n, N_FEATURES))
    return torch.tensor(x, dtype=torch.float32), torch.tensor(y)


def train(model, x, y, epochs, lr):
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    for _ in range(epochs):
        for i in range(0, len(x), 64):
            opt.zero_grad()
            loss = loss_fn(model(x[i:i + 64]), y[i:i + 64])
            loss.backward()
            opt.step()
    return float(loss)


def accuracy(model, x, y):
    model.eval()
    with torch.no_grad():
        return float((model(x).argmax(1) == y).float().mean())
