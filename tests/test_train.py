"""Training loop on a tiny synthetic problem (CPU, a few seconds)."""
import numpy as np
import pytest
import torch

from fedguard.config import load_config
from fedguard.data import ProcessedData
from fedguard.model import build_model
from fedguard.train import class_weights, fit_centralised, make_optimizer
from fedguard.utils import set_seed


def toy_data(seed=0, n=1200, f=12, k=3):
    rng = np.random.default_rng(seed)
    centres = rng.normal(scale=2.5, size=(k, f))
    def make(m):
        y = rng.integers(0, k, m)
        return (centres[y] + rng.normal(size=(m, f))).astype(np.float32), y
    (a, b), (c, d), (e, g) = make(n), make(300), make(300)
    return ProcessedData(a, b, c, d, e, g, classes=["Benign", "A", "B"], features=[f"f{i}" for i in range(f)],
                         fingerprint="toy")


def cfg(seed=0, **train):
    return load_config(overrides={"seed": seed, "train": {"max_epochs": 25, "patience": 5, "device": "cpu",
                                                          "batch_size": 64, **train}})


def run(seed, **train):
    set_seed(seed)
    data = toy_data()
    model = build_model(data.n_features, data.n_classes)
    return fit_centralised(model, data, cfg(seed, **train), torch.device("cpu"), log=lambda *a: None)


def test_learns_an_easy_problem():
    _, history, info = run(0)
    assert info["best_val_macro_f1"] > 0.9
    assert history[-1]["train_loss"] < history[0]["train_loss"]


def test_same_seed_gives_identical_training():
    _, h1, _ = run(1)
    _, h2, _ = run(1)
    assert [r["train_loss"] for r in h1] == [r["train_loss"] for r in h2]


def test_early_stopping_restores_best_epoch():
    model, history, info = run(2, patience=3)
    assert info["epochs_run"] <= 25
    assert info["best_val_macro_f1"] == max(h["val_macro_f1"] for h in history)


def test_optimizer_is_plain_sgd():
    opt = make_optimizer(build_model(12, 3), lr=0.05)
    g = opt.param_groups[0]
    assert isinstance(opt, torch.optim.SGD) and g["momentum"] == 0 and g["weight_decay"] == 0


@pytest.mark.parametrize("mode", ["sqrt", "balanced"])
def test_class_weights_favour_rare_classes(mode):
    y = np.array([0] * 900 + [1] * 90 + [2] * 10)
    w = class_weights(y, 3, mode)
    assert w[2] > w[1] > w[0] and float(w.mean()) == pytest.approx(1.0)
    assert class_weights(y, 3, "none") is None


def test_batched_loader_gives_the_same_batches_as_row_by_row():
    """make_loader's one-shot batch gathering (BatchTensorDataset) must yield exactly the batches of the plain
    row-by-row TensorDataset loader, with shuffling and with Opacus' Poisson sampling."""
    import warnings
    from torch.utils.data import DataLoader, TensorDataset
    from opacus import PrivacyEngine
    from fedguard.train import make_loader
    rng = np.random.default_rng(0)
    X = rng.normal(size=(700, 6)).astype(np.float32)
    y = rng.integers(0, 3, size=700)

    def old(seed):
        ds = TensorDataset(torch.as_tensor(X), torch.as_tensor(y, dtype=torch.long))
        return DataLoader(ds, batch_size=64, shuffle=True, generator=torch.Generator().manual_seed(seed))

    def private(loader):
        m = torch.nn.Linear(6, 3)
        opt = torch.optim.SGD(m.parameters(), lr=0.1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return PrivacyEngine().make_private(module=m, optimizer=opt, data_loader=loader, noise_multiplier=1.0,
                                                max_grad_norm=1.0, poisson_sampling=True)[2]

    for wrap in (lambda l: l, private):
        a = list(wrap(old(5)))
        b = list(wrap(make_loader(X, y, 64, shuffle=True, seed=5)))
        assert len(a) == len(b)
        for (xa, ya), (xb, yb) in zip(a, b):
            assert torch.equal(xa, xb) and torch.equal(ya, yb)
