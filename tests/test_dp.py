"""DP-SGD clients, privacy accounting and the declared noise radius (plan Phase 5, checkpoint tests)."""
import math

import torch
import torch.nn as nn

from fedguard import dp
from fedguard.fl_sim import FLSimulation, flat
from fedguard.model import build_model
from fedguard.partition import dirichlet_partition
from fedguard.train import make_loader, make_optimizer, train_epoch
from tests.test_engine import fl_cfg
from tests.test_train import toy_data

META = {"lr": 0.2, "sigma": 1.0, "C": 1.0, "T": 100, "B": 256}


def test_noise_radius_zero_without_noise():
    assert dp.noise_radius({**META, "sigma": 0.0}, 40_000) == 0.0


def test_noise_radius_linear_in_sigma_and_sqrt_t():
    r = dp.noise_radius(META, 40_000)
    assert math.isclose(r, 0.2 * 1.0 * 1.0 * math.sqrt(100 * 40_000) / 256)
    assert math.isclose(dp.noise_radius({**META, "sigma": 3.0}, 40_000), 3 * r)
    assert math.isclose(dp.noise_radius({**META, "T": 400}, 40_000), 2 * r)      # 4x steps -> 2x radius
    assert math.isclose(dp.noise_radius({**META, "B": 512}, 40_000), r / 2)


def test_sigma_calibration_meets_the_budget():
    q, steps, delta = 256 / 15_000, 58 * 40, 1 / 15_000
    s3, s1 = dp.calibrate_sigma(3.0, delta, q, steps), dp.calibrate_sigma(1.0, delta, q, steps)
    assert s1 > s3 > 0                                                   # more privacy -> more noise
    eps = dp.epsilon_spent(s3, q, steps, delta)
    assert 2.9 < eps <= 3.0 + 1e-6                                       # spends the budget, not more
    assert dp.epsilon_spent(s3, q, steps // 2, delta) < eps              # stopping early spends less
    assert dp.epsilon_spent(0.0, q, steps, delta) == math.inf


def test_privacy_plan_per_client():
    cfg = fl_cfg(local_epochs=2, max_rounds=10)
    cfg["dp"] = {"epsilon": 3.0, "clip": 1.0}
    small, big = dp.client_privacy_plan(300, cfg), dp.client_privacy_plan(3000, cfg)
    assert small["delta"] == 1 / 300 and big["T"] == 2 * math.ceil(3000 / 64)      # Opacus: ceil(n/B) per epoch
    assert big["B"] == int(3000 / math.ceil(3000 / 64)) and big["q"] == 1 / math.ceil(3000 / 64)
    assert small["sigma"] > big["sigma"] > 0                              # small clients need more noise
    assert dp.client_privacy_plan(300, fl_cfg())["sigma"] == 0.0          # no DP -> no noise


def test_plan_matches_what_opacus_actually_does():
    """Regression test for Findings F34: q, batches per epoch and expected batch size must be Opacus' own."""
    for n, B in [(3649, 256), (1856, 256), (93 * 256 - 10, 256), (300, 64), (128, 128)]:
        q, per_epoch, b_exp = dp.poisson_params(n, B)
        X = torch.zeros(n, 1, 4); y = torch.zeros(n, dtype=torch.long)
        loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X, y), batch_size=B)
        model = nn.Sequential(nn.Flatten(), nn.Linear(4, 2))
        _, opt, ld = dp.make_private(model, make_optimizer(model, 0.1), loader, 1.0, 1.0, 0, torch.device("cpu"))
        assert len(ld) == per_epoch and ld.sample_rate == q and opt.expected_batch_size == b_exp, n


def _one_dp_step(sigma, seed=3, n=128, B=128, lr=0.2, clip=1.0):
    data = toy_data()
    torch.manual_seed(0)
    model = build_model(data.n_features, data.n_classes)
    g = flat(model)
    torch.manual_seed(seed)                                               # dropout masks
    loader = make_loader(data.X_tr[:n], data.y_tr[:n], B, shuffle=True, seed=seed)
    m, opt, ld = dp.make_private(model, make_optimizer(model, lr), loader, sigma, clip, noise_seed=seed + 1,
                                 device=torch.device("cpu"))
    train_epoch(m, ld, opt, nn.CrossEntropyLoss(), torch.device("cpu"))  # one epoch of Poisson steps
    return flat(model) - g, g.numel()


def test_sigma_zero_is_deterministic_and_noise_is_isolated():
    clean_a, d = _one_dp_step(0.0)
    clean_b, _ = _one_dp_step(0.0)
    assert torch.equal(clean_a, clean_b)                                  # same batches, same dropout
    noisy, _ = _one_dp_step(1.5)
    measured = float((noisy - clean_a).norm())
    predicted = dp.noise_radius({"lr": 0.2, "sigma": 1.5, "C": 1.0, "T": 1, "B": 128}, d)
    assert abs(measured / predicted - 1) < 0.05                           # one step: formula is exact


def test_noise_radius_with_several_poisson_steps():
    """n = 300, B = 64 -> Opacus takes 5 steps with expected batch 60 (not 64, not 4 steps): the declared
    radius must use exactly those values (E0 found a 9-19% gap before the fix, Findings F34)."""
    clean, d = _one_dp_step(0.0, n=300, B=64, lr=0.01)
    noisy, _ = _one_dp_step(2.0, n=300, B=64, lr=0.01)
    q, steps, b_exp = dp.poisson_params(300, 64)
    predicted = dp.noise_radius({"lr": 0.01, "sigma": 2.0, "C": 1.0, "T": steps, "B": b_exp}, d)
    assert (steps, b_exp) == (5, 60)
    assert abs(float((noisy - clean).norm()) / predicted - 1) < 0.03


def test_dp_federated_run_is_deterministic_and_reports_epsilon():
    def go():
        data = toy_data()
        parts = dirichlet_partition(data.y_tr, 3, 0.5, min_size=50, seed=1)
        cfg = fl_cfg(1, max_rounds=3, min_rounds=3, patience=5)
        cfg["dp"] = {"epsilon": 8.0, "clip": 1.0}
        s = FLSimulation(cfg, data, parts, torch.device("cpu"), log=lambda *a: None)
        model, rounds, wlog, info = s.run()
        return flat(model), rounds, wlog, info
    v1, rounds, wlog, info = go()
    v2, *_ = go()
    assert torch.equal(v1, v2)
    assert 0 < info["epsilon_spent_max"] <= 8.0 + 1e-6
    assert all(r > 0 for r in info["declared_radius_per_client"])
    assert all(w["declared_radius"] > 0 for w in wlog)


def test_flower_client_dp_round_matches_engine():
    import copy as _copy
    from fedguard.flower_strategy import client_round
    data = toy_data()
    parts = dirichlet_partition(data.y_tr, 3, 0.5, min_size=50, seed=2)
    cfg = fl_cfg(2, max_rounds=4, local_epochs=2)
    cfg["dp"] = {"epsilon": 3.0, "clip": 1.0}
    s = FLSimulation(cfg, data, parts, torch.device("cpu"), log=lambda *a: None)
    g = flat(s.global_model)
    c = s.clients[1]
    engine = s.local_train(c, g, rnd=3)
    model = _copy.deepcopy(s.global_model)
    _, meta = client_round(model, data.X_tr[c.idx], data.y_tr[c.idx], cfg, c.cid, 3, s.loss_fn, torch.device("cpu"))
    assert torch.allclose(flat(model) - g, engine["delta"], atol=1e-6)     # same batches, same noise
    assert meta == engine["meta"]


def test_heterogeneous_privacy_per_client():
    cfg = fl_cfg(max_rounds=10)
    cfg["dp"] = {"epsilon": [1.0, 8.0], "clip": 1.0}
    assert dp.is_dp(cfg)
    p0, p1, p2 = (dp.client_privacy_plan(2000, cfg, cid) for cid in range(3))
    assert (p0["target_epsilon"], p1["target_epsilon"], p2["target_epsilon"]) == (1.0, 8.0, 1.0)
    assert p0["sigma"] > p1["sigma"] and p0["sigma"] == p2["sigma"]
    assert dp.noise_radius(p0, 40_000) > 2 * dp.noise_radius(p1, 40_000)      # the private client is much noisier


def test_radius_hardly_depends_on_client_size_at_one_epsilon():
    """Findings F36: at a shared epsilon, sigma grows for small clients exactly enough to keep r ~ constant."""
    cfg = fl_cfg(max_rounds=40, local_epochs=5)
    cfg["train"]["batch_size"] = 256
    cfg["dp"] = {"epsilon": 3.0, "clip": 2.0}
    r = [dp.noise_radius(dp.client_privacy_plan(n, cfg), 40_266) for n in (1200, 3300, 11000, 18500)]
    assert max(r) / min(r) < 1.25
