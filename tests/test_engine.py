"""Federated engine (fl_sim.py), FedAvg and the experiment runner, on a tiny synthetic problem (CPU)."""
import copy
import importlib.util

import numpy as np
import pytest
import torch

from fedguard.aggregators import fedavg, get_aggregator
from fedguard.config import load_config
from fedguard.fl_sim import FLSimulation, flat, round_seed, set_flat
from fedguard.partition import dirichlet_partition
from fedguard.train import make_loader, make_optimizer, train_epoch
from tests.test_train import toy_data

spec = importlib.util.spec_from_file_location("run_experiments", "scripts/run_experiments.py")
runner = importlib.util.module_from_spec(spec); spec.loader.exec_module(runner)


def fl_cfg(seed=0, **fl):
    return load_config(overrides={"seed": seed,
                                  "train": {"lr": 0.2, "batch_size": 64, "device": "cpu"},
                                  "fl": {"max_rounds": 6, "patience": 3, "min_rounds": 1, **fl}})


def sim(seed=0, n_clients=3, alpha=0.5, **fl):
    data = toy_data()
    parts = dirichlet_partition(data.y_tr, n_clients, alpha, min_size=50, seed=seed)
    return FLSimulation(fl_cfg(seed, **fl), data, parts, torch.device("cpu"), log=lambda *a: None)


def test_fedavg_is_sample_weighted_mean():
    X = np.array([[1.0, 0.0], [3.0, 2.0]])
    z, w, _ = fedavg(X, np.array([1, 3]))
    assert np.allclose(z, [2.5, 1.5]) and np.allclose(w, [0.25, 0.75])
    with pytest.raises(ValueError):
        get_aggregator("nope")


def test_one_client_one_round_equals_plain_local_training():
    s = sim(seed=5, n_clients=1, max_rounds=1)
    g0 = flat(s.global_model)
    expected = copy.deepcopy(s.global_model)                        # replicate the client by hand
    rs = round_seed(5, 1, 0)
    torch.manual_seed(rs)
    c = s.clients[0]
    loader = make_loader(s.data.X_tr[c.idx], s.data.y_tr[c.idx], 64, shuffle=True, seed=rs)
    train_epoch(expected, loader, make_optimizer(expected, 0.2), s.loss_fn, torch.device("cpu"))
    s.run()
    assert not torch.equal(flat(s.global_model), g0)                # the model moved ...
    assert torch.allclose(flat(s.global_model), flat(expected), atol=1e-6)   # ... exactly like local training


def test_local_training_never_modifies_the_global_vector():
    s = sim(seed=4, n_clients=3)
    g = flat(s.global_model)
    g_before = g.clone()
    out = s.local_train(s.clients[0], g, 1)
    assert torch.equal(g, g_before)                                  # regression test for Findings F24
    assert out["delta"].abs().max() > 0                              # the client really moved


def test_round_one_is_the_weighted_average_of_independent_clients():
    """With 3 clients, FedAvg's global model must equal the sample-weighted mean of 3 local models that
    each started from the SAME global weights (the bug in F24 trained them one after another)."""
    s = sim(seed=6, n_clients=3, max_rounds=1)
    g = flat(s.global_model)
    locals_, ns = [], []
    for c in s.clients:                                              # independent replicas, by hand
        m = copy.deepcopy(s.global_model)
        rs = round_seed(6, 1, c.cid)
        torch.manual_seed(rs)
        loader = make_loader(s.data.X_tr[c.idx], s.data.y_tr[c.idx], 64, shuffle=True, seed=rs)
        train_epoch(m, loader, make_optimizer(m, 0.2), s.loss_fn, torch.device("cpu"))
        locals_.append(flat(m)); ns.append(c.n)
    w = torch.tensor(ns, dtype=torch.float32) / sum(ns)
    expected = sum(wi * li for wi, li in zip(w, locals_))
    s.run()
    assert torch.allclose(flat(s.global_model), expected, atol=1e-5)
    assert not torch.allclose(locals_[0], locals_[1])                # clients really differ


def test_same_seed_same_federated_run():
    _, r1, w1, _ = sim(seed=3).run()
    _, r2, w2, _ = sim(seed=3).run()
    assert [x["val_macro_f1"] for x in r1] == [x["val_macro_f1"] for x in r2]
    assert [x["weight"] for x in w1] == [x["weight"] for x in w2]


def test_federated_training_learns_and_logs_weights():
    model, rounds, wlog, info = sim(seed=1, n_clients=4, alpha=100, max_rounds=8, patience=8).run()
    assert info["best_val_macro_f1"] > 0.8
    per_round = {}
    for row in wlog:
        per_round.setdefault(row["round"], []).append(row["weight"])
    assert all(abs(sum(v) - 1) < 1e-9 and len(v) == 4 for v in per_round.values())


def test_set_flat_roundtrip():
    s = sim(seed=0)
    v = flat(s.global_model) + 1.0
    set_flat(s.global_model, v)
    assert torch.equal(flat(s.global_model), v)


def test_sweep_expansion_names_and_shards(tmp_path):
    sweep = tmp_path / "s.yaml"
    sweep.write_text("base: configs/b1_fedavg.yaml\ngrid:\n  fl.alpha: [0.1, 0.5, 100]\n  seed: [42, 43]\n")
    cfgs = runner.expand_sweep(sweep)
    names = [runner.run_name(c) for c in cfgs]
    assert len(cfgs) == 6 and len(set(names)) == 6
    assert {c["fl"]["alpha"] for c in cfgs} == {0.1, 0.5, 100}
    assert names[0].startswith("b1_fedavg_a0.1_k5_e5_s42_")        # E = 5 since Findings F27
    assert runner.run_name(cfgs[0]) == names[0]                     # stable
    assert len(cfgs[0::2]) + len(cfgs[1::2]) == 6                    # two shards cover everything
    dp_cfg = runner.load_config("configs/b2_dpfedavg.yaml")
    assert "_eps3_c2_s42_" in runner.run_name(dp_cfg)               # DP runs carry epsilon and clip


@pytest.mark.parametrize("dp", [None, {"epsilon": 3.0, "clip": 1.0}], ids=["noDP", "DP"])
def test_pause_and_resume_from_checkpoint_gives_the_same_run(tmp_path, dp):
    from fedguard.fl_sim import PauseRequested

    def sim(**kw):
        s = globals()["sim"](**kw)
        if dp:
            s2 = FLSimulation({**s.cfg, "dp": dp}, s.data, [c.idx for c in s.clients], torch.device("cpu"),
                              log=lambda *a: None)
            return s2
        return s
    full = sim(seed=7, max_rounds=5, min_rounds=5, patience=10)
    v_full = flat(full.run()[0])
    ck = tmp_path / "checkpoint.pt"
    calls = {"n": 0}
    def pause_after_two():
        calls["n"] += 1
        return calls["n"] == 2
    first = sim(seed=7, max_rounds=5, min_rounds=5, patience=10)
    with pytest.raises(PauseRequested):
        first.run(checkpoint=ck, should_pause=pause_after_two)
    assert ck.exists()
    second = sim(seed=7, max_rounds=5, min_rounds=5, patience=10)      # a fresh process after the pause
    model, rounds, wlog, info = second.run(checkpoint=ck)
    assert [r["round"] for r in rounds] == [1, 2, 3, 4, 5] and info["rounds_run"] == 5
    assert torch.equal(flat(model), v_full)                             # identical to the uninterrupted run
