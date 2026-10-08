"""Attacks (plan Phase 6) and their wiring into the federated engine."""
import numpy as np
import pytest
import torch

from fedguard import attacks
from fedguard.fl_sim import FLSimulation, flat
from fedguard.partition import dirichlet_partition
from tests.test_engine import fl_cfg
from tests.test_train import toy_data


def test_label_flip_makes_every_attack_benign():
    y = np.array([0, 1, 2, 2, 0, 1])
    assert attacks.label_flip(y).tolist() == [0] * 6
    assert y.tolist() == [0, 1, 2, 2, 0, 1]                           # input not modified


def test_choose_attackers_count_and_determinism():
    m = attacks.choose_attackers(10, 0.3, seed=42)
    assert m.sum() == 3 and np.array_equal(m, attacks.choose_attackers(10, 0.3, seed=42))
    assert attacks.choose_attackers(10, 0.0, seed=1).sum() == 0
    with pytest.raises(ValueError):
        attacks.choose_attackers(10, 1.0, seed=1)
    with pytest.raises(ValueError):
        attacks.validate({"type": "backdoor"})


def make_sim(attack=None, aggregator="fedavg", seed=3, n_clients=5, dp=None):
    data = toy_data()
    parts = dirichlet_partition(data.y_tr, n_clients, 0.5, min_size=50, seed=seed)
    cfg = fl_cfg(seed, max_rounds=4, min_rounds=4, patience=10, aggregator={"name": aggregator})
    if attack:
        cfg["attack"] = attack
    if dp:
        cfg["dp"] = dp
    return FLSimulation(cfg, data, parts, torch.device("cpu"), log=lambda *a: None)


@pytest.mark.parametrize("dp", [None, {"epsilon": 3.0, "clip": 1.0}], ids=["noDP", "DP"])
def test_model_poisoning_transforms_the_honest_update(dp):
    honest = make_sim(dp=dp)
    sf = make_sim({"type": "sign_flip", "fraction": 0.4, "scale": 2.0}, dp=dp)
    ga = make_sim({"type": "gaussian", "fraction": 0.4}, dp=dp)
    att = [c.cid for c in sf.clients if c.is_attacker]
    assert len(att) == 2 and att == [c.cid for c in ga.clients if c.is_attacker]
    g = flat(honest.global_model)
    c = att[0]
    h = honest.local_train(honest.clients[c], g, 1)
    s = sf.local_train(sf.clients[c], g, 1)
    q = ga.local_train(ga.clients[c], g, 1)
    assert torch.allclose(s["delta"], -2.0 * h["delta"], atol=1e-6)     # sign flip of the same update
    assert abs(float(q["delta"].norm()) / float(h["delta"].norm()) - 10.0) < 0.5   # default gaussian scale 10
    assert s["meta"] == h["meta"] == q["meta"]                          # declared metadata looks honest
    honest_client = next(k for k in range(5) if k not in att)
    assert torch.equal(sf.local_train(sf.clients[honest_client], g, 1)["delta"],
                       honest.local_train(honest.clients[honest_client], g, 1)["delta"])


def test_label_flip_attacker_trains_on_benign_labels_only():
    lf = make_sim({"type": "label_flip", "fraction": 0.4})
    honest = make_sim()
    g = flat(honest.global_model)
    c = next(c.cid for c in lf.clients if c.is_attacker)
    assert not torch.allclose(lf.local_train(lf.clients[c], g, 1)["delta"],
                              honest.local_train(honest.clients[c], g, 1)["delta"])


def test_robust_aggregator_downweights_sign_flip_attackers_and_logs_shares():
    s = make_sim({"type": "sign_flip", "fraction": 0.4, "scale": 3.0}, aggregator="autogm")
    _, rounds, wlog, info = s.run()
    assert info["attackers"] and 0 <= info["attacker_share_mean"] < 0.2
    assert all("hrr" in r and "attacker_share" in r for r in rounds)
    fedavg = make_sim({"type": "sign_flip", "fraction": 0.4, "scale": 3.0})
    _, _, _, info_avg = fedavg.run()
    assert info_avg["attacker_share_mean"] > info["attacker_share_mean"]
