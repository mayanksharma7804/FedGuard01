"""AutoGM and FedGuard aggregators on synthetic updates (plan section 12.4). Every FedGuard test runs in both
modes (linear / quadrature) and with the R4 noise-length tolerance off and on."""
import numpy as np
import pytest

from fedguard.aggregators import autogm, fedguard, get_aggregator, project_simplex, too_clean, weighted_geomed

D = 4000
VARIANTS = [{"mode": m, "noise_tol_z": z} for m in ("linear", "quadrature") for z in (None, 3.0)]
IDS = [f"{v['mode']}-R4{'on' if v['noise_tol_z'] else 'off'}" for v in VARIANTS]


def direction(rng):
    mu = rng.normal(size=D)
    return mu / np.linalg.norm(mu)


def noisy(rng, mu, radius):
    """mu + Gaussian noise whose length is ~radius (what an honest DP client sends)."""
    return mu + rng.normal(scale=radius / np.sqrt(D), size=D)


def test_project_simplex():
    a = project_simplex(np.array([0.3, -2.0, 0.9, 0.1]))
    assert a.min() >= 0 and abs(a.sum() - 1) < 1e-12 and a[1] == 0
    assert np.allclose(project_simplex(np.zeros(4)), 0.25)


def test_weighted_geomed_resists_one_outlier():
    X = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [100.0, 100.0]])
    z = weighted_geomed(X, np.ones(5))
    assert np.linalg.norm(z - [0.5, 0.5]) < 1.0                       # the mean would be ~(20.4, 20.4)


@pytest.mark.parametrize("p", VARIANTS, ids=IDS)
def test_identical_updates_get_equal_weights(p):
    X = np.tile(direction(np.random.default_rng(0)), (6, 1))
    z, a, _ = fedguard(X, np.full(6, 100), np.zeros(6), p)
    assert np.allclose(a, 1 / 6) and np.allclose(z, X[0])
    _, a2, _ = autogm(X)
    assert np.allclose(a2, 1 / 6)


@pytest.mark.parametrize("p", VARIANTS, ids=IDS)
def test_far_attacker_gets_zero_weight(p):
    rng = np.random.default_rng(1)
    mu = direction(rng)
    X = np.stack([noisy(rng, mu, 0.1) for _ in range(9)] + [-5 * mu])
    z, a, _ = fedguard(X, None, np.full(10, 0.1), p)
    assert a[-1] < 1e-6                                              # verified to pass on the reference code
    honest_gm = weighted_geomed(X[:9], np.ones(9))
    assert np.linalg.norm(z - honest_gm) < 0.05 * np.linalg.norm(honest_gm)


def _xfail_quadrature_without_r4(p):
    if p["mode"] == "quadrature" and p["noise_tol_z"] is None:
        return pytest.param(p, marks=pytest.mark.xfail(strict=True, reason=(
            "known limitation (Findings, synthetic logic check): without R4, quadrature punishes the natural "
            "noise-length wobble of the most private client in 30-50% of seeds")))
    return p


@pytest.mark.parametrize("p", [_xfail_quadrature_without_r4(v) for v in VARIANTS], ids=IDS)
def test_noisy_honest_client_is_not_rejected_heterogeneous_privacy(p):
    """Honest clients with DIFFERENT privacy levels (radii 0.2 ... 2.0), each declaring its own noise, plus
    normal non-IID spread (length 0.3 vs a signal of length 1): the most private one must keep weight.
    AutoGM, which ignores r, rejects it (the conflict, paper O2)."""
    r = np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0])
    for seed in range(10):
        rng = np.random.default_rng(seed)
        mu = direction(rng)
        X = np.stack([noisy(rng, noisy(rng, mu, 0.3), ri) for ri in r])
        _, a, _ = fedguard(X, None, r, p)
        assert a[-1] > 0.5 / len(r), (seed, a)                         # not rejected (HRR definition)
        _, a_gm, _ = autogm(X)
        assert a_gm[-1] < 0.5 / len(r)                                # AutoGM wrongly rejects it


@pytest.mark.parametrize("mode", ["linear", "quadrature"])
def test_quiet_majority_needs_r4(mode):
    """9 quiet honest clients + 1 very private honest client (plan 12.3, Figure 12.3): the allowance s
    collapses, and only R4 stops the private client's natural noise-length wobble from being punished.
    The far attacker must still get zero weight with R4 on."""
    rejected_off, rejected_on = 0, 0
    for seed in range(40):
        rng = np.random.default_rng(100 + seed)
        mu = direction(rng)
        r = np.array([0.05] * 9 + [3.0])
        X = np.stack([noisy(rng, mu, ri) for ri in r])
        rejected_off += fedguard(X, None, r, {"mode": mode})[1][-1] < 0.5 / 10
        rejected_on += fedguard(X, None, r, {"mode": mode, "noise_tol_z": 3.0})[1][-1] < 0.5 / 10
        Xa = np.vstack([X[:9], -5 * mu, X[9:]])
        ra = np.append(np.insert(r[:9], 9, 0.05), 3.0)
        assert fedguard(Xa, None, ra, {"mode": mode, "noise_tol_z": 3.0})[1][9] < 1e-6
    assert rejected_on <= 2 and rejected_on < rejected_off


@pytest.mark.parametrize("p", VARIANTS, ids=IDS)
def test_permutation_invariance(p):
    rng = np.random.default_rng(3)
    mu = direction(rng)
    r = rng.uniform(0.1, 1.0, size=8)
    X = np.stack([noisy(rng, mu, ri) for ri in r[:6]] + [-3 * mu, rng.normal(scale=0.1, size=D)])
    z, a, _ = fedguard(X, None, r, p)
    perm = rng.permutation(8)
    z2, a2, _ = fedguard(X[perm], None, r[perm], p)
    assert np.allclose(z, z2, atol=1e-8) and np.allclose(a[perm], a2, atol=1e-8)


@pytest.mark.parametrize("p", VARIANTS, ids=IDS)
def test_scale_invariance(p):
    rng = np.random.default_rng(4)
    mu = direction(rng)
    r = rng.uniform(0.1, 1.0, size=8)
    X = np.stack([noisy(rng, mu, ri) for ri in r[:7]] + [-4 * mu])
    _, a, _ = fedguard(X, None, r, p)
    _, a10, _ = fedguard(10 * X, None, 10 * r, p)
    assert np.allclose(a, a10, atol=1e-8)
    assert np.allclose(autogm(X)[1], autogm(10 * X)[1], atol=1e-8)


def test_autogm_rejects_far_attacker_and_lambda_controls_strictness():
    rng = np.random.default_rng(5)
    mu = direction(rng)
    X = np.stack([noisy(rng, mu, 0.1) for _ in range(9)] + [-5 * mu])
    _, a, _ = autogm(X, params={"lam_scale": 1.0})
    assert a[-1] < 1e-6
    _, a_soft, _ = autogm(X[:9], params={"lam_scale": 100.0})       # huge lambda -> nearly equal weights
    assert np.allclose(a_soft, 1 / 9, atol=0.01)


def test_registry_and_too_clean():
    assert get_aggregator("autogm") is autogm and get_aggregator("fedguard") is fedguard
    with pytest.raises(ValueError):
        fedguard(np.ones((3, 4)), None, np.zeros(3), {"mode": "cubic"})
    r = np.array([1.0, 1.0])
    assert list(too_clean(np.array([0.2, 1.0]), r, D)) == [True, False]
