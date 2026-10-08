"""Cross-check of our AutoGM against the AutoGM of the Blades benchmark (plan 11.2, 'a fair baseline').

The two functions below are a line-by-line NumPy port of Blades v0.1.0 (by AutoGM's first author; Apache-2.0):
  https://github.com/lishenghui/blades/blob/v0.1.0/src/blades/aggregators/autogm.py
  https://github.com/lishenghui/blades/blob/v0.1.0/src/blades/aggregators/geomed.py
Blades uses a fixed absolute lambda, so our autogm is called with params {"lam": <same value>}.
"""
import numpy as np
import pytest

from fedguard.aggregators import autogm, weighted_geomed

def blades_geomed(U, weights, maxiter=100, eps=1e-6, ftol=1e-10, fixed=False):
    """Faithful NumPy port of blades v0.1.0 Geomed.__call__ (fixed=True: keep the alphas, i.e. textbook Weiszfeld)."""
    alphas = np.asarray(weights, float); w = alphas.copy()
    obj = lambda m, ws: sum(a * np.linalg.norm(m - p) for a, p in zip(ws, U))
    med = U.mean(0); ov = obj(med, w)
    for _ in range(maxiter):
        pov = ov
        base = alphas if fixed else w
        w = np.array([max(eps, a / max(eps, np.linalg.norm(med - p))) for a, p in zip(base, U)]); w = w / w.sum()
        med = (w[:, None] * U).sum(0)
        ov = obj(med, w)
        if abs(pov - ov) < ftol * ov: break
    return med

def blades_autogm(U, lamb=None, maxiter=100, eps=1e-6, ftol=1e-10, fixed=False):
    lamb = 1 * len(U) if lamb is None else lamb
    alpha = np.ones(len(U)) / len(U)
    med = blades_geomed(U, alpha, maxiter, eps, ftol, fixed)
    obj = lambda m, a: sum(x * np.linalg.norm(m - p) for x, p in zip(a, U))
    g = obj(med, alpha) + lamb * np.linalg.norm(alpha) ** 2 / 2
    dist = np.zeros(len(U))
    for _ in range(maxiter):
        pg = g
        dist = np.linalg.norm(U - med, axis=1)
        idxs = np.argsort(dist); eta_opt = 1e16
        for p in range(len(idxs)):
            eta = (dist[idxs[:p + 1]].sum() + lamb) / (p + 1)
            if eta - dist[idxs[p]] < 0: break
            eta_opt = eta
        alpha = np.maximum(eta_opt - dist, 0) / lamb
        med = blades_geomed(U, alpha, maxiter, eps, ftol, fixed)
        g = obj(med, alpha) + lamb * np.linalg.norm(alpha) ** 2 / 2
        if abs(pg - g) < ftol * g: break
    return med, alpha


@pytest.mark.parametrize("trial", range(4))
def test_our_autogm_matches_blades(trial):
    rng = np.random.default_rng(trial)
    D = 2000
    mu = rng.normal(size=D); mu /= np.linalg.norm(mu)
    honest = [mu + rng.normal(scale=0.3 / np.sqrt(D), size=D) + rng.normal(scale=rng.uniform(0.1, 1.5) / np.sqrt(D), size=D)
              for _ in range(7)]
    X = np.stack(honest + [-3 * mu, -3 * mu + 0.1 * rng.normal(size=D) / np.sqrt(D)])
    lam = float(np.median(np.linalg.norm(X - weighted_geomed(X, np.ones(9)), axis=1)))
    z_ours, a_ours, _ = autogm(X, params={"lam": lam, "outer": 100})
    z_bl, a_bl = blades_autogm(X, lam)
    assert np.abs(a_ours - a_bl).max() < 1e-6
    assert np.linalg.norm(z_ours - z_bl) < 1e-6 * np.linalg.norm(z_ours)
    assert a_ours[-2:].max() < 1e-9                                   # both attackers get zero weight
