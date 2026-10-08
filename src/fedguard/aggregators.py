"""Server-side aggregation rules. PURE NumPy - no PyTorch, no Flower - so the same function is used by our
engine (fl_sim.py) and by the Flower strategy wrapper, and can be unit-tested on synthetic vectors.

Every aggregator has the same signature:

    z, weights, info = aggregate(X, n, r, params)

    X       (K, d) float array - one row per client = that client's update (delta = local - global)
    n       (K,)   number of training samples each client used
    r       (K,)   each client's DECLARED expected DP-noise radius (0 when there is no DP)
    params  dict   aggregator settings from the config
    z       (d,)   the aggregated update the server adds to the global model
    weights (K,)   the weight each client received (sums to 1) - logged every round
    info    dict   extra diagnostics (distances, residuals, ...)

Reference: the project guide's Appendix A (tested there on synthetic data), adapted to this signature.
"""
import numpy as np


def fedavg(X, n, r=None, params=None):
    """Sample-weighted mean (McMahan et al., ref [5]). One extreme update can move it anywhere."""
    X = np.asarray(X, dtype=np.float64)
    n = np.asarray(n, dtype=np.float64)
    w = n / n.sum()
    return w @ X, w, {}


def project_simplex(v):
    """Euclidean projection of v onto {a : a >= 0, sum(a) = 1} (Duchi et al., 2008)."""
    v = np.asarray(v, dtype=np.float64)
    u = np.sort(v)[::-1]
    css = np.cumsum(u)
    k = np.arange(1, v.size + 1)
    rho = np.nonzero(u * k > (css - 1.0))[0][-1]
    theta = (css[rho] - 1.0) / (rho + 1.0)
    return np.maximum(v - theta, 0.0)


def weighted_geomed(X, w, z0=None, iters=100, tol=1e-7, eps=1e-10):
    """Weighted geometric median via Weiszfeld iterations: repeatedly average the points with weights
    w_k / ||x_k - z||, so far-away points pull less than in a mean."""
    w = np.asarray(w, dtype=np.float64)
    z = np.average(X, axis=0, weights=w) if z0 is None else z0.copy()
    for _ in range(iters):
        dist = np.linalg.norm(X - z, axis=1)
        beta = w / np.maximum(dist, eps)
        z_new = beta @ X / beta.sum()
        if np.linalg.norm(z_new - z) <= tol * max(1.0, np.linalg.norm(z)):
            return z_new
        z = z_new
    return z


def autogm(X, n=None, r=None, params=None):
    """AutoGM (Li, Ngai, Voigt 2023, ref [13]):  min_{z, a in simplex}  sum_k a_k ||z - x_k|| + (lam/2)||a||^2.
    Alternates (1) z = a-weighted geometric median, (2) a = simplex projection of -dist/lam (far clients get
    less weight, beyond a threshold zero). lam = lam_scale * median distance, so it is scale-free
    (implementation choice; the paper treats lam as a user hyper-parameter -> tuned on validation).
    AutoGM ignores the declared noise r: an honest client that adds a lot of DP noise looks 'far'."""
    p = params or {}
    lam_scale, outer = float(p.get("lam_scale", 1.0)), int(p.get("outer", 20))
    X = np.asarray(X, dtype=np.float64)
    K = len(X)
    a = np.full(K, 1.0 / K)
    z = weighted_geomed(X, a)
    d = np.linalg.norm(X - z, axis=1)
    for _ in range(outer):
        d = np.linalg.norm(X - z, axis=1)
        lam = lam_scale * max(np.median(d), 1e-12)
        a = project_simplex(-d / lam)
        z = weighted_geomed(X, a, z0=z)
    d = np.linalg.norm(X - z, axis=1)
    return z, a, {"dist": d}


def fedguard(X, n=None, r=None, params=None):
    """FedGuard = AutoGM whose trust weights are computed on the displacement that the client's DECLARED DP
    noise and normal non-IID spread cannot explain (paper rule e_k = max(0, dist_k - r_k - s), refined):

      R1  r_eff: the centre z = sum_j b_j x_j is itself noisy, so the noise in (x_k - z) has expected squared
          length (1 - b_k)^2 r_k^2 + sum_{j != k} b_j^2 r_j^2   (b = effective Weiszfeld weights).
          params use_r_eff=False uses r_k alone (ablation).
      R2  mode 'quadrature': residual = sqrt(dist^2 - r_eff^2) (noise ~90 deg to the signal in high d);
          mode 'linear': residual = dist - r_eff (closest to the paper's text).
      R3  allowance s = median(residual) + k_mad * 1.4826 * MAD(residual), from noise-cleaned residuals.
      R4  noise_tol_z (optional): also forgive z standard deviations of the noise LENGTH itself:
          ||noise|| ~ r_eff +/- r_eff/sqrt(2d)  ->  linear: - z*r_eff/sqrt(2d);
          ||noise||^2 ~ r_eff^2 +/- r_eff^2*sqrt(2/d)  ->  quadrature: - z*r_eff^2*sqrt(2/d) inside the root.
    Excess e = max(0, residual - s); trust a = simplex projection of -e / (lam_scale * s)."""
    p = params or {}
    lam_scale, k_mad = float(p.get("lam_scale", 1.0)), float(p.get("k_mad", 2.0))
    outer, mode = int(p.get("outer", 20)), p.get("mode", "quadrature")
    use_r_eff, z_tol = bool(p.get("use_r_eff", True)), p.get("noise_tol_z")
    z_tol = 0.0 if z_tol is None else float(z_tol)
    if mode not in ("linear", "quadrature"):
        raise ValueError(f"unknown fedguard mode '{mode}'")
    X = np.asarray(X, dtype=np.float64)
    K, dim = X.shape
    r = np.zeros(K) if r is None else np.asarray(r, dtype=np.float64)
    a = np.full(K, 1.0 / K)
    z = weighted_geomed(X, a)
    info = {}
    for _ in range(outer):
        d = np.linalg.norm(X - z, axis=1)
        if use_r_eff:
            b = a / np.maximum(d, 1e-10)
            b = b / b.sum()                                         # effective linear weights of the centre
            tot = np.sum(b ** 2 * r ** 2)
            r_eff = np.sqrt(np.maximum(0.0, (1 - b) ** 2 * r ** 2 + tot - b ** 2 * r ** 2))
        else:
            r_eff = r
        if mode == "linear":
            res = np.maximum(0.0, d - r_eff - z_tol * r_eff / np.sqrt(2 * dim))
        else:
            res = np.sqrt(np.maximum(0.0, d ** 2 - r_eff ** 2 - z_tol * r_eff ** 2 * np.sqrt(2.0 / dim)))
        med = np.median(res)
        mad = 1.4826 * np.median(np.abs(res - med))
        s = med + k_mad * mad
        e = np.maximum(0.0, res - s)
        lam = lam_scale * max(s, 1e-12)
        a = project_simplex(-e / lam)
        z = weighted_geomed(X, a, z0=z)
        info = {"dist": d, "r_eff": r_eff, "residual": res, "s": s, "excess": e}
    return z, a, info


def too_clean(dist, r, d_params, z_score=6.0):
    """Optional consistency check (future work, paper section 7): in high dimension the length of Gaussian
    noise concentrates tightly around r, so a client that declares radius r but sends an update much CLEANER
    than that is suspicious (e.g. an attacker who declares large sigma but adds no noise)."""
    spread = (np.asarray(r, dtype=np.float64) / np.sqrt(d_params)) / np.sqrt(2.0)
    return np.asarray(dist) < (np.asarray(r) - z_score * spread)


AGGREGATORS = {"fedavg": fedavg, "autogm": autogm, "fedguard": fedguard}


def get_aggregator(name):
    try:
        return AGGREGATORS[name]
    except KeyError:
        raise ValueError(f"unknown aggregator '{name}'; available: {sorted(AGGREGATORS)}") from None
