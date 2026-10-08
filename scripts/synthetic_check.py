"""Synthetic LOGIC check of AutoGM vs FedGuard (plan 12.3). This is not a result: the synthetic noise follows
FedGuard's own assumptions (Gaussian, honestly declared). It only shows that the code does what the idea says.

    python scripts/synthetic_check.py

Scenario A (heterogeneous privacy + attackers): d = 4000, 10 clients; honest update = shared signal (length 1)
+ non-IID spread (length 0.3) + DP noise of radius noise_scale x U(0.5, 1.5); 3 sign-flip attackers who also add
DP noise and declare a normal radius. 10 seeds per noise level.
Scenario B (quiet majority): 9 quiet honest clients (radius 0.05) + 1 very private honest client (radius 3),
200 seeds; does the private client keep its weight, and is a -5x attacker still excluded?
Writes results/synthetic/*.csv and results/figures/synthetic_*.png.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fedguard.aggregators import autogm, fedguard
from fedguard.metrics import attacker_weight_share, honest_rejection_rate
from fedguard.utils import REPO_ROOT

D, K = 4000, 10
METHODS = {
    "AutoGM (lambda x1)": (autogm, {"lam_scale": 1.0}),
    "AutoGM (lambda x4)": (autogm, {"lam_scale": 4.0}),
    "FedGuard linear": (fedguard, {"mode": "linear"}),
    "FedGuard quadrature": (fedguard, {"mode": "quadrature"}),
    "FedGuard linear + R4": (fedguard, {"mode": "linear", "noise_tol_z": 3.0}),
    "FedGuard quadrature + R4": (fedguard, {"mode": "quadrature", "noise_tol_z": 3.0}),
}


def gauss(rng, length):
    return rng.normal(scale=length / np.sqrt(D), size=D)


def scenario_a(noise_scale, seed, n_att=3, spread=0.3):
    rng = np.random.default_rng(seed)
    mu = rng.normal(size=D); mu /= np.linalg.norm(mu)
    r = noise_scale * rng.uniform(0.5, 1.5, size=K)
    att = np.zeros(K, bool); att[:n_att] = True
    X = np.stack([(-1 if att[k] else 1) * (mu + gauss(rng, spread)) + gauss(rng, r[k]) for k in range(K)])
    return X, r, att


def main():
    out = REPO_ROOT / "results" / "synthetic"; out.mkdir(parents=True, exist_ok=True)
    fig_dir = REPO_ROOT / "results" / "figures"
    rows = []
    for ns in np.arange(0.25, 3.01, 0.25):
        for seed in range(10):
            X, r, att = scenario_a(ns, seed)
            for name, (fn, p) in METHODS.items():
                _, a, _ = fn(X, None, r, p)
                rows.append({"noise_scale": ns, "seed": seed, "method": name,
                             "hrr": honest_rejection_rate(a, att), "attacker_share": attacker_weight_share(a, att)})
    a_df = pd.DataFrame(rows)
    a_df.to_csv(out / "scenario_a.csv", index=False, float_format="%.4f")
    agg = a_df.groupby(["method", "noise_scale"])[["hrr", "attacker_share"]].mean().reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for name, g in agg.groupby("method"):
        ls = "--" if name.startswith("AutoGM") else "-"
        axes[0].plot(g.noise_scale, g.hrr, ls, marker="o", ms=3, label=name)
        axes[1].plot(g.noise_scale, g.attacker_share, ls, marker="o", ms=3, label=name)
    axes[0].set_ylabel("honest rejection rate"); axes[1].set_ylabel("attackers' weight share (3 of 10)")
    for ax in axes:
        ax.set_xlabel("DP noise scale (radius / signal length)"); ax.grid(color="#eee")
    axes[1].axhline(0.3, color="k", lw=0.8, ls=":"); axes[0].legend(fontsize=7, frameon=False)
    fig.suptitle("Synthetic LOGIC check (not a result): heterogeneous privacy + 3 smart sign-flip attackers, 10 seeds",
                 fontsize=10)
    fig.tight_layout(); fig.savefig(fig_dir / "synthetic_scenario_a.png", dpi=150); plt.close(fig)

    rows = []
    for seed in range(200):
        rng = np.random.default_rng(1000 + seed)
        mu = rng.normal(size=D); mu /= np.linalg.norm(mu)
        r = np.array([0.05] * 9 + [3.0])
        X = np.stack([mu + gauss(rng, ri) for ri in r])
        Xa = np.vstack([X, -5 * mu]); ra = np.append(r, 0.05)
        for name, (fn, p) in METHODS.items():
            _, a, _ = fn(X, None, r, p)
            _, aa, _ = fn(Xa, None, ra, p)
            rows.append({"seed": seed, "method": name, "private_rejected": a[-1] < 0.5 / K,
                         "attacker_weight": aa[-1]})
    b_df = pd.DataFrame(rows)
    b_df.to_csv(out / "scenario_b.csv", index=False, float_format="%.4f")
    tb = b_df.groupby("method").agg(private_rejected=("private_rejected", "mean"),
                                    attacker_weight_max=("attacker_weight", "max"))
    print("Scenario A (mean over seeds):"); print(agg.pivot(index="noise_scale", columns="method", values="hrr").round(2).to_string())
    print(agg.pivot(index="noise_scale", columns="method", values="attacker_share").round(2).to_string())
    print("\nScenario B (quiet majority, 200 seeds):"); print(tb.round(3).to_string())
    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.barh(tb.index, tb.private_rejected * 100, color="#E76F51")
    for i, v in enumerate(tb.private_rejected * 100):
        ax.text(v + 1, i, f"{v:.1f}%  (attacker weight max {tb.attacker_weight_max.iloc[i]:.3f})", va="center", fontsize=8)
    ax.set_xlabel("% of seeds where the very private honest client is rejected"); ax.set_xlim(0, 115)
    ax.set_title("Synthetic LOGIC check: quiet majority + 1 private client (200 seeds)", fontsize=10)
    fig.tight_layout(); fig.savefig(fig_dir / "synthetic_scenario_b.png", dpi=150); plt.close(fig)
    print("-> results/synthetic/, results/figures/synthetic_scenario_[ab].png")


if __name__ == "__main__":
    main()
