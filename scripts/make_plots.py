"""Figures and summary tables from results/master.csv (run scripts/collect_results.py first).

    python scripts/make_plots.py --b1          # Phase 4: FedAvg vs non-IID level, vs the B0 ceiling
    python scripts/make_plots.py --b2          # Phase 5: DP-FedAvg, macro-F1 vs epsilon
    python scripts/make_plots.py --attacks     # Phase 6: attacks vs FedAvg / DP-FedAvg / AutoGM (+ FedGuard later)
    python scripts/make_plots.py --e4          # Phase 6: O2 conflict heatmaps (HRR, macro-F1) over epsilon x alpha

More experiment groups are added here phase by phase. Every figure is made by code (plan rule: no
hand-made Excel charts).
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fedguard.utils import REPO_ROOT

RES = REPO_ROOT / "results"
FIG = RES / "figures"
B0_SUMMARY = RES / "b0_centralised" / "summary.csv"
KEYS = ["test_macro_f1", "test_accuracy", "test_benign_fpr", "test_attack_detection_rate", "test_binary_f1"]


def b0_mean(metric="macro_f1"):
    s = pd.read_csv(B0_SUMMARY)
    return float(s.loc[s["seed"] == "mean", metric].iloc[0]), float(s.loc[s["seed"] == "std", metric].iloc[0])


def b1(master):
    df = master[(master["name"] == "b1_fedavg") & (master["attack"] == "none") & master["epsilon"].isna()]
    if df.empty:
        print("no b1_fedavg runs yet"); return
    out = RES / "b1_fedavg"; out.mkdir(parents=True, exist_ok=True)
    table = df.groupby("alpha")[KEYS + ["best_round", "rounds_run", "seconds_per_round"]].agg(["mean", "std", "count"])
    table.to_csv(out / "summary_by_alpha.csv", float_format="%.4f")
    b0, b0s = b0_mean()
    print("B1 FedAvg (test, mean +/- std over seeds) vs B0 ceiling", f"{b0:.4f} +/- {b0s:.4f}")
    for a, g in df.groupby("alpha"):
        print(f"  alpha={a:<5g} macro-F1 {g.test_macro_f1.mean():.4f} +/- {g.test_macro_f1.std():.4f}  "
              f"acc {g.test_accuracy.mean():.4f}  FPR {g.test_benign_fpr.mean():.4f}  "
              f"detection {g.test_attack_detection_rate.mean():.4f}  gap to B0 {g.test_macro_f1.mean() - b0:+.4f}  (n={len(g)})")

    # 1) macro-F1 per alpha vs the B0 band
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    al = sorted(df["alpha"].unique()); x = np.arange(len(al))
    m = [df[df.alpha == a].test_macro_f1.mean() for a in al]; s = [df[df.alpha == a].test_macro_f1.std() for a in al]
    ax.bar(x, m, yerr=s, capsize=4, color="#1F3A5F", label="B1 FedAvg (test, 3 seeds)")
    ax.axhspan(b0 - b0s, b0 + b0s, color="#2A9D8F", alpha=0.25); ax.axhline(b0, color="#2A9D8F", label=f"B0 centralised {b0:.3f}")
    for i, v in enumerate(m): ax.text(i, v + 0.015, f"{v:.3f}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels([f"alpha={a:g}" for a in al]); ax.set_ylabel("test macro-F1")
    ax.set_ylim(0, max(max(m), b0) + 0.12); ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.set_title("Cost of federation and non-IID data (5 clients, no DP)", fontsize=10)
    fig.tight_layout(); fig.savefig(FIG / "b1_fedavg_by_alpha.png", dpi=150); plt.close(fig)

    # 2) validation macro-F1 vs round, one line per alpha (mean over seeds)
    fig, ax = plt.subplots(figsize=(8, 3.6))
    colors = {0.1: "#E76F51", 0.5: "#6D597A", 100: "#2A9D8F"}
    for a in al:
        curves = [pd.read_csv(RES / "runs" / r / "rounds.csv").set_index("round")["val_macro_f1"]
                  for r in df[df.alpha == a]["run"]]
        c = pd.concat(curves, axis=1)
        ax.plot(c.index, c.mean(axis=1), color=colors.get(a), label=f"alpha={a:g}")
        ax.fill_between(c.index, c.min(axis=1), c.max(axis=1), color=colors.get(a), alpha=0.15)
    ax.axhline(b0, color="#2A9D8F", ls="--", lw=1, label="B0 test macro-F1")
    ax.set_xlabel("federated round"); ax.set_ylabel("validation macro-F1"); ax.grid(color="#eee")
    ax.legend(frameon=False, fontsize=8); ax.set_title("FedAvg learning curves (mean, band = min-max over seeds)", fontsize=10)
    fig.tight_layout(); fig.savefig(FIG / "b1_fedavg_rounds.png", dpi=150); plt.close(fig)

    # 3) per-class recall per alpha
    rec_cols = [c for c in df.columns if c.startswith("test_recall_")]
    pc = df.groupby("alpha")[rec_cols].mean().T
    pc.index = [c.replace("test_recall_", "") for c in pc.index]
    pc.to_csv(out / "per_class_recall_by_alpha.csv", float_format="%.4f")
    fig, ax = plt.subplots(figsize=(9, 3.6)); w = 0.8 / len(al); x = np.arange(len(pc))
    for i, a in enumerate(al):
        ax.bar(x + (i - (len(al) - 1) / 2) * w, pc[a], w, color=colors.get(a), label=f"alpha={a:g}")
    ax.set_xticks(x); ax.set_xticklabels(pc.index, rotation=35, ha="right"); ax.set_ylim(0, 1.05)
    ax.set_ylabel("test recall"); ax.legend(frameon=False, fontsize=8)
    ax.set_title("FedAvg per-class recall by non-IID level (mean over seeds)", fontsize=10)
    fig.tight_layout(); fig.savefig(FIG / "b1_fedavg_per_class_recall.png", dpi=150); plt.close(fig)
    print("figures -> results/figures/b1_fedavg_*.png, tables -> results/b1_fedavg/")


def eps_label(e):
    return "inf (no DP)" if pd.isna(e) else f"{e:g}"


def b2(master):
    df = master[(master["name"] == "b2_dpfedavg") & (master["attack"] == "none")].copy()
    if df.empty:
        print("no b2_dpfedavg runs yet"); return
    out = RES / "b2_dpfedavg"; out.mkdir(parents=True, exist_ok=True)
    df["eps_key"] = df["epsilon"].fillna(np.inf)
    order = sorted(df["eps_key"].unique(), reverse=True)             # inf, 8, 3, 1
    keys = KEYS + ["epsilon_spent_max", "sigma_min", "sigma_max", "best_round", "rounds_run", "seconds_per_round"]
    table = df.groupby("eps_key")[keys].agg(["mean", "std", "count"]).loc[order]
    table.to_csv(out / "summary_by_epsilon.csv", float_format="%.4f")
    b0, b0s = b0_mean()
    b1 = master[(master["name"] == "b1_fedavg") & (master["alpha"] == 0.5) & master["epsilon"].isna()]
    print("B2 DP-FedAvg (alpha 0.5, test, mean +/- std over seeds); B0", f"{b0:.4f}",
          "; B1 (alpha 0.5, up to 120 rounds)", f"{b1.test_macro_f1.mean():.4f}" if len(b1) else "n/a")
    for e in order:
        g = df[df.eps_key == e]
        spent = g.epsilon_spent_max.mean() if g.epsilon_spent_max.notna().any() else np.inf
        print(f"  eps={eps_label(g.epsilon.iloc[0]):<11s} macro-F1 {g.test_macro_f1.mean():.4f} +/- {g.test_macro_f1.std():.4f}"
              f"  FPR {g.test_benign_fpr.mean():.4f}  detection {g.test_attack_detection_rate.mean():.4f}"
              f"  eps spent {spent:.2f}  sigma {g.sigma_min.mean():.2f}-{g.sigma_max.mean():.2f}  (n={len(g)})")

    colors = ["#2A9D8F", "#6D597A", "#E9A23B", "#E76F51"]
    # 1) macro-F1 vs epsilon, with B0 and B1 references
    fig, ax = plt.subplots(figsize=(6.5, 3.6)); x = np.arange(len(order))
    m = [df[df.eps_key == e].test_macro_f1.mean() for e in order]; sd = [df[df.eps_key == e].test_macro_f1.std() for e in order]
    ax.errorbar(x, m, yerr=sd, marker="o", capsize=4, color="#1F3A5F", label="B2 DP-FedAvg (test, 3 seeds)")
    for i, v in enumerate(m): ax.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=8)
    ax.axhline(b0, color="#2A9D8F", ls="--", lw=1, label=f"B0 centralised {b0:.3f}")
    if len(b1): ax.axhline(b1.test_macro_f1.mean(), color="#6D597A", ls=":", lw=1.2,
                           label=f"B1 FedAvg, 120-round budget {b1.test_macro_f1.mean():.3f}")
    ax.set_xticks(x); ax.set_xticklabels([eps_label(df[df.eps_key == e].epsilon.iloc[0]) for e in order])
    ax.set_xlabel("privacy budget epsilon per client (delta = 1/N)   <- less private | more private ->")
    ax.set_ylabel("test macro-F1"); ax.set_ylim(0, max(m + [b0]) + 0.12); ax.legend(frameon=False, fontsize=8)
    ax.set_title("Cost of privacy: DP-FedAvg, 5 clients, alpha = 0.5, 40-round budget", fontsize=10)
    fig.tight_layout(); fig.savefig(FIG / "b2_dpfedavg_by_epsilon.png", dpi=150); plt.close(fig)

    # 2) learning curves per epsilon
    fig, ax = plt.subplots(figsize=(8, 3.6))
    for e, col in zip(order, colors):
        c = pd.concat([pd.read_csv(RES / "runs" / r / "rounds.csv").set_index("round")["val_macro_f1"]
                       for r in df[df.eps_key == e]["run"]], axis=1)
        ax.plot(c.index, c.mean(axis=1), color=col, label=f"eps={eps_label(df[df.eps_key == e].epsilon.iloc[0])}")
        ax.fill_between(c.index, c.min(axis=1), c.max(axis=1), color=col, alpha=0.15)
    ax.set_xlabel("federated round"); ax.set_ylabel("validation macro-F1"); ax.grid(color="#eee")
    ax.legend(frameon=False, fontsize=8); ax.set_title("DP-FedAvg learning curves (mean, band = min-max over seeds)", fontsize=10)
    fig.tight_layout(); fig.savefig(FIG / "b2_dpfedavg_rounds.png", dpi=150); plt.close(fig)

    # 3) per-class recall per epsilon
    rec_cols = [c for c in df.columns if c.startswith("test_recall_")]
    pc = df.groupby("eps_key")[rec_cols].mean().T.loc[:, order]
    pc.index = [c.replace("test_recall_", "") for c in pc.index]
    pc.columns = [eps_label(df[df.eps_key == e].epsilon.iloc[0]) for e in order]
    pc.to_csv(out / "per_class_recall_by_epsilon.csv", float_format="%.4f")
    fig, ax = plt.subplots(figsize=(9, 3.6)); w = 0.8 / len(order); x = np.arange(len(pc))
    for i, (col, c) in enumerate(zip(pc.columns, colors)):
        ax.bar(x + (i - (len(order) - 1) / 2) * w, pc[col], w, color=c, label=f"eps={col}")
    ax.set_xticks(x); ax.set_xticklabels(pc.index, rotation=35, ha="right"); ax.set_ylim(0, 1.05)
    ax.set_ylabel("test recall"); ax.legend(frameon=False, fontsize=8)
    ax.set_title("DP-FedAvg per-class recall by privacy level (mean over seeds)", fontsize=10)
    fig.tight_layout(); fig.savefig(FIG / "b2_dpfedavg_per_class_recall.png", dpi=150); plt.close(fig)
    print("figures -> results/figures/b2_dpfedavg_*.png, tables -> results/b2_dpfedavg/")


def method_label(row):
    dp = "no DP" if pd.isna(row["epsilon"]) else f"DP eps {row['epsilon']}"
    return f"{ {'fedavg': 'FedAvg', 'autogm': 'AutoGM', 'fedguard': 'FedGuard'}[row['aggregator']] } ({dp})"


ATTACK_ORDER = ["none", "label_flip", "sign_flip", "gaussian"]


def attacks(master, name="p6_attacks"):
    df = master[master["name"] == name].copy()
    if df.empty:
        print(f"no {name} runs yet"); return
    out = RES / name; out.mkdir(parents=True, exist_ok=True)
    df["method"] = df.apply(method_label, axis=1)
    keys = ["test_macro_f1", "test_binary_f1", "test_benign_fpr", "test_attack_detection_rate", "hrr_mean",
            "attacker_share_mean"]
    table = df.groupby(["method", "attack", "attack_fraction"])[keys].agg(["mean", "std", "count"])
    table.to_csv(out / "summary.csv", float_format="%.4f")
    print(df.groupby(["method", "attack"])[["test_macro_f1", "test_binary_f1", "hrr_mean", "attacker_share_mean"]]
          .mean().round(3).to_string())
    methods = sorted(df["method"].unique())
    att = [a for a in ATTACK_ORDER if a in set(df["attack"])]
    colors = ["#1F3A5F", "#6D597A", "#E9A23B", "#E76F51", "#2A9D8F", "#8D99AE"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.8))
    for ax, (metric, label) in zip(axes, [("test_macro_f1", "test macro-F1"), ("test_binary_f1", "test binary F1"),
                                          ("attacker_share_mean", "attackers' weight share (mean over rounds)")]):
        w = 0.8 / len(methods); x = np.arange(len(att))
        for i, (mth, c) in enumerate(zip(methods, colors)):
            vals = [df[(df.method == mth) & (df.attack == a)][metric].mean() for a in att]
            ax.bar(x + (i - (len(methods) - 1) / 2) * w, vals, w, color=c, label=mth)
        ax.set_xticks(x); ax.set_xticklabels(att); ax.set_ylabel(label); ax.grid(axis="y", color="#eee")
    axes[0].legend(frameon=False, fontsize=7)
    fig.suptitle(f"Attacks ({df.attack_fraction.max():g} of clients) vs aggregators - K = {df.clients.iloc[0]}, "
                 f"alpha = {df.alpha.iloc[0]:g}", fontsize=10)
    fig.tight_layout(); fig.savefig(FIG / f"{name}.png", dpi=150); plt.close(fig)
    print(f"figure -> results/figures/{name}.png, table -> results/{name}/summary.csv")


def e4(master, name="e4_conflict"):
    df = master[(master["name"] == name) & (master["attack"] == "none")].copy()
    if df.empty:
        print(f"no {name} runs yet"); return
    out = RES / name; out.mkdir(parents=True, exist_ok=True)
    df["eps"] = df["epsilon"].fillna("inf").astype(str)
    eps_order = [e for e in ["inf", "8.0", "3.0", "1.0", "mix:1-3-8"] if e in set(df["eps"])]
    for agg, g in df.groupby("aggregator"):
        piv = {m: g.pivot_table(index="alpha", columns="eps", values=m, aggfunc="mean").reindex(columns=eps_order)
               for m in ("hrr_mean", "test_macro_f1")}
        for m, t in piv.items():
            t.to_csv(out / f"{agg}_{m}.csv", float_format="%.4f")
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
        for ax, (m, t), cmap, lab in zip(axes, piv.items(), ["Reds", "Blues"], ["HRR (honest clients rejected)", "test macro-F1"]):
            im = ax.imshow(t.values, cmap=cmap, vmin=0, vmax=1 if m == "hrr_mean" else max(0.65, np.nanmax(t.values)))
            for (i, j), v in np.ndenumerate(t.values):
                ax.text(j, i, "-" if np.isnan(v) else f"{v:.2f}", ha="center", va="center", fontsize=9)
            ax.set_xticks(range(len(t.columns))); ax.set_xticklabels([f"eps {c}" for c in t.columns], fontsize=8)
            ax.set_yticks(range(len(t.index))); ax.set_yticklabels([f"alpha {a:g}" for a in t.index])
            ax.set_title(lab, fontsize=10); fig.colorbar(im, ax=ax, fraction=0.046)
        fig.suptitle(f"O2 conflict (no attackers): {agg}, K = {g.clients.iloc[0]}, mean over seeds", fontsize=10)
        fig.tight_layout(); fig.savefig(FIG / f"{name}_{agg}.png", dpi=150); plt.close(fig)
        print(f"{agg}: HRR"); print(piv["hrr_mean"].round(3).to_string())
        print("macro-F1"); print(piv["test_macro_f1"].round(3).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attacks", action="store_true")
    ap.add_argument("--e4", action="store_true")
    ap.add_argument("--b1", action="store_true")
    ap.add_argument("--b2", action="store_true")
    args = ap.parse_args()
    master = pd.read_csv(RES / "master.csv")
    if args.b1:
        b1(master)
    if args.b2:
        b2(master)
    if args.attacks:
        attacks(master)
    if args.e4:
        e4(master)


if __name__ == "__main__":
    main()
