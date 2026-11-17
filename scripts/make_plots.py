"""Figures and summary tables from results/master.csv (run scripts/collect_results.py first).

    python scripts/make_plots.py              # everything
    python scripts/make_plots.py --b1         # Phase 4: FedAvg vs non-IID level, vs the B0 ceiling
    python scripts/make_plots.py --b2         # Phase 5: DP-FedAvg, macro-F1 vs epsilon
    python scripts/make_plots.py --tune       # Phase 7: FedGuard settings on the VALIDATION split (selection table)
    python scripts/make_plots.py --e5         # E5 main attack table (+ paired FedGuard - AutoGM differences)
    python scripts/make_plots.py --mixed      # E5m mixed privacy (eps 1/3/8 across clients)
    python scripts/make_plots.py --e4         # E4 / O2 conflict heatmaps (HRR, macro-F1) over epsilon x alpha
    python scripts/make_plots.py --e6         # E6 ablation (one change at a time from the frozen FedGuard)
    python scripts/make_plots.py --e7         # E7 stress (attacker share, epsilon)

Runs are selected by their exact settings (all K = 10 runs share the name "k10"); only 40-round runs are used
(Findings F46). Every figure is made by code (plan rule: no hand-made Excel charts). Trust weights of one run:
scripts/demo_dashboard.py --run results/runs/<run> --save <png>.
"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

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
    try:
        return "inf (no DP)" if pd.isna(e) else f"{float(e):g}"
    except (TypeError, ValueError):
        return str(e)


def b2(master):
    df = master[(master["name"] == "b2_dpfedavg") & (master["attack"] == "none")].copy()
    if df.empty:
        print("no b2_dpfedavg runs yet"); return
    out = RES / "b2_dpfedavg"; out.mkdir(parents=True, exist_ok=True)
    df["eps_key"] = df["epsilon"].fillna(np.inf)
    def _eps_sort_key(x):
        try: return -float(x)
        except: return -np.inf
    order = sorted(df["eps_key"].unique(), key=_eps_sort_key)  # inf, 8, 3, 1
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


# ---------------------------------------------------------------- K = 10 experiments (Phases 6-8)
# Every K = 10 run shares the name "k10" (F40), so each experiment is selected by its exact settings - never by
# averaging everything with that name. FedGuard's settings come from configs/fedguard_frozen.yaml (F50).
AUTOGM_MAIN = {"lam_scale": 4.0, "outer": 20}
ATTACK_ORDER = ["none", "label_flip", "sign_flip", "gaussian"]
SEEDS = [42, 43, 44]
COLORS = {"FedAvg (no DP)": "#8D99AE", "AutoGM (no DP)": "#B8B8D1", "FedAvg": "#1F3A5F", "AutoGM": "#E9A23B",
          "FedGuard": "#2A9D8F"}


def params_key(d):
    return json.dumps({k: v for k, v in d.items() if k != "name"}, sort_keys=True)


def frozen_fedguard():
    path = REPO_ROOT / "configs" / "fedguard_frozen.yaml"
    if not path.exists():
        return None
    return yaml.safe_load(open(path))["fl"]["aggregator"]


def k10(master):
    df = master[(master["name"] == "k10") & (master["max_rounds"] == 40)].copy()
    df["eps"] = df["epsilon"].apply(eps_str)
    return df


def eps_str(x):
    if pd.isna(x):
        return "inf"
    try:
        return f"{float(x):g}"
    except (TypeError, ValueError):
        return str(x)


def method_of(row, fg_key):
    """Readable method name, or None for runs that are not one of the compared methods (e.g. tuning variants)."""
    if row["aggregator"] == "fedavg":
        name = "FedAvg"
    elif row["aggregator"] == "autogm" and row["agg_params"] == params_key(AUTOGM_MAIN):
        name = "AutoGM"
    elif row["aggregator"] == "fedguard" and fg_key is not None and row["agg_params"] == fg_key:
        name = "FedGuard"
    else:
        return None
    return name + (" (no DP)" if row["eps"] == "inf" else "")


def with_methods(df):
    fg = frozen_fedguard()
    fg_key = params_key(fg) if fg else None
    df = df.copy()
    df["method"] = df.apply(lambda r: method_of(r, fg_key), axis=1)
    return df[df["method"].notna()]


def mean_std(df, by, metrics):
    g = df.groupby(by)[metrics]
    out = g.mean().round(4).add_suffix("_mean").join(g.std().round(4).add_suffix("_std")).join(g.size().rename("n"))
    return out.reset_index()


def bars(ax, df, groups, methods, metric, label):
    w = 0.8 / max(len(methods), 1)
    x = np.arange(len(groups))
    for i, m in enumerate(methods):
        sub = [df[(df.method == m) & (df.group == g)][metric] for g in groups]
        ax.bar(x + (i - (len(methods) - 1) / 2) * w, [s.mean() if len(s) else np.nan for s in sub], w,
               yerr=[s.std() if len(s) > 1 else 0 for s in sub], capsize=2, color=COLORS.get(m, "#999"), label=m)
    ax.set_xticks(x); ax.set_xticklabels(groups, fontsize=8); ax.set_ylabel(label); ax.grid(axis="y", color="#eee")


def tune(master):
    """FedGuard settings on the VALIDATION split (p7_fedguard_tune): mean val macro-F1 over no attack and
    30% sign flip, seed 42 - the same protocol as AutoGM's lambda (F39). Test numbers are NOT shown."""
    df = k10(master)
    df = df[(df.aggregator == "fedguard") & (df.seed == 42) & (df.eps == "3") & (df.alpha == 0.5)
            & (df.attack.isin(["none", "sign_flip"])) & ((df.attack == "none") | (df.attack_fraction == 0.3))]
    if df.empty:
        print("no FedGuard tuning runs yet"); return
    out = RES / "p7_fedguard_tune"; out.mkdir(parents=True, exist_ok=True)
    piv = df.pivot_table(index="agg_params", columns="attack", values=["val_macro_f1", "hrr_mean", "attacker_share_mean"])
    piv.columns = [f"{a}_{m}" for m, a in piv.columns]
    piv["mean_val_macro_f1"] = piv[[c for c in piv.columns if c.endswith("val_macro_f1")]].mean(axis=1, skipna=False)
    piv = piv.sort_values("mean_val_macro_f1", ascending=False)
    piv.to_csv(out / "selection_val.csv", float_format="%.4f")
    print("FedGuard settings on VALIDATION (seed 42, eps 3, alpha 0.5), best first:")
    print(piv.round(3).to_string())
    complete = piv["mean_val_macro_f1"].notna().sum()
    print(f"{complete} of 8 settings have both scenarios finished -> results/p7_fedguard_tune/selection_val.csv")


def e5(master):
    """E5 main attack table: K = 10, alpha 0.5, eps 3 (+ no-DP references), 30% attackers, seeds 42-44."""
    df = with_methods(k10(master))
    df = df[(df.alpha == 0.5) & (df.eps.isin(["3", "inf"])) & ((df.attack == "none") | (df.attack_fraction == 0.3))]
    if df.empty:
        print("no E5 runs yet"); return
    out = RES / "e5_attacks"; out.mkdir(parents=True, exist_ok=True)
    metrics = ["test_macro_f1", "test_binary_f1", "test_benign_fpr", "test_attack_detection_rate", "hrr_mean",
               "attacker_share_mean", "epsilon_spent_max"]
    mean_std(df, ["method", "attack"], metrics).to_csv(out / "summary.csv", index=False)
    per_class = df.groupby(["method", "attack"])[[c for c in df.columns if c.startswith("test_recall_")]].mean()
    per_class.round(4).to_csv(out / "per_class_recall.csv")
    # paired (same seed = same partition and attackers) FedGuard - AutoGM differences
    a = df[df.method == "AutoGM"].set_index(["attack", "seed"])["test_macro_f1"]
    f = df[df.method == "FedGuard"].set_index(["attack", "seed"])["test_macro_f1"]
    paired = (f - a).dropna().rename("fedguard_minus_autogm").reset_index()
    paired.to_csv(out / "paired_fedguard_vs_autogm.csv", index=False, float_format="%.4f")
    print(mean_std(df, ["method", "attack"], ["test_macro_f1", "hrr_mean", "attacker_share_mean"]).to_string(index=False))
    if not paired.empty:
        print("paired FedGuard - AutoGM (test macro-F1):"); print(paired.round(3).to_string(index=False))
    df["group"] = df["attack"]
    methods = [m for m in ["FedAvg (no DP)", "FedAvg", "AutoGM", "FedGuard"] if m in set(df.method)]
    groups = [g for g in ATTACK_ORDER if g in set(df.group)]
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.8))
    for ax, (m, lab) in zip(axes, [("test_macro_f1", "test macro-F1"), ("test_binary_f1", "test binary F1 (attack vs benign)"),
                                   ("attacker_share_mean", "attackers' weight share")]):
        bars(ax, df, groups, methods, m, lab)
    axes[2].axhline(0.3, color="k", ls=":", lw=0.8)
    axes[0].legend(frameon=False, fontsize=7)
    fig.suptitle("E5: attacks (30% of clients) - K = 10, alpha 0.5, DP eps 3 unless 'no DP'; mean +/- std over seeds 42-44", fontsize=10)
    fig.tight_layout(); fig.savefig(FIG / "e5_attacks.png", dpi=150); plt.close(fig)
    print("figure -> results/figures/e5_attacks.png, tables -> results/e5_attacks/")


def weights_by_epsilon(runs):
    """Mean trust weight per declared-epsilon group, from the runs' weights logs."""
    rows = []
    for r in runs:
        run_dir = RES / "runs" / r
        test = json.load(open(run_dir / "metrics_test.json"))
        eps = test.get("epsilon_target_per_client")
        w = pd.read_csv(run_dir / "weights_log.csv")
        w = w[~w["is_attacker"]]
        w["eps"] = w["client"].map(lambda c: eps[c])
        rows.append(w.groupby("eps")["weight"].mean().rename(r))
    return pd.concat(rows, axis=1).mean(axis=1) if rows else pd.Series(dtype=float)


def mixed(master):
    """E5m mixed privacy (eps 1/3/8 across clients): the paper's hospital-and-bank scenario."""
    df = with_methods(k10(master))
    df = df[(df.alpha == 0.5) & (df.eps == "mix:1-3-8") & ((df.attack == "none") | (df.attack_fraction == 0.3))]
    if df.empty:
        print("no mixed-privacy runs yet"); return
    out = RES / "e5_mixed_privacy"; out.mkdir(parents=True, exist_ok=True)
    metrics = ["test_macro_f1", "test_binary_f1", "hrr_mean", "attacker_share_mean"]
    mean_std(df, ["method", "attack"], metrics).to_csv(out / "summary.csv", index=False)
    print(mean_std(df, ["method", "attack"], metrics).to_string(index=False))
    groups = [g for g in ["none", "sign_flip"] if g in set(df.attack)]
    df["group"] = df["attack"]
    methods = [m for m in ["FedAvg", "AutoGM", "FedGuard"] if m in set(df.method)]
    fig, axes = plt.subplots(1, 4, figsize=(18, 3.8))
    for ax, (m, lab) in zip(axes[:3], [("test_macro_f1", "test macro-F1"), ("hrr_mean", "honest rejection rate"),
                                       ("attacker_share_mean", "attackers' weight share")]):
        bars(ax, df, groups, methods, m, lab)
    ax = axes[3]
    table = {}
    for m in methods:
        s = weights_by_epsilon(df[df.method == m]["run"])
        table[m] = s
    tw = pd.DataFrame(table)
    tw.round(4).to_csv(out / "mean_weight_by_epsilon.csv")
    if not tw.empty:
        tw.index = [f"eps {e:g}" for e in tw.index]
        tw.plot.bar(ax=ax, color=[COLORS[m] for m in tw.columns], rot=0)
        ax.axhline(0.1, color="k", ls=":", lw=0.8); ax.axhline(0.05, color="k", ls="--", lw=0.8)
        ax.set_ylabel("mean weight of an honest client"); ax.set_title("who is trusted, by privacy level", fontsize=9)
        ax.legend(frameon=False, fontsize=7)
    axes[0].legend(frameon=False, fontsize=7)
    fig.suptitle("Mixed privacy (eps 1/3/8 across clients) - K = 10, alpha 0.5; mean +/- std over seeds 42-44 "
                 "(dotted: fair share 1/K, dashed: rejection threshold 0.5/K)", fontsize=10)
    fig.tight_layout(); fig.savefig(FIG / "e5_mixed_privacy.png", dpi=150); plt.close(fig)
    print("figure -> results/figures/e5_mixed_privacy.png, tables -> results/e5_mixed_privacy/")


def e4(master):
    """E4 / O2 conflict: honest clients only, eps x alpha heatmaps of HRR and macro-F1 (seeds 42, 43)."""
    df = with_methods(k10(master))
    df = df[(df.attack == "none") & (df.seed.isin([42, 43])) & (df.method.str.startswith(("AutoGM", "FedGuard")))]
    if df.empty:
        print("no E4 runs yet"); return
    out = RES / "e4_conflict"; out.mkdir(parents=True, exist_ok=True)
    df["aggname"] = df["method"].str.replace(" (no DP)", "", regex=False)
    eps_order = [e for e in ["inf", "8", "3", "1", "mix:1-3-8"] if e in set(df["eps"])]
    aggs = [a for a in ["AutoGM", "FedGuard"] if a in set(df["aggname"])]
    fig, axes = plt.subplots(len(aggs), 2, figsize=(11, 3.4 * len(aggs)), squeeze=False)
    for row, agg in zip(axes, aggs):
        g = df[df["aggname"] == agg]
        for ax, m, cmap, lab in zip(row, ["hrr_mean", "test_macro_f1"], ["Reds", "Blues"],
                                    ["HRR (honest clients rejected)", "test macro-F1"]):
            t = g.pivot_table(index="alpha", columns="eps", values=m, aggfunc="mean").reindex(columns=eps_order)
            n = g.pivot_table(index="alpha", columns="eps", values=m, aggfunc="count").reindex(columns=eps_order)
            t.to_csv(out / f"{agg.lower()}_{m}.csv", float_format="%.4f")
            im = ax.imshow(t.values, cmap=cmap, vmin=0, vmax=1 if m == "hrr_mean" else 0.5)
            for (i, j), v in np.ndenumerate(t.values):
                k = n.values[i, j]
                ax.text(j, i, "-" if np.isnan(v) else f"{v:.2f}" + ("" if k == 2 else f"\n(n={int(k)})"),
                        ha="center", va="center", fontsize=8)
            ax.set_xticks(range(len(t.columns))); ax.set_xticklabels([f"eps {c}" for c in t.columns], fontsize=8)
            ax.set_yticks(range(len(t.index))); ax.set_yticklabels([f"alpha {a:g}" for a in t.index])
            ax.set_title(f"{agg}: {lab}", fontsize=10); fig.colorbar(im, ax=ax, fraction=0.046)
            print(f"{agg} {m}:"); print(t.round(3).to_string())
    fig.suptitle("E4 / O2 conflict (no attackers), K = 10, mean over seeds 42 and 43", fontsize=10)
    fig.tight_layout(); fig.savefig(FIG / "e4_conflict.png", dpi=150); plt.close(fig)
    print("figure -> results/figures/e4_conflict.png, tables -> results/e4_conflict/")


ABLATION_FACTORS = {"mode": "R2 mode", "noise_tol_z": "R4 (noise_tol_z)", "s_floor": "s floor",
                    "lam_scale": "lambda_scale", "use_r_eff": "R1 (use_r_eff)"}


def e6(master):
    """E6 ablation: FedGuard variants that differ from the frozen settings in exactly ONE option."""
    fg = frozen_fedguard()
    if fg is None:
        print("E6 needs configs/fedguard_frozen.yaml (F50)"); return
    df = k10(master)
    df = df[(df.aggregator == "fedguard") & (df.alpha == 0.5) & (df.eps == "3")
            & ((df.attack == "none") | (df.attack_fraction == 0.3)) & (df.attack.isin(["none", "sign_flip"]))]
    base = {k: v for k, v in fg.items() if k != "name"}
    rows = []
    for _, r in df.iterrows():
        p = json.loads(r["agg_params"])
        diff = {k for k in set(p) | set(base) if p.get(k) != base.get(k)}
        if len(diff) > 1:
            continue
        variant = "frozen" if not diff else f"{ABLATION_FACTORS.get(next(iter(diff)), next(iter(diff)))} = {p.get(next(iter(diff)))}"
        rows.append({**r.to_dict(), "variant": variant})
    if not rows:
        print("no E6 runs yet"); return
    ab = pd.DataFrame(rows)
    out = RES / "e6_ablation"; out.mkdir(parents=True, exist_ok=True)
    metrics = ["test_macro_f1", "hrr_mean", "attacker_share_mean"]
    s = mean_std(ab, ["variant", "attack"], metrics)
    s.to_csv(out / "summary.csv", index=False)
    print(s.to_string(index=False))
    variants = ["frozen"] + sorted(v for v in ab.variant.unique() if v != "frozen")
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
    for ax, (m, lab) in zip(axes, [("test_macro_f1", "test macro-F1"), ("hrr_mean", "honest rejection rate"),
                                   ("attacker_share_mean", "attackers' weight share")]):
        for k, (att, c) in enumerate([("none", "#2A9D8F"), ("sign_flip", "#E76F51")]):
            sub = [ab[(ab.variant == v) & (ab.attack == att)][m] for v in variants]
            y = np.arange(len(variants)) + (k - 0.5) * 0.38
            ax.barh(y, [x.mean() if len(x) else np.nan for x in sub], 0.38,
                    xerr=[x.std() if len(x) > 1 else 0 for x in sub], color=c, label=att, capsize=2)
        ax.set_yticks(range(len(variants))); ax.set_yticklabels(variants if ax is axes[0] else [], fontsize=8)
        ax.set_xlabel(lab); ax.grid(axis="x", color="#eee")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle(f"E6 ablation - one change at a time from the frozen FedGuard ({params_key(fg)}); "
                 "K = 10, alpha 0.5, eps 3, seeds 42-44", fontsize=9)
    fig.tight_layout(); fig.savefig(FIG / "e6_ablation.png", dpi=150); plt.close(fig)
    print("figure -> results/figures/e6_ablation.png, table -> results/e6_ablation/summary.csv")


def e7(master):
    """E7 stress (seed 42): attacker share 10-40% (sign flip, eps 3) and privacy eps 8 / 3 / 1."""
    df = with_methods(k10(master))
    df = df[(df.alpha == 0.5) & (df.seed == 42) & (~df.method.str.contains("no DP"))]
    out = RES / "e7_stress"; out.mkdir(parents=True, exist_ok=True)
    sf = df[(df.attack == "sign_flip") & (df.eps == "3")]
    ep = df[(df.eps.isin(["8", "3", "1"])) & ((df.attack == "none") | (df.attack_fraction == 0.3))
            & (df.attack.isin(["none", "sign_flip"]))]
    if sf.attack_fraction.nunique() < 2 and ep.eps.nunique() < 2:
        print("no E7 runs yet"); return
    sf[["method", "attack_fraction", "test_macro_f1", "attacker_share_mean", "hrr_mean"]].sort_values(
        ["method", "attack_fraction"]).to_csv(out / "attacker_share.csv", index=False, float_format="%.4f")
    ep[["method", "eps", "attack", "test_macro_f1", "attacker_share_mean", "hrr_mean"]].sort_values(
        ["method", "attack", "eps"]).to_csv(out / "epsilon.csv", index=False, float_format="%.4f")
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.8))
    for m in ["FedAvg", "AutoGM", "FedGuard"]:
        s = sf[sf.method == m].sort_values("attack_fraction")
        if not s.empty:
            axes[0].plot(s.attack_fraction * 100, s.test_macro_f1, marker="o", color=COLORS[m], label=m)
            axes[1].plot(s.attack_fraction * 100, s.attacker_share_mean, marker="o", color=COLORS[m], label=m)
        for att, ls in [("none", "-"), ("sign_flip", "--")]:
            e = ep[(ep.method == m) & (ep.attack == att)].copy()
            if not e.empty:
                e["e"] = e["eps"].astype(float); e = e.sort_values("e")
                axes[2].plot(e["e"], e.test_macro_f1, marker="o", ls=ls, color=COLORS[m], label=f"{m}, {att}")
    axes[1].plot([10, 40], [0.1, 0.4], color="k", ls=":", lw=0.8, label="fair share")
    axes[0].set_xlabel("attackers (% of clients, sign flip)"); axes[0].set_ylabel("test macro-F1")
    axes[1].set_xlabel("attackers (% of clients, sign flip)"); axes[1].set_ylabel("attackers' weight share")
    axes[2].set_xlabel("epsilon per client"); axes[2].set_ylabel("test macro-F1"); axes[2].set_xscale("log")
    for ax in axes:
        ax.grid(color="#eee"); ax.legend(frameon=False, fontsize=7)
    fig.suptitle("E7 stress - K = 10, alpha 0.5, seed 42 (single seed: read trends, not small differences)", fontsize=10)
    fig.tight_layout(); fig.savefig(FIG / "e7_stress.png", dpi=150); plt.close(fig)
    print("figure -> results/figures/e7_stress.png, tables -> results/e7_stress/")


GROUPS = {"b1": b1, "b2": b2, "tune": tune, "e5": e5, "mixed": mixed, "e4": e4, "e6": e6, "e7": e7}


def main():
    ap = argparse.ArgumentParser(description="With no option, every figure is made.")
    for g in GROUPS:
        ap.add_argument(f"--{g}", action="store_true")
    args = ap.parse_args()
    master = pd.read_csv(RES / "master.csv")
    chosen = [g for g in GROUPS if getattr(args, g)] or list(GROUPS)
    for g in chosen:
        print(f"===== {g}")
        GROUPS[g](master)


if __name__ == "__main__":
    main()
