"""Experiment E0 (plan 10.1, step 5): does the server's noise-radius formula predict the real DP noise?

    python scripts/e0_calibration.py                     # sigma {0.5,1,2} x 5 seeds x 5 clients x 2 start models
    python scripts/e0_calibration.py --clip 1.0 --seeds 5

For every (start model, client, seed): train the client once with sigma = 0 (clipping only) and once with
each sigma, from the SAME global model with the SAME Poisson batches and dropout masks (only the noise
generator differs, fedguard.dp.make_private). Then
    measured  = ||delta_noisy - delta_clean||
    predicted = lr * sigma * C * sqrt(T * d) / B          (fedguard.dp.noise_radius)
Ratio measured/predicted ~ 1 means the formula holds; if it is consistently off we fit one factor kappa.
Also logged: cos(clean update, noise part) - near 0 means noise is ~90 degrees to the signal (refinement R2).

Start models: 'init' = random initialisation (round 1), 'warm' = global model after 5 FedAvg rounds (no DP).
Writes results/e0_calibration/e0_pairs.csv, e0_table.csv and results/figures/e0_*.png.
"""
import argparse
import copy
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from fedguard.config import load_config
from fedguard.data import load_processed
from fedguard.fl_sim import FLSimulation, flat
from fedguard.partition import load_partition, partition_path
from fedguard.train import get_device
from fedguard.utils import REPO_ROOT, set_seed

OUT = REPO_ROOT / "results" / "e0_calibration"
FIG = REPO_ROOT / "results" / "figures"
SIGMAS = [0.5, 1.0, 2.0]


def make_sim(cfg, data, parts, device):
    return FLSimulation(cfg, data, parts, device, log=lambda *a: None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/b2_dpfedavg.yaml")
    ap.add_argument("--clip", type=float, default=None, help="default: dp.clip of the config")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--warm-rounds", type=int, default=5)
    args = ap.parse_args()

    cfg = load_config(REPO_ROOT / args.config)
    clip = args.clip if args.clip is not None else float(cfg["dp"]["clip"])
    set_seed(cfg["seed"])
    device = get_device(cfg["train"]["device"])
    data = load_processed(cfg["dataset"])
    f = cfg["fl"]
    parts, _ = load_partition(partition_path(cfg["dataset"], f["alpha"], f["clients"], cfg["seed"]),
                              fingerprint=data.fingerprint)

    # start models: random init, and the global model after a few plain FedAvg rounds
    plain = copy.deepcopy(cfg); plain["dp"] = {"epsilon": None}
    sim0 = make_sim(plain, data, parts, device)
    starts = {"init": flat(sim0.global_model)}
    warm = copy.deepcopy(plain); warm["fl"].update(max_rounds=args.warm_rounds, min_rounds=args.warm_rounds)
    t0 = time.perf_counter()
    wm, *_ = make_sim(warm, data, parts, device).run()
    starts["warm"] = flat(wm)
    print(f"warm start: {args.warm_rounds} FedAvg rounds in {time.perf_counter() - t0:.0f}s")

    dpcfg = copy.deepcopy(cfg); dpcfg["dp"] = {"epsilon": 3.0, "clip": clip}   # only to switch DP-SGD on
    sim = make_sim(dpcfg, data, parts, device)
    rows = []
    for start, g in starts.items():
        for c in sim.clients:
            for s in range(1, args.seeds + 1):            # 'seed' = round index -> new batches, dropout, noise
                c.meta["sigma"] = 0.0
                clean = sim.local_train(c, g, rnd=1000 + s)["delta"]
                for sigma in SIGMAS:
                    c.meta["sigma"] = sigma
                    out = sim.local_train(c, g, rnd=1000 + s)
                    noise = out["delta"] - clean
                    pred = sim.noise_radius(out["meta"])
                    meas = float(noise.norm())
                    cos = float(torch.dot(noise, clean) / (noise.norm() * clean.norm() + 1e-12))
                    rows.append({"start": start, "client": c.cid, "n": c.n, "T": out["meta"]["T"], "seed": s,
                                 "sigma": sigma, "C": clip, "predicted": pred, "measured": meas,
                                 "ratio": meas / pred, "clean_norm": float(clean.norm()),
                                 "noisy_norm": float(out["delta"].norm()), "cos_noise_clean": cos})
                print(f"  {start:4s} client {c.cid} (n={c.n:5d}, T={out['meta']['T']:4d}) seed {s}  ratios "
                      + " ".join(f"{r['ratio']:.3f}" for r in rows[-len(SIGMAS):]), flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "e0_pairs.csv", index=False, float_format="%.6g")
    table = (df.groupby(["start", "sigma"])
               .agg(predicted_mean=("predicted", "mean"), measured_mean=("measured", "mean"),
                    measured_std=("measured", "std"), ratio_mean=("ratio", "mean"), ratio_std=("ratio", "std"),
                    ratio_min=("ratio", "min"), ratio_max=("ratio", "max"), cos_mean=("cos_noise_clean", "mean"),
                    snr_clean_over_r=("clean_norm", "mean"), pairs=("ratio", "size")).reset_index())
    table["snr_clean_over_r"] = table["snr_clean_over_r"] / table["predicted_mean"]
    # least-squares kappa through the origin: measured ~ kappa * predicted
    kappa = {st: float((g.measured * g.predicted).sum() / (g.predicted ** 2).sum()) for st, g in df.groupby("start")}
    table["kappa_start"] = table["start"].map(kappa)
    table.to_csv(OUT / "e0_table.csv", index=False, float_format="%.4f")
    print("\nE0 table (mean over clients x seeds):")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("kappa (least squares) per start model:", {k: round(v, 4) for k, v in kappa.items()})

    # figures: predicted vs measured scatter, and ratio vs T
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    colors = {0.5: "#2A9D8F", 1.0: "#6D597A", 2.0: "#E76F51"}
    for (st, sigma), g in df.groupby(["start", "sigma"]):
        axes[0].scatter(g.predicted, g.measured, s=14, color=colors[sigma], marker="o" if st == "init" else "^",
                        alpha=0.75, label=f"sigma={sigma:g}, {st}")
    lim = [0, df[["predicted", "measured"]].to_numpy().max() * 1.05]
    axes[0].plot(lim, lim, "k--", lw=1, label="measured = predicted")
    axes[0].set_xlabel("predicted radius  lr*sigma*C*sqrt(T*d)/B"); axes[0].set_ylabel("measured ||noisy - clean||")
    axes[0].legend(fontsize=7, frameon=False); axes[0].set_title("E0: noise-radius formula vs measurement", fontsize=10)
    for (st, sigma), g in df.groupby(["start", "sigma"]):
        m = g.groupby("T").ratio.mean()
        axes[1].plot(m.index, m.values, marker="o" if st == "init" else "^", color=colors[sigma],
                     ls="-" if st == "init" else ":", label=f"sigma={sigma:g}, {st}")
    axes[1].axhline(1, color="k", lw=1, ls="--"); axes[1].set_xscale("log")
    axes[1].set_xlabel("local steps T per round (one point per client)"); axes[1].set_ylabel("measured / predicted")
    axes[1].legend(fontsize=7, frameon=False); axes[1].set_title("Ratio by number of local steps", fontsize=10)
    fig.tight_layout(); FIG.mkdir(parents=True, exist_ok=True); fig.savefig(FIG / "e0_calibration.png", dpi=150)
    print("-> results/e0_calibration/, results/figures/e0_calibration.png")


if __name__ == "__main__":
    main()
