"""Phase 2: make every non-IID client partition the experiments need, plus a plot.

    python scripts/make_partitions.py            # alpha {0.1, 0.5, 100} x K {5, 10} x seeds {42, 43, 44}
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from fedguard.data import load_processed
from fedguard.partition import dirichlet_partition, load_partition, partition_path, save_partition
from fedguard.utils import REPO_ROOT


def plot(dataset, classes, y, n_clients, seed, alphas):
    cmap = plt.get_cmap("tab10")
    fig, axs = plt.subplots(1, len(alphas), figsize=(4.2 * len(alphas), 0.45 * n_clients + 1.6), sharey=True)
    for ax, a in zip(axs, alphas):
        parts, _ = load_partition(partition_path(dataset, a, n_clients, seed))
        for k, p in enumerate(parts):
            frac = np.bincount(y[p], minlength=len(classes)) / len(p)
            ax.barh(k, frac, left=np.r_[0, np.cumsum(frac)[:-1]], color=[cmap(i) for i in range(len(classes))], height=0.75)
            ax.text(1.01, k, f"{len(p):,}", va="center", fontsize=7)
        ax.set_title(f"alpha = {a:g}", fontsize=10); ax.set_xlim(0, 1.18)
        ax.set_xlabel("share of each class in the client's data", fontsize=8)
    axs[0].set_yticks(range(n_clients)); axs[0].set_yticklabels([f"client {k}" for k in range(n_clients)])
    handles = [plt.Rectangle((0, 0), 1, 1, color=cmap(i)) for i in range(len(classes))]
    fig.legend(handles, classes, loc="upper center", ncol=min(len(classes), 5), fontsize=8, frameon=False,
               bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(f"{dataset}: Dirichlet partitions, {n_clients} clients, seed {seed} (numbers = rows per client)", fontsize=10)
    out = REPO_ROOT / "results" / "figures" / f"partitions_{dataset}_k{n_clients}_seed{seed}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="nf_unsw")
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.1, 0.5, 100])
    ap.add_argument("--clients", type=int, nargs="+", default=[5, 10])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--min-size", type=int, default=200)
    args = ap.parse_args()

    data = load_processed(args.dataset)
    y = data.y_tr
    print(f"{args.dataset}: {len(y):,} training rows, {data.n_classes} classes, fingerprint {data.fingerprint}")
    print(f"{'file':38s} {'smallest':>9s} {'largest':>9s}  classes/client (min-max)")
    for k in args.clients:
        for a in args.alphas:
            for s in args.seeds:
                parts = dirichlet_partition(y, k, a, min_size=args.min_size, seed=s)
                path = save_partition(parts, y, data.n_classes, dataset=args.dataset, alpha=a, seed=s,
                                      min_size=args.min_size, fingerprint=data.fingerprint)
                sizes = [len(p) for p in parts]
                ncls = [len(np.unique(y[p])) for p in parts]
                print(f"{path.name:38s} {min(sizes):9,} {max(sizes):9,}  {min(ncls)}-{max(ncls)}")
    for k in args.clients:
        print("plot:", plot(args.dataset, data.classes, y, k, args.seeds[0], args.alphas).relative_to(REPO_ROOT))


if __name__ == "__main__":
    main()
