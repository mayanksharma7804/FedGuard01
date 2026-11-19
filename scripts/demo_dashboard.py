"""Live trust-weight dashboard for the FedGuard demo (plan section 13.3, step 3).

    python scripts/demo_dashboard.py                                  # follow the live Flower demo (results/demo)
    python scripts/demo_dashboard.py --dir results/demo_autogm        # another demo output folder
    python scripts/demo_dashboard.py --run results/runs/<run name>    # replay a finished engine run (backup)
    python scripts/demo_dashboard.py --run results/runs/<run> --save results/figures/x.png   # one image, no window

Left: every client's trust weight per round (attackers red, the most private clients blue, others grey;
dashed lines = fair share 1/K and the honest-rejection threshold 0.5/K). Top right: the latest round's
weights as bars. Bottom right: validation macro-F1 of the global model.
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import yaml
from matplotlib.animation import FuncAnimation

REPO = Path(__file__).resolve().parents[1]


def read_csv(path):
    try:
        return pd.read_csv(path) if Path(path).exists() else pd.DataFrame()
    except (pd.errors.EmptyDataError, PermissionError, OSError):
        return pd.DataFrame()                       # the file is being replaced right now - try next tick


class Source:
    """Reads either a Flower demo folder or a finished engine run folder."""

    def __init__(self, folder: Path, engine_run: bool):
        self.folder, self.engine = folder, engine_run

    def clients(self):
        if self.engine:
            cfg = yaml.safe_load(open(self.folder / "config.yaml"))
            test = json.load(open(self.folder / "metrics_test.json"))
            k = len(test["client_sizes"])
            eps = test.get("epsilon_target_per_client") or [None] * k
            att = set(test.get("attackers") or [])
            title = (f"{cfg['fl']['aggregator']['name']}  attack={cfg['attack']['type']}\n"
                     f"(run {self.folder.name})")
            return [{"client": c, "n": test["client_sizes"][c], "epsilon": eps[c], "attacker": c in att}
                    for c in range(k)], title
        info_path = self.folder / "clients.json"
        if not info_path.exists():
            return [], "waiting for the Flower server to start ..."
        info = json.load(open(info_path))
        agg = info["aggregator"]["name"]
        return info["clients"], f"{agg}  attack={info['attack']['type']}  (live Flower run)"

    def weights(self):
        return read_csv(self.folder / ("weights_log.csv" if self.engine else "flower_weights.csv"))

    def rounds(self):
        return read_csv(self.folder / ("rounds.csv" if self.engine else "flower_rounds.csv"))


def label(c):
    eps = "no DP" if c["epsilon"] is None else f"eps {c['epsilon']:g}"
    return f"client {c['client']} ({eps}{', ATTACKER' if c['attacker'] else ''})"


def color(c, min_eps):
    if c["attacker"]:
        return "#d62728"
    if c["epsilon"] is not None and c["epsilon"] == min_eps:
        return "#1f77b4"
    return "#7f7f7f"


def draw(src, axes):
    ax_w, ax_b, ax_f = axes
    for ax in axes:
        ax.clear()
    clients, title = src.clients()
    w, r = src.weights(), src.rounds()
    if not clients:
        ax_w.set_title(title)
        return
    k = len(clients)
    eps_values = [c["epsilon"] for c in clients if c["epsilon"] is not None]
    min_eps = min(eps_values) if eps_values and len(set(eps_values)) > 1 else None
    if not w.empty:
        for c in clients:
            s = w[w["client"] == c["client"]].sort_values("round")
            ax_w.plot(s["round"], s["weight"], color=color(c, min_eps), marker="o", ms=3,
                      lw=2.5 if c["attacker"] or c["epsilon"] == min_eps else 1.0,
                      ls="--" if c["attacker"] else "-", label=label(c))
        last = w[w["round"] == w["round"].max()].set_index("client")["weight"]
        vals = [last.get(c["client"], 0.0) for c in clients]
        bars = ax_b.bar([str(c["client"]) for c in clients], vals, color=[color(c, min_eps) for c in clients])
        for b, v in zip(bars, vals):
            ax_b.text(b.get_x() + b.get_width() / 2, v + 0.005, f"{v:.2f}", ha="center", fontsize=7)
        ax_b.set_title(f"trust weights in round {int(w['round'].max())}")
    for ax in (ax_w, ax_b):
        ax.axhline(1.0 / k, color="k", lw=0.8, ls=":")
        ax.axhline(0.5 / k, color="k", lw=0.8, ls="--")
        ax.set_ylim(0, max(0.35, (w["weight"].max() * 1.1) if not w.empty else 0.35))
    ax_w.set_xlabel("round"); ax_w.set_ylabel("trust weight")
    ax_w.xaxis.get_major_locator().set_params(integer=True)
    ax_w.set_title(title, fontsize=10)
    ax_w.legend(fontsize=7, loc="upper right", ncol=1 if k <= 6 else 2)
    ax_b.set_xlabel("client")
    if not r.empty and "val_macro_f1" in r:
        ax_f.plot(r["round"], r["val_macro_f1"], color="#2ca02c", marker="o", ms=3)
        ax_f.set_title(f"validation macro-F1 (last {r['val_macro_f1'].iloc[-1]:.3f})")
    ax_f.set_xlabel("round"); ax_f.set_ylim(0, max(0.4, r["val_macro_f1"].max() * 1.1 if not r.empty else 0.4))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/demo", help="Flower demo output folder (live)")
    ap.add_argument("--run", help="engine run folder to replay instead (results/runs/<run>)")
    ap.add_argument("--interval", type=float, default=2.0, help="seconds between refreshes")
    ap.add_argument("--save", help="write one PNG and exit (no window)")
    args = ap.parse_args()
    folder = REPO / (args.run or args.dir)
    src = Source(folder, engine_run=bool(args.run))

    fig = plt.figure(figsize=(13, 6.5))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.6, 1])
    axes = (fig.add_subplot(gs[:, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 1]))
    fig.suptitle("Who does the server trust?  (dotted: fair share 1/K, dashed: rejection threshold 0.5/K)",
                 fontsize=11)
    if args.save:
        draw(src, axes)
        fig.tight_layout()
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.save, dpi=150)
        print("saved", args.save)
        return
    _anim = FuncAnimation(fig, lambda _i: draw(src, axes), interval=args.interval * 1000, cache_frame_data=False)
    plt.show()


if __name__ == "__main__":
    main()
