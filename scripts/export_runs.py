"""Pack finished runs into one zip so they can be moved to another machine (multi-laptop / Kaggle sharding).

    python scripts/export_runs.py                      # all finished runs -> results/exports/runs_<host>_<time>.zip
    python scripts/export_runs.py --done-list-only     # only results/exports/done_<host>.txt (run names)
    python scripts/export_runs.py --publish-done       # add this machine's finished runs to results/done_runs.txt
                                                       # (tracked in git: every machine skips those runs)

The zip holds each finished run folder (config, metrics, per-round logs - a few hundred KB per run, never the
*.pt checkpoints) plus DONE_RUNS.txt. A done list (one run name per line) is also written next to the zip: give
it to the other machines, which skip those runs with  run_experiments.py --skip-file <done list>.
Import on the main laptop with scripts/import_runs.py.
"""
import argparse
import os
import platform
import time
import zipfile
from pathlib import Path

from fedguard.utils import REPO_ROOT

RUNS = REPO_ROOT / "results" / "runs"
EXPORTS = REPO_ROOT / "results" / "exports"
KEEP = ("config.yaml", "metrics_test.json", "metrics_val.json", "meta.json", "rounds.csv", "weights_log.csv",
        "train.log")


def finished_runs():
    """Top-level federated run folders that are complete (metrics_test.json is written last)."""
    return sorted(d for d in RUNS.iterdir()
                  if d.is_dir() and not d.name.startswith(("_", "smoke"))
                  and (d / "metrics_test.json").exists() and (d / "rounds.csv").exists())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", help="zip path (default results/exports/runs_<host>_<time>.zip)")
    ap.add_argument("--done-list-only", action="store_true", help="write only the list of finished run names")
    ap.add_argument("--publish-done", action="store_true",
                    help="merge the finished run names into results/done_runs.txt (commit + push it)")
    args = ap.parse_args()
    EXPORTS.mkdir(parents=True, exist_ok=True)
    label = os.environ.get("FEDGUARD_MACHINE") or platform.node()
    host = "".join(ch for ch in label if ch.isalnum() or ch in "-_") or "host"
    runs = finished_runs()
    names = "\n".join(d.name for d in runs) + "\n"
    done_path = EXPORTS / f"done_{host}.txt"
    done_path.write_text(names, encoding="utf-8")
    print(f"{len(runs)} finished runs -> done list {done_path.relative_to(REPO_ROOT)}")
    if args.publish_done:
        shared = REPO_ROOT / "results" / "done_runs.txt"
        old = {ln.strip() for ln in open(shared, encoding="utf-8")} if shared.exists() else set()
        allnames = sorted((old | {d.name for d in runs}) - {""})
        kept = [n for n in allnames if not n.startswith("#")]
        header = "# finished runs on any machine (scripts/export_runs.py --publish-done); skipped everywhere"
        shared.write_text("\n".join([header] + kept) + "\n", encoding="utf-8")
        print(f"results/done_runs.txt now lists {len(kept)} runs - commit and push it")
    if args.done_list_only or args.publish_done:
        return
    out = Path(args.out) if args.out else EXPORTS / f"runs_{host}_{time.strftime('%Y%m%d-%H%M')}.zip"
    out = out if out.is_absolute() else REPO_ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("DONE_RUNS.txt", names)
        for d in runs:
            for f in KEEP:
                if (d / f).exists():
                    z.write(d / f, f"runs/{d.name}/{f}")
    print(f"{len(runs)} runs -> {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
