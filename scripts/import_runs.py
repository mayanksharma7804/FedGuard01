"""Import finished runs exported on another machine (scripts/export_runs.py) into results/runs/.

    python scripts/import_runs.py results/exports/runs_LAPTOP2_20261118-0930.zip [more zips ...]
    python scripts/import_runs.py --check results/exports/runs_kaggle_20261118-2100.zip   # verify only, copy nothing

Every run is checked before it is copied:
  - all result files are present and metrics_test.json is complete;
  - the folder name equals the name computed from its own config.yaml (the config was not edited or mixed up);
  - it was trained on exactly this processed dataset (data fingerprint) - otherwise it is refused;
  - torch / opacus versions are reported if they differ from this laptop (allowed: same pinned code, other GPU).
A run that is already finished here is never overwritten; an unfinished local copy (checkpoint only) is
replaced by the finished import. Afterwards: python scripts/collect_results.py
"""
import argparse
import json
import shutil
import sys
import zipfile
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_experiments import run_name  # noqa: E402  (same naming rule as the runner)

from fedguard.data import load_processed  # noqa: E402
from fedguard.utils import REPO_ROOT  # noqa: E402

RUNS = REPO_ROOT / "results" / "runs"
REQUIRED = ("config.yaml", "metrics_test.json", "metrics_val.json", "rounds.csv", "weights_log.csv")


def check_run(files: dict, name: str, fingerprint: str):
    """files: {filename: bytes}. Returns (ok, message)."""
    missing = [f for f in REQUIRED if f not in files]
    if missing:
        return False, f"missing {missing}"
    cfg = yaml.safe_load(files["config.yaml"])
    if run_name(cfg) != name:
        return False, f"name does not match its config (expected {run_name(cfg)})"
    test = json.loads(files["metrics_test.json"])
    if test.get("data_fingerprint") != fingerprint:
        return False, f"trained on other data (fingerprint {test.get('data_fingerprint')} != {fingerprint})"
    meta = json.loads(files.get("meta.json", b"{}"))
    note = ""
    if meta.get("torch") and meta["torch"].split("+")[0] != torch.__version__.split("+")[0]:
        note = f" (note: torch {meta['torch']} there, {torch.__version__} here)"
    return True, "ok" + note


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("zips", nargs="+")
    ap.add_argument("--check", action="store_true", help="verify only, copy nothing")
    args = ap.parse_args()
    fingerprint = load_processed("nf_unsw").fingerprint
    added = skipped = refused = 0
    for zpath in args.zips:
        with zipfile.ZipFile(zpath) as z:
            groups = {}
            for info in z.infolist():
                parts = Path(info.filename).parts
                if len(parts) != 3 or parts[0] != "runs" or info.is_dir():
                    continue                                            # DONE_RUNS.txt etc.
                if any(p in ("..", "") or ":" in p for p in parts):     # never write outside results/runs
                    continue
                groups.setdefault(parts[1], {})[parts[2]] = z.read(info)
        print(f"{zpath}: {len(groups)} runs")
        for name, files in sorted(groups.items()):
            dest = RUNS / name
            if (dest / "metrics_test.json").exists():
                skipped += 1
                continue
            ok, msg = check_run(files, name, fingerprint)
            if not ok:
                refused += 1
                print(f"  REFUSED {name}: {msg}")
                continue
            if msg != "ok":
                print(f"  {name}: {msg}")
            if args.check:
                added += 1
                continue
            if dest.exists():
                shutil.rmtree(dest)                     # unfinished local copy (checkpoint only)
            dest.mkdir(parents=True)
            for f, data in files.items():
                (dest / f).write_bytes(data)
            added += 1
    verb = "would import" if args.check else "imported"
    print(f"{verb} {added}, already here {skipped}, refused {refused}")
    if added and not args.check:
        print("next: python scripts/collect_results.py && python scripts/make_plots.py")
    sys.exit(1 if refused else 0)


if __name__ == "__main__":
    main()
