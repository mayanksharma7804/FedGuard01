"""Phase 2: raw CSV -> leakage-safe train/val/test arrays (data/processed/<dataset>.npz + manifest).

    python scripts/prepare_data.py                    # NF-UNSW-NB15-v2, cap 20k/class, seed 42
    python scripts/prepare_data.py --cap 20000 --seed 42
"""
import argparse
import json
import time

from fedguard.data import DATASETS, prepare


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="nf_unsw", choices=sorted(DATASETS))
    ap.add_argument("--cap", type=int, default=20_000, help="max training rows per class")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    t0 = time.time()
    m = prepare(DATASETS[args.dataset], cap=args.cap, seed=args.seed)
    c = m["cleaning"]
    print(f"dataset            : {m['dataset']}  (fingerprint {m['fingerprint']})")
    print(f"raw rows           : {c['rows_raw']:,}")
    print(f"exact duplicates   : {c['rows_exact_duplicates']:,}  (after dropping {', '.join(c['dropped_columns'])})")
    print(f"conflicting vectors: {c['vectors_with_conflicting_labels']:,}  (kept the majority label)")
    print(f"unique rows        : {c['rows_after_dedup']:,}")
    print(f"features           : {len(m['features'])}")
    print(f"rows train/val/test: {m['rows']['train']:,} / {m['rows']['val']:,} / {m['rows']['test']:,}"
          f"  (train before cap {m['rows']['train_before_cap']:,})")
    print("\nclass counts (train / val / test):")
    for cls in m["classes"]:
        n = [m["class_counts"][s].get(cls, 0) for s in ("train", "val", "test")]
        print(f"  {cls:15s} {n[0]:7,} {n[1]:7,} {n[2]:7,}")
    print(f"\ndone in {time.time() - t0:.0f}s -> data/processed/{m['dataset']}.npz")


if __name__ == "__main__":
    main()
