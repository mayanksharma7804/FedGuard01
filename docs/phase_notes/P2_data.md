# Phase 2 note - Data pipeline

**Dates:** 7 Oct 2026  |  **Lead:** R1 (Data)  |  **Objectives:** O1, O5

## What we did
- Placed both datasets in `data/raw/` and recorded sources and hashes in `data/README.md`
  (NF-UNSW-NB15-v2 SHA-1 matches the official UQ manifest).
- Wrote the leakage-safe pipeline `src/fedguard/data.py` + `scripts/prepare_data.py`.
- Wrote the Dirichlet partitioner `src/fedguard/partition.py` + `scripts/make_partitions.py`;
  made 18 partition files (alpha {0.1, 0.5, 100} x K {5, 10} x seeds {42, 43, 44}).
- EDA notebook `notebooks/01_eda.ipynb` (executed, outputs saved) + figures `results/figures/eda_*.png`, `partitions_*.png`.
- Tests: `tests/test_partition.py`, `tests/test_data_pipeline.py`, `tests/test_processed_data.py` - 27 passing.

## Numbers we got
| Step | Rows |
|---|---|
| Raw | 2,390,275 |
| Exact duplicates after dropping identifiers | 2,282,035 (95.5%) |
| Feature vectors with conflicting labels | 1,563 (majority label kept) |
| Unique rows | 104,022 (38 features) |
| Train / val / test | 45,225 / 8,322 / 20,805 (train before the 20k cap: 74,895) |

Training classes after cap: Benign 20,000 · Exploits 15,734 · Fuzzers 4,856 · DoS 2,234 · Generic 1,241 ·
Analysis 524 · Reconnaissance 246 · Backdoor 200 · Shellcode 154 · Worms 36.
Processed-data fingerprint: `7d2ba3260f5740ab` (partition files are tied to it).

## What went wrong / surprises
- **95.5% of the dataset is duplicates** once IPs, ports and `DNS_QUERY_ID` are removed. Without
  de-duplication the same flow would sit in train and test. Published results that skip this
  (e.g. ref [1], 91-93%) are not directly comparable with ours - say so in the report.
- **`DNS_QUERY_ID` is an identifier** (random transaction number, 64k distinct values). Keeping it
  hid 131k duplicate benign rows. Dropped together with IPs and ports.
- **All attacks come from 4 of the 40 source IPs** - IP columns would let the model cheat.
- **The leakage test caught 1 test row identical to a train row** after `log1p` + float32 (two raw
  values became the same float). Fix: apply `log1p` as float32 *before* de-duplication (a fixed
  formula, so no leakage). Now no row is shared between train, val and test.
- Rare classes are tiny after cleaning (Worms: 36 train / 4 val / 10 test rows). Their per-class
  scores will be noisy; report them, do not drop them.
- With alpha = 0.1 and 10 clients the smallest client has 215 rows (min_size = 200).

## Decisions taken (and why)
- Drop `IPV4_SRC_ADDR`, `IPV4_DST_ADDR`, `L4_SRC_PORT`, `L4_DST_PORT`, `DNS_QUERY_ID` (identifiers) and `Label` (twin of `Attack`).
- Conflicting labels: keep the majority label per feature vector (ties: alphabetical), keeping rare-class rows instead of deleting them.
- Benign = class 0 (the label-flip attack in Phase 6 flips everything to class 0).
- Train cap 20,000 rows per class; validation and test keep the natural distribution.
- Training is ~45k rows, smaller than the plan's estimate, so each round is ~177 steps (batch 256): runs will be faster than the Phase 1 estimate.

## Next steps
- Phase 3: CNN-LSTM, training loop, metrics, centralised baseline B0 (3 seeds) on `nf_unsw.npz`.
- Phase 8 (E8): CICIoT2023 - check duplicates across the Kaggle train/val/test files first.
