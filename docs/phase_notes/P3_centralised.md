# Phase 3 note - Centralised CNN-LSTM baseline (B0)

**Dates:** 7 Oct 2026  |  **Lead:** R2 (Model + DP)  |  **Objective:** O1 (accuracy ceiling)

## What we did
- `src/fedguard/model.py`: CNN-LSTM with Opacus `DPLSTM` (40,266 parameters, Opacus validator OK).
- `src/fedguard/metrics.py`: macro-F1, weighted-F1, accuracy, per-class precision/recall/F1, confusion
  matrix, **benign FPR**, **attack detection rate** and **binary F1** (benign vs any attack), plus HRR and
  attacker weight share for later phases.
- `src/fedguard/train.py`: plain SGD (momentum 0, no weight decay), CrossEntropy, batch 256, early stopping on
  **validation** macro-F1 with `min_epochs`, optional class weights, best-epoch restore.
- `src/fedguard/config.py` (YAML + overrides + run id), `configs/b0_centralised.yaml`,
  `scripts/run_centralised.py` (skips finished runs, writes a live `train.log`), `scripts/compare_runs.py`.
- Tests: `test_model.py`, `test_metrics.py`, `test_train.py` (42 tests in total pass).
- Hyperparameter selection on the **validation split only** (seed 42): class weight x learning rate grid.
- Final B0: seeds 42, 43, 44.

## Selection (validation macro-F1, seed 42) - `results/b0_centralised/selection_grid_val.csv`
| class weight \ lr | 0.05 | 0.1 | 0.2 |
|---|---|---|---|
| none | 0.590 | 0.625 | **0.632** |
| sqrt | 0.555 | 0.565 | 0.586 |
| balanced | 0.469 | 0.506 | 0.484 |

Chosen: **no class weights, lr 0.2**, max 300 epochs, patience 30, min 20 epochs.

## Final B0 result (test set, 3 seeds) - `results/b0_centralised/summary.csv`
| Metric | Mean +/- std |
|---|---|
| **Macro-F1** | **0.597 +/- 0.004** |
| Weighted-F1 | 0.896 +/- 0.001 |
| Accuracy | 0.898 +/- 0.002 |
| Benign FPR | 0.068 +/- 0.005 |
| Attack detection rate | 0.950 +/- 0.008 |
| Binary F1 | 0.912 +/- 0.000 |
| Best epoch | 107-180 (mean 140); ~4.5 s/epoch |

Per-class test recall (mean): Benign 0.93 · Exploits 0.93 · Fuzzers 0.88 · Generic 0.62 · Reconnaissance 0.57 ·
Analysis 0.39 · DoS 0.36 · Shellcode 0.35 · Backdoor 0.32 · Worms 0.30 (`per_class.csv`).

Figures: `results/figures/b0_centralised_confusion.png`, `..._per_class_recall.png`, `..._learning_curves.png`.

## What went wrong / surprises
- Early stopping with patience 10 stopped plain SGD too early and made the first class-weight comparison
  unfair; after the fix, the conclusion reversed (sqrt looked best, "none" actually is). Findings F18-F19.
- Class weights raise rare-class recall but cost precision and macro-F1 (F20).
- Long parallel runs heat the laptop (GPU 86 C) and slow epochs up to ~3x (F21).
- Validation macro-F1 (~0.63) is a little higher than test (~0.60): the expected optimism from choosing
  settings on validation. The test set was never used for any choice.

## Decisions taken
- One setting for every later baseline and every client: plain SGD, lr 0.2, batch 256, no class weights.
- B0 is the ceiling: federated (B1) and private (B2) runs will be compared against 0.597 macro-F1.

## Next steps (Phase 4)
- Federated engine `fl_sim.py`, run manager with `--resume`, FedAvg (B1) for alpha {0.1, 0.5, 100} x 3 seeds,
  Flower parity check.
