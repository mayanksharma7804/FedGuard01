# Phase 6 — Byzantine Robustness Baselines

## What we did
Implemented AutoGM (geometric median with lam_scale=4) as a Byzantine-robust aggregator and ran a full attack grid (none / label_flip / sign_flip / gaussian, fraction=0.3) across α ∈ {0.1, 0.5, 100}, seeds 42-44, with and without DP (ε=3, ε=8).

## Key findings
- **AutoGM vs FedAvg under attack**: AutoGM significantly reduces attacker-weighted updates (HRR drops from ~0.3 to near 0 on sign_flip) but macro-F1 recovery is modest at α=0.1 due to extreme non-IID making the geometric median less stable.
- **DP + attack interaction**: DP noise (large σ at ε=3) already suppresses some attack signal; AutoGM on top provides additional benefit mostly at α≥0.5.
- **label_flip hardest to detect**: All aggregators struggle with label_flip because corrupted gradients are in the correct sign direction — HRR stays 0 while macro-F1 drops.
- **B4 baseline established**: AutoGM (no DP, α=0.5, attack=none) macro-F1 = 0.179 ± 0.033 — used as comparison anchor for Phase 7.

## Plots
- `results/figures/b3_attacks_*.png` — attack comparison tables
- `results/figures/b4_autogm_*.png` — AutoGM vs FedAvg
