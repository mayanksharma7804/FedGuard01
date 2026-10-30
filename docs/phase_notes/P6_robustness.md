# Phase 6 — Byzantine Robustness Baselines

## What we did
Implemented AutoGM (geometric median with lam_scale=4) as a Byzantine-robust aggregator and ran a full attack grid (none / label_flip / sign_flip / gaussian, fraction=0.3) across α ∈ {0.1, 0.5, 100} and seeds 42-44. Two privacy settings: no-DP (ε=∞) and DP ε=3. (ε=8 was planned but not run — ε=8 behaves similarly to ε=3 per B2 findings, F38.)

## Key findings
- **HRR definition**: HRR = honest rejection rate = share of honest clients whose trust weight falls below 0.5/K. High HRR means the aggregator penalises clean clients.
- **AutoGM vs FedAvg under attack**: AutoGM's geometric median down-weights outlier updates. On sign_flip at α=0.5, attacker weight share drops toward zero. However, macro-F1 recovery is modest because DP noise (large σ at ε=3) already degrades all model quality.
- **label_flip hardest**: All aggregators fail on label_flip because corrupted gradients point in the correct direction — neither geometric median nor FedAvg can distinguish them.
- **B4 baseline (apples-to-apples 20-round α=0.5 ε=3)**: AutoGM macro-F1 = 0.148, HRR near 0 on sign_flip (honest clients protected). FedAvg = 0.172 on clean data.

## Plots
- `results/figures/p6_attacks.png` — attack comparison: FedAvg vs AutoGM across attack types
- `results/figures/e4_conflict_autogm.png`, `e4_conflict_fedavg.png` — ε×α heatmaps (HRR, macro-F1)
