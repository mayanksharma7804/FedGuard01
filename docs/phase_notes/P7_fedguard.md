# Phase 7 — FedGuard Full Attack Grid (B5)

## What we did
Ran FedGuard (quadrature mode, k_mad=2.0, lam_scale=1.0, noise_tol_z=3.0, outer=20) against all 4 attack types (none / label_flip / sign_flip / gaussian at fraction=0.3) across seeds 42-44, with DP ε=3, α=0.5, K=10.

## FedGuard components (from src/fedguard/aggregators.py)
- **R1 (r_eff)**: The weighted centre z is itself DP-noisy, so the noise in (x_k − z) is larger than r_k alone. R1 computes r_eff = sqrt((1−b_k)²·r_k² + cross-term) to account for this.
- **R2 (mode)**: `quadrature`: residual = sqrt(dist² − r_eff²); `linear`: residual = dist − r_eff. Quadrature treats noise as approximately orthogonal to the signal in high dimension.
- **R3 (MAD allowance)**: Outlier threshold s = median(residual) + k_mad · 1.4826 · MAD(residual). Clients with excess = max(0, residual − s) > 0 are down-weighted.
- **R4 (noise_tol_z)**: Also subtracts z standard-deviations of the DP noise *length* itself before computing excess, since ||noise|| concentrates around r_eff with std ≈ r_eff/√(2d).

**HRR** = honest rejection rate: share of honest clients whose trust weight falls below 0.5/K (half of a fair share). High HRR means FedGuard is incorrectly penalising clean clients.

## Key results (apples-to-apples: 20 rounds, α=0.5, ε=3, K=10, 30% attackers, 3 seeds, test macro-F1 mean)

| attack | FedAvg+DP | AutoGM λ4 | FedGuard |
|---|---|---|---|
| none | 0.172 | 0.148 | 0.122 |
| label_flip | 0.125 | 0.080 | 0.080 |
| sign_flip | 0.122 | 0.077 | 0.117 |
| gaussian | 0.108 | 0.152 | 0.176 |

FedGuard's clear win is on **gaussian attack** (+0.068 vs FedAvg+DP, all 3 seeds). On sign_flip the gain over AutoGM (+0.040) comes from a single seed where AutoGM collapsed. On none/label_flip FedGuard is at parity or below.

Note: 29% of all runs are near-degenerate at ε=3 (macro-F1 ≈ 0.080). Differences of 0.02-0.04 in that regime are likely noise.

## Plots
- `results/figures/b5_fedguard_vs_autogm.png`
- `results/figures/weights_fedguard_sign_flip.png`, `weights_autogm_sign_flip.png`
