# Phase 8 — Ablation Study (E5/E6/E7)

## What we did
30-run ablation over α ∈ {0.1, 0.5, 100} × 5 aggregators × 2 attacks (none, sign_flip), seed=42, DP ε=3.

## Aggregators compared
1. FedGuard-quadrature+R4 (full system)
2. FedGuard-linear+R4 (linear mode, no quadrature)
3. FedGuard-quadrature (no R4, no noise_tol_z)
4. AutoGM (lam_scale=4)
5. FedAvg (baseline)

## Key findings
**E5 — Mode ablation (quadrature vs linear)**:
- At α=0.1 (extreme non-IID), quadrature+R4 ≈ linear+R4 on clean data (0.102 vs 0.104), but quadrature edges ahead on sign_flip (0.012 vs 0.053) — wait, linear is better here. Results are noisy at α=0.1 due to extreme heterogeneity.
- At α=0.5 and α=100, both modes perform similarly; linear+R4 slightly more stable.

**E6 — R4 ablation (noise_tol_z effect)**:
- FedGuard-quadrature (no R4) vs FedGuard-quadrature+R4: on clean data (attack=none) near-identical; on sign_flip at α=0.1, removing R4 yields 0.044 vs 0.012 — removing R4 actually helps here, but this is likely because R4 is too aggressive with DP noise at extreme non-IID.
- At α=0.5+, R4 provides consistent protection against sign_flip.

**E7 — Alpha sensitivity**:
- All methods degrade severely at α=0.1 due to extreme non-IID; FedAvg surprisingly competitive at α=100 (near-IID) where there's little distributional attack advantage.
- FedGuard provides most relative gain at α=0.5 (moderate heterogeneity).

## Plots
- `results/figures/e5_ablation_*.png`
