# Phase 8 — Ablation Study (E5)

## What we did
30-run ablation over α ∈ {0.1, 0.5, 100} × 5 aggregators × 2 attacks (none, sign_flip), seed=42, DP ε=3, K=10.

## Aggregators compared
1. FedGuard-quadrature+R4 (full system: quadrature residual + noise_tol_z=3.0)
2. FedGuard-linear+R4 (linear residual + noise_tol_z=3.0)
3. FedGuard-quadrature (no R4: quadrature residual only, no noise_tol_z)
4. AutoGM (lam_scale=4.0)
5. FedAvg (baseline)

## Key findings

**E5 — Mode ablation (quadrature vs linear)**:
- At α=0.5 and α=100, both modes perform similarly on clean data and sign_flip.
- At α=0.1 (extreme non-IID), all methods degrade severely; differences are within noise.

**R4 ablation (noise_tol_z effect)**:
- Removing R4 (FedGuard-quadrature vs FedGuard-quadrature+R4): on clean data nearly identical; on sign_flip at α=0.5, R4 provides modest protection. At α=0.1, R4 at z=3.0 appears slightly over-aggressive under extreme non-IID + DP noise.

**Alpha sensitivity**:
- α=0.1: all methods near-collapse at ε=3. Differences not meaningful.
- α=0.5: moderate heterogeneity; FedGuard provides clearest gains on gaussian attack (Phase 7).
- α=100: near-IID; FedAvg competitive with any robust aggregator.

## Limitation
Single seed (42) and one epsilon value (ε=3) limits statistical conclusions. The mixed-ε sweep (Phase 9) directly tests the core FedGuard hypothesis of heterogeneous privacy handling.

## Plots
- `results/figures/e5_ablation_none.png`
- `results/figures/e5_ablation_sign_flip.png`
