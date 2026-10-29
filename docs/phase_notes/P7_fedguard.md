# Phase 7 — FedGuard Full Attack Grid (B5)

## What we did
Ran FedGuard (quadrature mode, k_mad=2.0, lam_scale=1.0, noise_tol_z=3.0, outer=20) against all 4 attack types (none / label_flip / sign_flip / gaussian at fraction=0.3) across seeds 42-44, with DP ε=3.

## FedGuard components
- **R1 (r_eff)**: Effective rank filter — clients with low-rank updates penalised
- **R2 (quadrature)**: Gaussian quadrature integration over trust weight distribution
- **R3 (MAD allowance)**: MAD-based outlier threshold, k_mad=2.0
- **R4 (noise_tol_z)**: DP noise tolerance zone, z=3.0 — allows for expected DP noise before flagging

## Key findings
- FedGuard (quadrature+R4, DP ε=3): macro-F1 on sign_flip = 0.110 vs AutoGM 0.082, vs FedAvg 0.092 — moderate improvement.
- Trust weights (attacker_share_mean) drop from 0.30 to ~0.06 on sign_flip — FedGuard correctly down-weights attackers.
- gaussian attack: FedGuard 0.176 vs AutoGM 0.082 — biggest gain, because R4 correctly distinguishes DP noise from Gaussian attack noise.
- label_flip: all methods struggle (~0.080-0.106), as expected — label corruption is gradient-direction-preserving.
- HRR (attacker high-weight rate) driven to 0.00 on sign_flip/gaussian by FedGuard.

## Plots
- `results/figures/b5_fedguard_vs_autogm.png`
- `results/figures/weights_*.png` — trust weight distributions per attack
