# Phase 6 note - Attacks, AutoGM (B3, B4) and the O2 conflict study

**Dates:** 8 Oct - 19 Nov 2026 (re-run 17-19 Nov, F46) | **Lead:** R4 (Robustness) | **Objectives:** O2, O4

> Replaces the 31 Oct note, which was written from 20-round runs stopped during the DP warm-up (F46).

## What we did
- Three attacks (`src/fedguard/attacks.py`): label flip (data poisoning, attack rows relabelled Benign), sign flip
  (minus the honest update), gaussian (random noise, 10x the honest length). 30% of the K = 10 clients, chosen once
  per run from the seed; attackers also run DP-SGD and declare an honest-looking sigma.
- AutoGM tuned on validation (lambda = lambda_scale x median distance), cross-checked against the Blades reference
  implementation to 1e-11 (F37). B3 = AutoGM without DP (seed 42), B4 = AutoGM + DP (3 seeds).
- E4 / O2 conflict: honest clients only, epsilon {inf, 3, 1, mixed 1-3-8} x alpha {0.1, 0.5, 100} x 2 seeds.
- Every K = 10 run: E = 5, 40-round privacy budget, patience 15, at least 10 rounds.

## Numbers we got
| Result | Value | Finding |
|---|---|---|
| AutoGM lambda on validation | small lambda collapses onto one client (HRR 0.90); large lambda lets sign-flip attackers keep 31-37%; **lambda_scale 4** chosen | F39 |
| Attacks vs FedAvg (no DP, test macro-F1) | none 0.370, label flip 0.303, sign flip 0.264, gaussian 0.338 | F54 |
| Attacks vs FedAvg / AutoGM (eps 3) | gaussian: FedAvg 0.145, AutoGM 0.206 (attackers' weight 0); sign flip: 0.163 / 0.152 (AutoGM gives attackers 0.32) | F54 |
| AutoGM honest rejection (E4) | 0.27-0.36 without DP, ~0 at one shared epsilon, **0.40 under mixed privacy** (every eps-1 client weight 0) | F47, F56 |

## What went wrong / surprises
- The O2 conflict appears without DP as well: under non-IID data AutoGM sidelines unusual honest clients (alpha 0.1:
  its model collapses, macro-F1 0.099).
- At one shared epsilon every client declares the same noise radius (F36), so AutoGM rejects almost nobody - the
  conflict is driven by unequal privacy levels, which is why the mixed-privacy scenario was added.
- 129 runs made with a 20-round / patience-5 setting were invalid and re-run (F46).

## Decision
lambda_scale 4 frozen for every AutoGM run. Plots: `e5_attacks.png`, `e4_conflict.png`.
