# Phase 7 note - FedGuard aggregator (B5)

**Dates:** 7 Oct (code) - 19 Nov 2026 (results) | **Lead:** R4 | **Objective:** O3

> Replaces the 31 Oct note (20-round runs, settings not chosen on validation - F46).

## What we did
- `aggregators.fedguard`: AutoGM whose trust is computed on the displacement that declared noise and normal non-IID
  spread cannot explain. Refinements: R1 centre-noise-corrected radius, R2 quadrature residual, R3 allowance
  median + 2 MAD of residuals, R4 noise-length tolerance, and an s floor so the allowance cannot collapse to 0 (F48).
- Unit tests (plan 12.4) in both modes, R4 off / on, s floor; one documented expected failure (F31).
- Settings chosen on the **validation** split (8 variants x {none, 30% sign flip}, seed 42, eps 3), rule fixed in
  advance (F50), then frozen in `configs/fedguard_frozen.yaml`.
- B5 / E5 main attack table and the mixed-privacy comparison, 3 seeds; Flower strategy + live demo.

## Numbers we got
| Result | Value | Finding |
|---|---|---|
| Chosen settings | quadrature + R4 (z 3) + s floor, lambda_scale 4: val 0.1780 vs runner-up 0.1779; honest rejection 0.005 vs 0.200 | F50 |
| E5 (shared eps 3) | tied with AutoGM on every attack (paired -0.018..+0.011); gaussian stopped, sign / label flip not | F54 |
| **Mixed privacy** | **AutoGM HRR 0.400 (eps-1 clients weight 0) vs FedGuard 0.004 (every honest client 0.10) at equal macro-F1 (0.219 / 0.214)** | F55 |
| Mixed privacy + 30% sign flip | FedGuard 0.118, AutoGM 0.113, FedAvg 0.153 (seed 44 collapse for both robust aggregators) | F55 |
| Live demo (5 clients, 10 rounds) | eps-1 hospital rejected in 10/10 rounds by AutoGM, 5/10 by FedGuard; attacker 0 for both | F52 |

## What went wrong / surprises
- The allowance s collapsed to 0 under DP whenever most residuals were hidden by noise; the s floor fixed it (F48).
- At one shared epsilon FedGuard cannot do better than AutoGM - every client has the same radius and the sign-flip
  displacement is buried in the noise (F39).

## Decision
Frozen settings used for every test-set experiment and the demo. Plots: `e5_mixed_privacy.png`,
`weights_mixed_privacy_{autogm,fedguard}.png`, `e5_attacks.png`.
