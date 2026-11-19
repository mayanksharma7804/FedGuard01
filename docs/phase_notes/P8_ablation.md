# Phase 8 note - Ablation (E6), stress tests (E7) and the experiment campaign

**Dates:** 18-19 Nov 2026 | **Lead:** R5 (Experiments + writing) | **Objectives:** O4, O5

> Replaces the 31 Oct note (a single-seed, 20-round ablation, F46).

## What we did
- E6: one change at a time from the frozen FedGuard - R2 mode, R4, s floor, lambda_scale, R1 - 3 seeds, no attack
  and 30% sign flip (`configs/sweeps/e6_ablation_*.yaml`).
- E7: attacker share 10-40% (sign flip, eps 3) and epsilon 8 / 1, FedGuard vs AutoGM vs FedAvg, seed 42.
- The 144-run queue was finished on the team laptop plus six Kaggle accounts (T4 GPUs): explicit run lists per
  machine, zips imported with checks (`scripts/import_runs.py`); docs/FedGuard_Faster_Training_Guide.pdf.

## Numbers we got
| Result | Value | Finding |
|---|---|---|
| E6 | macro-F1 0.196-0.203 (none), 0.144-0.150 (sign flip) for every variant; no R4: HRR 0.033; no s floor: 0.013-0.032; no R1: equal weights | F57 |
| E7 attacker share | attackers' weight at 10/20/30/40%: AutoGM 0.115/0.232/0.325/0.406, FedGuard 0.095/0.174/0.289/0.364; all methods break at 40% | F58 |
| E7 epsilon | eps 8 + 30% sign flip: FedGuard 0.170 vs AutoGM 0.110 (FedAvg 0.173) | F58 |
| Cross-machine | same run on two T4s bit-identical (36 of 36 pairs: 20 + 16); T4 vs RTX 3050: 0.0034 macro-F1 | F53 |

## What went wrong / surprises
- A thermal-guard process died with a Claude session and stranded training for ~40 min (F51); the guard now lives
  inside the training process, and WMI-launched jobs opt out of Windows power throttling.
- One Kaggle share was run twice by mistake; the duplicate became a second determinism check (F53).

## Decision
Breaking points reported: 40% attackers, and epsilon <= 3 for sign flip. Plots: `e6_ablation.png`, `e7_stress.png`.
