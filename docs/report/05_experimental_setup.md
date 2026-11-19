# Chapter 5 - Experimental setup

> DRAFT (19 Nov 2026) - all experiments completed (144 / 144 queue runs).

## 5.1 Hardware and software

| Item | Value |
|---|---|
| Machine | one Windows 11 laptop: Intel Core i5-12450H (8 cores), 16 GB RAM, NVIDIA RTX 3050 Laptop GPU 4 GB |
| Software | Python 3.11.17, PyTorch 2.14.1 (CUDA 12.6 build), Opacus 1.6.0, Flower 1.39.0, NumPy 2.4, scikit-learn 1.9 - exact versions in `requirements.txt` (cross-platform lock file) |
| Platform | native Windows (no WSL, no Docker): our own deterministic engine for experiments, Flower deployment mode (no Ray) for the parity check and the demo |
| Run time | K = 10, DP, 40 rounds: ~30 min per run on the laptop (37 s per round plus thermal pauses, F49) or ~25 min on a Kaggle T4; the campaign used the team laptop plus six Kaggle accounts (F53) |

Thermal limits shaped the schedule: two parallel jobs heated the CPU to 95 C, while one job with turbo boost off
gives the same throughput; a thermal guard pauses training (never changing results) whenever the CPU or GPU
reaches 78 C (F49). An NVIDIA driver failure caused by a loose charger once hung two jobs for ~4 hours (F41);
since then every round is checkpointed and a stall detector watches the logs.

## 5.2 Fixed settings (all experiments)

| Setting | Value | Source |
|---|---|---|
| Data | NF-UNSW-NB15-v2, de-duplicated, identifiers dropped; train 45,225 (cap 20k/class), val 8,322, test 20,805 | F07-F12 |
| Model | CNN-LSTM with DPLSTM, 40,266 parameters | F06 |
| Client optimiser | plain SGD, lr 0.2, batch 256, no class weights | F19 |
| Local epochs per round | E = 5 | F27 |
| Early stopping | best validation round; patience 15 rounds, at least 10 rounds | F28, F46 |
| Rounds | B1: up to 120; every K = 10 run and every DP run: up to 40 (= the privacy budget) | F29, F46 |
| DP | clip C = 2, delta = 1/N per client, RDP accounting, sigma calibrated for 40 rounds | F33-F35 |
| Attackers | 30% of the clients (3 of 10), DP-SGD + honest-looking declared sigma | F40 |
| AutoGM | lambda_scale = 4 (chosen on validation) | F39 |
| FedGuard | chosen on validation, then frozen | F48, F50 |
| Seeds | 42, 43, 44 (partition, initial model and DP noise change together) | |

## 5.3 Experiment matrix

| ID | Experiment | Grid | Runs | Status |
|---|---|---|---|---|
| E0 | noise-radius calibration | 5 clients x sigma {0.5, 1, 2} x 5 seeds x 2 start models | 150 pairs | done (F35) |
| E1 | B0 centralised ceiling | 3 seeds | 3 | done (F22) |
| E2 | B1 FedAvg, K = 5 | alpha {0.1, 0.5, 100} x 3 seeds | 9 | done (F29) |
| E3 | B2 DP-FedAvg, K = 5, alpha 0.5 | epsilon {inf, 8, 3, 1} x 3 seeds | 12 | done (F38) |
| - | AutoGM lambda (validation) | lambda_scale {0.5, 1, 2, 4, 8} x {none, sign flip} | 10 | done (F39) |
| - | FedGuard settings (validation) | 8 variants x {none, sign flip} | 16 | done (F50) |
| E4 | O2 conflict (honest clients only) | {AutoGM, FedGuard} x epsilon {inf, 3, 1, mixed} x alpha {0.1, 0.5, 100} x 2 seeds | 48 | done (F56) |
| E5 | main attack table, K = 10, alpha 0.5, epsilon 3 | {FedAvg no DP, FedAvg, AutoGM, FedGuard} x {none, label flip, sign flip, gaussian} x 3 seeds (+ AutoGM no DP, seed 42 = B3) | 52 | done (F54) |
| E5m | mixed privacy (epsilon 1/3/8 across clients) | {FedAvg, AutoGM, FedGuard} x {none, sign flip} x 3 seeds | 18 | done (F55) |
| E6 | FedGuard ablation (one change at a time) | mode, R4, s floor, lambda_scale; 3 seeds at the selected setting's neighbours | ~24 | done, 30 runs (F57) |
| E7 | stress | attacker share {10, 20, 40%} (sign flip) and epsilon {8, 1}, FedGuard vs AutoGM | ~16 | done (F58) |
| E8 | second dataset (CICIoT2023) | - | - | dropped (time), future work |

## 5.4 Deviations from the implementation plan (to be explained in the report)

1. **Experiments in our own engine, Flower for the parity check and the demo** (plan D1); engine parity 6.9e-17 (F26).
2. **No Docker**: the demo is a multi-laptop Flower deployment with a one-laptop fallback (plan D2, `docs/demo.md`).
3. **K = 10 clients** for every robustness experiment (plan D3), so 30% attackers is three clients.
4. **Refinements R1-R4 adopted and an s floor added** (plan D4, F48), each tested as an ablation.
5. **Reduced grids** (F40): one attacker share (30%) in the main table, the attacker-share curve with seed 42 only;
   epsilon 8 dropped from E4 because B2 showed it behaves like epsilon 3 (F38); E4 with 2 seeds; E8 dropped.
6. **A mixed-privacy scenario added** (F36): with one shared epsilon every client declares nearly the same noise,
   which does not test FedGuard's per-client idea.
7. **Runs made between 21 and 31 October with a 20-round budget and patience 5 are not used** (F46): they stopped
   DP runs during the warm-up; all affected experiments were re-run with the documented 40-round budget.
8. **Results are not directly comparable with papers that keep duplicates** (F13): 95.5% of the dataset is
   duplicate rows; our numbers come from a de-duplicated, leakage-checked protocol and are lower.
