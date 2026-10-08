# Phase 5 note - Differential privacy (B2) and noise-radius calibration (E0)

**Dates:** 8 Oct 2026  |  **Lead:** R2 (Model + DP)  |  **Objective:** O2 (foundation)

## What we did
- `src/fedguard/dp.py`:
  - `poisson_params` - sample rate, Poisson batches per epoch and expected batch size **exactly as Opacus'
    `make_private` sets them** (q = 1/ceil(n/B); F34);
  - `client_privacy_plan` - sigma chosen ONCE per client for T_round x max_rounds steps (RDP accountant,
    delta = 1/N), per-client epsilon supported (`dp.epsilon: [1, 3, 8]`, F36);
  - `epsilon_spent` - epsilon for the rounds actually run; `noise_radius` = lr*sigma*C*sqrt(T*d)/B;
  - `make_private` - Opacus wrapper with its own noise generator (sigma and sigma = 0 runs share batches and
    dropout - needed by E0 and for determinism).
- DP-SGD in the engine (`fl_sim.py`) and in the Flower client (`flower_strategy.client_round`); a test shows
  both produce the identical DP update. Run logs print sigma / delta / radius per client and epsilon spent.
- `scripts/measure_grad_norms.py` (per-example gradient norms -> C range), `scripts/e0_calibration.py` (E0),
  `make_plots.py --b2`, configs `b2_dpfedavg.yaml`, sweeps `p5_clip_pilot.yaml`, `b2_dpfedavg.yaml`, `smoke.yaml`.
- Tests: `tests/test_dp.py` (radius is 0 at sigma = 0, linear in sigma, ~sqrt(T); budget met; plan equals what
  Opacus really does; one-step and multi-step noise isolation; determinism; Flower = engine; per-client
  epsilon; radius ~independent of client size).

## Decisions (all on the validation split)
| Setting | Value | Why |
|---|---|---|
| Privacy unit | record-level DP per client, delta = 1/N | plan 10.1 |
| Budget | sigma for 40 rounds x E = 5 local epochs | plan compute budget; F27 |
| Clipping C | **2** (pilot {0.5, 1, 2, 4, 8} at eps 3: 0.21 / 0.22 / **0.26** / 0.23 / 0.21) | F30, F33 |
| Optimiser | plain SGD lr 0.2, expected batch 256, Poisson sampling | radius formula needs it |

## Results
**E0 - noise-radius calibration (F35)** - `results/e0_calibration/e0_table.csv`, `results/figures/e0_calibration.png`
| start | sigma | predicted r | measured | ratio mean +/- std | pairs |
|---|---|---|---|---|---|
| init | 0.5 / 1 / 2 | 2.00 / 3.99 / 7.99 | 2.01 / 4.02 / 8.02 | 1.004 +/- 0.005 | 75 |
| warm | 0.5 / 1 / 2 | 2.00 / 3.99 / 7.99 | 2.00 / 3.99 / 7.99 | 1.000 +/- 0.005 | 75 |

kappa = 1.005 (init) / 0.999 (warm) - **no correction needed**. cos(noise, clean update) ~ -0.005: noise is
orthogonal to the signal (supports quadrature, R2). Per-round SNR from a trained model: 0.55 at sigma 0.5,
0.14 at sigma 2.

**B2 - DP-FedAvg, alpha 0.5, 5 clients (F38)** - `results/b2_dpfedavg/summary_by_epsilon.csv`
| epsilon | macro-F1 | binary F1 | benign FPR |
|---|---|---|---|
| inf | **0.513 +/- 0.056** | 0.904 | 0.058 |
| 8 | **0.253 +/- 0.036** | 0.851 | 0.095 |
| 3 | **0.219 +/- 0.029** | 0.853 | 0.114 |
| 1 | **0.183 +/- 0.028** | 0.800 | 0.110 |

With DP, seven rare attack classes fall to ~0 recall; Benign / Exploits / (Fuzzers) remain.
Figures: `results/figures/b2_dpfedavg_by_epsilon.png`, `..._rounds.png`, `..._per_class_recall.png`.

## What went wrong / surprises
- **E0 caught a real bug (F34):** we declared q = B/n, floor(n/B) steps and B = 256, but Opacus samples with
  q = 1/ceil(n/B) and divides by int(n / batches). The radius was 5-20% too small (worse for small clients) and
  the accounting slightly optimistic (pilot spent 2.84-3.05 instead of 3). Fixed before B2; regression tests.
- My first per-example gradient-norm measurement was 512x too big (sum vs mean loss, F30).
- The privacy cost is much larger than hoped (F33, F38): with 2k-19k rows per client and a 40k-parameter model
  the update is >90% noise at every epsilon tested.
- At one shared epsilon all clients get almost the same radius regardless of size (F36) - FedGuard's
  per-client correction needs a mixed-privacy scenario to show its point.

## Checkpoint
- [x] B2 ran for all four epsilon values; epsilon-vs-macro-F1 graph done.
- [x] E0 table filled; kappa reported (1.00, no correction needed).
- [x] `tests/test_dp.py`: radius 0 at sigma = 0, linear in sigma and in sqrt(T) (+ more).

## Next steps (Phase 6)
Attacks (label flip, sign flip, Gaussian; smart attackers with DP), AutoGM with lambda tuned on validation
(already cross-checked with Blades, F37), B3/B4, the O2 conflict experiment E4 incl. a mixed-privacy scenario.
