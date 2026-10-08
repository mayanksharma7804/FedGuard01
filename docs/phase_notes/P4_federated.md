# Phase 4 note - Federated engine and FedAvg (B1)

**Dates:** 7-8 Oct 2026  |  **Lead:** R3 (FL systems)  |  **Objective:** O1 (federated baseline)

## What we did
- `src/fedguard/fl_sim.py`: our own deterministic FL engine (one Windows process, no Ray). Each round every
  client starts from the global model, trains E local epochs (plain SGD), sends delta = local - global and
  declared metadata (n, lr, sigma, C, B, T); the server aggregates the deltas and evaluates on validation.
  Per-client/per-round seed `crc32("{seed}-{round}-{client}")`; best round restored at the end; test once.
- `src/fedguard/aggregators.py`: pure-NumPy aggregator interface `aggregate(X, n, r, params) -> (z, w, info)`
  with `fedavg` (AutoGM / FedGuard plug in without engine changes).
- `scripts/run_experiments.py`: `--config` / `--sweep` / `--set k=v` / `--shard i/n` / `--dry-run` /
  `--resume`. Finished runs (`metrics_test.json` exists) are skipped, so a stopped sweep just restarts.
  Each run writes `config.yaml`, `rounds.csv`, `weights_log.csv`, `metrics_val.json`, `metrics_test.json`,
  `meta.json` (git commit, host, time), and a live `train.log`.
- `scripts/collect_results.py` -> `results/master.csv`; `scripts/make_plots.py --b1`.
- Flower: `src/fedguard/flower_strategy.py` (`AggregatorStrategy` = Flower FedAvg + our aggregator) and
  `flower_app/` (ServerApp/ClientApp), run in deployment mode on Windows (`scripts/windows/flower_parity.ps1`).
- Tests: `tests/test_engine.py` (FedAvg weighted mean; 1 client = local training; clients never modify the
  global vector; round 1 = weighted mean of independent clients; same seed => identical run; sweep names).
- Smoke sweep `configs/sweeps/smoke.yaml` (2 rounds x 2 configs, about 1 minute).

## Results
**Flower parity (F26):** our engine and Flower agree to **6.9e-17** in validation macro-F1 over 5 rounds -
after the parity check had exposed a critical engine bug (F24).

**Local epochs (F27, validation, equal compute):** E = 5 best and smoothest (val macro-F1 0.548), ~45 rounds.

**B1 FedAvg, 5 clients, E = 5 (test, 3 seeds; F29)** - `results/b1_fedavg/summary_by_alpha.csv`
| alpha | macro-F1 | gap to B0 (0.597) | accuracy | binary F1 | benign FPR | best round |
|---|---|---|---|---|---|---|
| 100 (IID) | **0.603 +/- 0.005** | +0.006 | 0.896 | 0.911 | 0.074 | 100 |
| 0.5 | **0.537 +/- 0.076** | -0.060 | 0.888 | 0.906 | 0.057 | 53 |
| 0.1 | **0.435 +/- 0.028** | -0.163 | 0.873 | 0.894 | 0.067 | 67 |

Per-class recall (alpha 100 -> 0.1): Worms 0.30 -> 0.03, Backdoor 0.36 -> 0.13, DoS 0.36 -> 0.20,
Generic 0.61 -> 0.29; Benign and Exploits stay ~0.9. Figures: `results/figures/b1_fedavg_by_alpha.png`,
`b1_fedavg_rounds.png`, `b1_fedavg_per_class_recall.png`.

Comparison with ref [1] (91-93%): our accuracy 0.87-0.90 / binary F1 0.89-0.91 on de-duplicated data
without identifiers (F07, F13) - not directly comparable; macro-F1 is our main metric.

## What went wrong / surprises
- **Engine bug (F24):** `vector_to_parameters` made the model's parameters views of the global vector, so
  FedAvg silently became sequential training. Caught only by the Flower parity check; fixed with a copying
  `set_flat`, regression tests added, affected runs quarantined.
- 5 local Flower SuperNodes + 2 trainings ran the laptop out of RAM (F25): never run Flower next to training.
- Patience 8 stopped non-IID runs inside temporary dips (F28): re-ran B1 with patience 15 / max 120.
- alpha 0.5 has a large seed spread (+/- 0.08) because each seed has its own partition (F29).

## Decisions taken
- All federated baselines: plain SGD lr 0.2, batch 256, **E = 5**, patience 15, min 10 rounds.
- Main experiments at alpha 0.5; every method runs on the same 3 partitions; report paired comparisons.
- Experiments run in our engine; Flower is used for the parity proof and the live demo.

## Checkpoint
- [x] B1 macro-F1 vs round graph; alpha table with gap to B0; comparison with ref [1].
- [x] `--resume`: finished runs are skipped (B1 sweep was stopped and restarted; `--dry-run` lists done/todo).
- [x] Flower parity matches; `tests/test_engine.py` (same seed => identical metrics) passes.

## Next steps (Phase 5)
DP-SGD clients with Opacus (sigma fixed once per client for the round budget), epsilon accounting,
E0 noise-radius calibration, B2 DP-FedAvg for epsilon {inf, 8, 3, 1}.
