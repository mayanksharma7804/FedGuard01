# FedGuard

Privacy-preserving and Byzantine-robust federated intrusion detection.
B.Tech CSE final-year project, AKGEC Ghaziabad.

**Team:** Mayank Sharma, Harshit Singhal, Harsh Tripathi, Vaishali Deshwal, Anuj Kumar.

---

## What is FedGuard?

FedGuard is a federated learning aggregation algorithm designed for network intrusion detection systems (IDS). It combines:

- **Federated Learning** (FedAvg, K = 5 or 10 clients, Dirichlet non-IID partitioning α ∈ {0.1, 0.5, 100})
- **Differential Privacy** (DP-SGD with Opacus, per-client RDP accounting, ε ∈ {1, 3, 8})
- **Byzantine robustness that understands DP noise** (FedGuard = noise-aware AutoGM): each client declares (sigma, C, B, T, lr); the server predicts its noise radius r = lr·sigma·C·sqrt(T·d)/B and distrusts only the displacement that noise and normal non-IID spread cannot explain. Refinements: R1 centre-noise-corrected radius r_eff, R2 quadrature residual sqrt(dist² - r_eff²), R3 allowance s = median + k·MAD of the residuals, R4 noise-length tolerance, plus an optional floor on s (F48)

**Dataset:** NF-UNSW-NB15-v2 — 104,022 network flow records, 38 features, 10 traffic classes.  
**Model:** CNN-LSTM with 40,266 parameters (no BatchNorm, compatible with DP-SGD).

---

## Key Results (valid runs only - see Findings F46)

All numbers are **test** macro-F1 (mean ± std over seeds 42/43/44), best round chosen on validation.
DP runs: delta = 1/N per client, clipping C = 2, 40-round privacy budget. Every number traces to a row of
`results/master.csv`.

**Ceiling and cost of federation / privacy (K = 5 clients)**

| Setting | macro-F1 | Finding |
|---|---|---|
| B0 centralised | 0.597 ± 0.004 | F22 |
| B1 FedAvg, alpha = 100 / 0.5 / 0.1 | 0.603 / 0.537 / 0.435 | F29 |
| B2 DP-FedAvg alpha 0.5, eps = inf / 8 / 3 / 1 (40 rounds) | 0.513 / 0.253 / 0.219 / 0.183 | F38 |

**Attacks (K = 10, alpha 0.5, 30% attackers, 40 rounds)**

| Aggregator | Privacy | none | label flip | sign flip | gaussian |
|---|---|---|---|---|---|
| FedAvg | no DP | 0.370 ± 0.015 | 0.303 ± 0.019 | 0.264 ± 0.036 | 0.338 ± 0.019 |
| FedAvg | eps 3 | 0.218 ± 0.006 | 0.173 ± 0.036 | 0.163 ± 0.015 | 0.145 ± 0.012 |
| AutoGM (lam_scale 4, tuned, F39) | eps 3 | 0.210 ± 0.022 | *running* | *running* | *running* |
| FedGuard | eps 3 | *tuning on validation (F48)* | | | |

**Mixed privacy (eps 1 / 3 / 8 across clients) - the paper's core scenario:** AutoGM gives weight **0**
to every eps-1 (most private) client in every round (HRR 0.40); FedGuard keeps them (F47). The 3-seed,
40-round comparison is running.

Status (17 Nov 2026): the 20-round runs of 21-31 Oct were stopped during the DP warm-up and are
superseded (F46); the affected experiments are being re-run. FedGuard's settings are chosen on the
validation split before any FedGuard test result is looked at.

---

## Setup (Windows, no Linux needed)

Keep the repo in a path **without spaces** (e.g. `D:\Final_Project`).

```powershell
powershell -ExecutionPolicy Bypass -File scripts\windows\setup_windows.ps1
.\.venv\Scripts\Activate.ps1
python scripts\verify_setup.py
```

The setup script installs uv, creates `.venv` with Python 3.11, installs the exact versions in
`requirements.txt`, and installs the CUDA build of PyTorch when an NVIDIA GPU is present.

---

## Folder layout

| Folder | What is inside |
|---|---|
| `configs/` | one YAML per experiment, `sweeps/` lists of experiments |
| `data/raw`, `data/processed` | datasets (never committed to Git) |
| `partitions/` | saved non-IID client splits |
| `src/fedguard/` | the Python package (data, model, training, DP, attacks, aggregators, engine) |
| `scripts/` | setup, data preparation, experiment runner, plots |
| `tests/` | pytest unit tests |
| `results/` | `runs/` (not committed), `master.csv`, `figures/` |
| `docs/phase_notes/` | P1–P8 phase notes |
| `Findings.md` | all findings F01-F48 (bugs, decisions, measurements, results) |

---

## Running experiments

```powershell
# Start / resume the experiment queue (configs\sweeps\QUEUE.txt) - 2 jobs in the background
powershell -ExecutionPolicy Bypass -File scripts\windows\queue_start.ps1
# Progress, running jobs, GPU temperature
powershell -ExecutionPolicy Bypass -File scripts\windows\queue_status.ps1
# Pause safely (after the current round; add -AfterRun to finish the current run first)
powershell -ExecutionPolicy Bypass -File scripts\windows\queue_pause.ps1

# One experiment / one sweep
python scripts
un_experiments.py --config configs1_fedavg.yaml
python scripts
un_experiments.py --sweep configs\sweeps\p6_autogm.yaml --shard 0/2

# Collect results into master.csv and regenerate every figure
python scripts\collect_results.py
python scripts\make_plots.py
```

---

## Where the work runs

Everything runs on one Windows laptop with an NVIDIA RTX 3050 (4 GB). Two training jobs share the GPU
(more does not add throughput, F16/F40); every round is checkpointed so a stall (F41) or a pause loses nothing.
