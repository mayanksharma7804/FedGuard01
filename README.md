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

## Key Results (final, 19 Nov 2026)

Test macro-F1, mean +/- std over seeds 42/43/44 unless stated; best round chosen on validation. DP: delta = 1/N per
client, clip C = 2, 40-round privacy budget. FedGuard's settings were frozen on the validation split before any
FedGuard test result was seen (Findings F50). Every number traces to `results/master.csv` (207 runs); figures in
`results/figures/`, tables in `results/<experiment>/`.

**1. The privacy-robustness conflict, and FedGuard's answer - mixed privacy (epsilon 1 / 3 / 8 across 10 clients)**

| Aggregator | macro-F1 | Honest clients sidelined (HRR) | Mean weight of an honest client at eps 1 / 3 / 8 |
|---|---|---|---|
| FedAvg | 0.190 +/- 0.012 | 0.333 (sample sizes) | 0.087 / 0.125 / 0.116 |
| AutoGM (tuned) | 0.219 +/- 0.033 | **0.400** | **0.000** / 0.091 / 0.238 |
| **FedGuard** | 0.214 +/- 0.038 | **0.004** | **0.102 / 0.102 / 0.101** |

AutoGM silences every most-private client in every round; FedGuard gives every honest client its fair share
(1/K = 0.10) at the same accuracy (F55, `e5_mixed_privacy.png`). Across non-IID levels and privacy settings
AutoGM sidelines 27-40% of honest clients, FedGuard at most 10% (F56, `e4_conflict.png`).

**2. Attacks (K = 10, alpha 0.5, one shared epsilon of 3, 30% attackers)**

| Aggregator | none | label flip | sign flip | gaussian |
|---|---|---|---|---|
| FedAvg, no DP | 0.370 +/- 0.015 | 0.303 +/- 0.019 | 0.264 +/- 0.036 | 0.338 +/- 0.019 |
| FedAvg | 0.218 +/- 0.006 | 0.173 +/- 0.036 | 0.163 +/- 0.015 | 0.145 +/- 0.012 |
| AutoGM | 0.210 +/- 0.022 | 0.123 +/- 0.039 | 0.152 +/- 0.002 | 0.206 +/- 0.009 |
| FedGuard | 0.203 +/- 0.013 | 0.126 +/- 0.041 | 0.150 +/- 0.001 | 0.212 +/- 0.005 |

FedGuard and AutoGM are tied (paired per-seed differences -0.018 to +0.011). Both remove gaussian attackers
(weight 0); against sign flip and label flip neither keeps attackers below their share, because under DP noise
of length ~13 the attack's extra displacement (~1-1.5) is invisible to any distance-based rule (F39, F54).

**3. Where FedGuard does not win (reported, not hidden)**
- Mixed privacy **with** 30% sign-flip attackers: FedGuard 0.118, AutoGM 0.113, FedAvg **0.153** - both robust
  aggregators collapse in one of three seeds (F55).
- 40% attackers break every method (macro-F1 0.080, only Benign predicted); FedGuard helps against attackers mainly
  when the noise is small (epsilon 8: 0.170 vs AutoGM 0.110) (F58).

**4. Baselines (K = 5)**

| Setting | macro-F1 | Finding |
|---|---|---|
| Centralised (B0) | 0.597 +/- 0.004 | F22 |
| FedAvg alpha 100 / 0.5 / 0.1 (B1) | 0.603 / 0.537 / 0.435 | F29 |
| DP-FedAvg alpha 0.5, eps inf / 8 / 3 / 1 (B2) | 0.513 / 0.253 / 0.219 / 0.183 | F38 |
| Declared noise radius vs measured (E0) | measured / predicted = 1.000-1.005 | F35 |

Training ran on the team laptops and on Kaggle T4 GPUs (the `machine` column of `master.csv`); the same run on two
T4s is bit-identical, and on a T4 vs the laptop's RTX 3050 it differs by 0.0034 macro-F1 (F53).

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
| `Findings.md` | all findings F01-F58 (bugs, decisions, measurements, results) |

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
