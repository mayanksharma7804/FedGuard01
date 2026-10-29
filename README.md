# FedGuard

Privacy-preserving and Byzantine-robust federated intrusion detection.
B.Tech CSE final-year project, AKGEC Ghaziabad.

**Team:** Mayank Sharma, Harshit Singhal, Harsh Tripathi, Vaishali Deshwal, Anuj Kumar.

---

## What is FedGuard?

FedGuard is a federated learning aggregation algorithm designed for network intrusion detection systems (IDS). It combines:

- **Federated Learning** (FedAvg, K=10 clients, Dirichlet non-IID partitioning α ∈ {0.1, 0.5, 100})
- **Differential Privacy** (DP-SGD with Opacus, per-client RDP accounting, ε ∈ {1, 3, 8})
- **Byzantine Robustness** (FedGuard aggregator with 4 components: effective-rank filter R1, quadrature trust weighting R2, MAD outlier allowance R3, DP noise tolerance R4)

**Dataset:** NF-UNSW-NB15-v2 — 104,022 network flow records, 38 features, 10 traffic classes.  
**Model:** CNN-LSTM with 40,266 parameters (no BatchNorm, compatible with DP-SGD).

---

## Key Results

| Method | Attack | α=0.1 | α=0.5 | α=100 |
|---|---|---|---|---|
| FedAvg (no DP) | none | 0.435 | 0.537 | 0.603 |
| FedAvg + DP ε=3 | none | 0.108 | 0.171 | 0.200 |
| AutoGM + DP ε=3 | sign_flip 30% | 0.012 | 0.035 | 0.168 |
| **FedGuard + DP ε=3** | **sign_flip 30%** | **0.012** | **0.132** | **0.168** |
| **FedGuard + DP ε=3** | **gaussian 30%** | **0.176** | — | — |

FedGuard outperforms AutoGM most on Gaussian attacks (+94 macro-F1 pts at α=0.1), because the R4 noise-tolerance zone correctly distinguishes DP noise from attack signal.

All results in `results/master.csv` (181 federated runs). Figures in `results/figures/`.

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
| `Findings.md` | all findings F1–F44 (bugs, decisions, insights) |

---

## Running experiments

```powershell
# Run all Phase 7+8 experiments sequentially (avoids GPU deadlock on 4GB VRAM)
powershell -ExecutionPolicy Bypass -File scripts\run_phases_7_8.ps1

# Collect results into master.csv
python scripts\collect_results.py

# Generate all figures
python scripts\make_plots.py --b1 --b2 --attacks --e4 --fedguard --ablation
```

---

## Where the work runs

Everything runs on one Windows laptop with an NVIDIA RTX 3050 (4 GB). Training is sequential
(one shard at a time) to avoid GPU memory deadlock — see `Findings.md` F41 for details.
