# FedGuard

Privacy-preserving and Byzantine-robust federated intrusion detection.
B.Tech CSE final-year project, AKGEC Ghaziabad.

Team: Mayank Sharma, Harshit Singhal, Harsh Tripathi, Vaishali Deshwal, Anuj Kumar.

The full plan is in `FedGuard_Implementation_Plan.pdf` (8 phases). Project rules for Claude Code
are in `CLAUDE.md`.

## Setup (Windows, no Linux needed)

Keep the repo in a path **without spaces** (e.g. `D:\Final_Project`).

```powershell
powershell -ExecutionPolicy Bypass -File scripts\windows\setup_windows.ps1
.\.venv\Scripts\Activate.ps1
python scripts\verify_setup.py
```

The setup script installs uv, creates `.venv` with Python 3.11, installs the exact versions in
`requirements.txt`, and installs the CUDA build of PyTorch when an NVIDIA GPU is present.

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
| `docs/` | phase notes and how-to notes |

## Where the work runs

- **Laptop 1** (development): code, tests, small runs on synthetic data.
- **Laptop 2** (RTX 4050): real dataset and long training runs. See `docs/laptop2_steps.md`.
