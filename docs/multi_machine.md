# Training on several machines (teammate laptops, Kaggle)

Full step-by-step guide: **[FedGuard_Faster_Training_Guide.pdf](FedGuard_Faster_Training_Guide.pdf)**.

Summary - every machine runs ONE job on its own share of the SAME queue:

| Who | Command |
|---|---|
| main laptop | `scripts\windows\queue_start.ps1 -Machine 0 -Machines N` |
| teammate laptop k | `scripts\windows\queue_start.ps1 -Machine k -Machines N` (same git commit, same `data\processed\nf_unsw.npz`) |
| Kaggle notebook | `scripts/kaggle/fedguard_kaggle.ipynb` -> `bash scripts/kaggle/kaggle_run.sh k N` (or `k1,k2` for two T4 GPUs) |
| laptop without NVIDIA GPU | `queue_start.ps1 -Only nodp -Threads 4` (runs without DP: nearly as fast on a CPU, F06); the GPU laptop then uses `-Only dp` |
| Kaggle + CPU laptop | Kaggle notebook `ONLY = "dp"` (open it in a browser on any laptop); all GPU machines split only the DP runs |
| hand back results | `python scripts/export_runs.py` -> zip in `results/exports/` |
| main laptop imports | `python scripts/import_runs.py <zip>` (checks name/config, data fingerprint; never overwrites) |
| share what is finished | `python scripts/export_runs.py --publish-done`, commit + push `results/done_runs.txt` |
| final sweep | main laptop: `queue_start.ps1` (single machine) runs whatever is still missing |

All machines must print the same **queue id** (first line of the training log).
