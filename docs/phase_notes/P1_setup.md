# Phase 1 note - Setup and foundations

**Dates:** started 7 Oct 2026  |  **Lead:** R3 (FL systems)

## What we did (laptop 1)
- Created the repo `Project_Final` with the planned folder layout, `pyproject.toml` (package `fedguard`), README, CLAUDE.md, .gitignore, .gitattributes.
- Made one exact, cross-platform lock file: `uv pip compile requirements.in --universal --python-version 3.11 -o requirements.txt` (596 lines).
- Ran `scripts/windows/setup_windows.ps1`: uv + Python 3.11.17 in `.venv`, all packages, CUDA PyTorch.
- Proved Flower deployment mode works on native Windows (1 SuperLink + 2 SuperNodes, 3 FedAvg rounds): `scripts/windows/flower_check.ps1`.
- Timed the real CNN-LSTM with and without DP-SGD on CPU and GPU: `scripts/check_gpu_dp.py`.

## Numbers we got (laptop 1: RTX 3050 4 GB, driver 566.07)
| Item | Value |
|---|---|
| Python / torch / opacus / flwr | 3.11.17 / 2.14.1+cu126 / 1.6.0 / 1.39.0 |
| CNN-LSTM parameters | 40,266 (target 40-60k), Opacus validator OK |
| sec/step, GPU, no DP | 0.022 (about 7.5 min per 40-round run) |
| sec/step, GPU, DP-SGD | 0.045 (about 15 min per 40-round run) |
| sec/step, CPU, DP-SGD | 0.213 (about 71 min per run, so DP must run on the GPU) |
| Flower check | 3 rounds, train loss 0.171 -> 0.048 -> 0.025 |

## What went wrong / surprises
- `torch 2.14.1` has **no cu128 build**; the setup script now uses **cu126** (works with drivers >= 527). cu130/cu132 would need drivers >= 580.
- Flower 1.39 changed its CLI: `flwr new` pulls templates online, the Control API moved to **HTTP port 8000**, and SuperLink/SuperNode need `.venv\Scripts` on PATH. All fixed in `flower_check.ps1`; details in `docs/flower_windows.md`.
- The project path contains spaces (`D:\Hokage X Pirate king\...`). Everything worked anyway (no Ray in our stack). Anyone cloning the repo elsewhere should still use a path without spaces.

## Decisions taken
- Python environment tool: **uv** (same on Windows and Linux).
- Flower is used in deployment mode only (no Ray). Experiments will run in our own engine (Phase 4).
- **One laptop only** (this one, RTX 3050 4 GB) for coding, data and training. The second laptop is not used.
- GitHub repo: https://github.com/mayanksharma7804/FedGuard01 (branch `main`).

## Still to do for Phase 1 (team)
- [x] Create the GitHub repo and push.
- [ ] Download NF-UNSW-NB15-v2 into `data\raw\` and record its SHA-256 in `data/README.md` (start of Phase 2).
- [ ] Every member: learning tasks in plan section 6.1 (PyTorch basics, Flower, Opacus tutorials).
- [ ] Discuss decisions D1-D7 (plan section 3) with the project guide.
