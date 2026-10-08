# FedGuard - project memory for Claude Code

B.Tech CSE final-year project (AKGEC). Privacy-preserving + Byzantine-robust federated intrusion
detection. Paper: "FedGuard: A Scoping Review and Proposed Framework..." (Sept 2026).
Plan: FedGuard_Implementation_Plan.pdf (8 phases).

## Main idea
Clients add DP noise and declare (sigma, C, B, T, lr). The server predicts each client's noise
radius r_i = lr * sigma * C * sqrt(T * d) / B, removes it from the distance, and only distrusts the
displacement that noise + non-IID heterogeneity cannot explain (noise-aware AutoGM).

## Hard rules (do not break)
- Primary platform is native Windows 11 (no WSL, no Docker). Linux laptop is only for long sweeps.
- Python 3.11 in .venv (made by uv). Never use the Microsoft Store Python.
- Same CNN-LSTM (with opacus DPLSTM) in every baseline. No BatchNorm.
- Plain SGD (momentum 0, no weight decay) for every client, so the noise-radius formula holds.
- Seeds fixed (random, numpy, torch); partitions loaded from partitions/*.npz, never re-drawn.
- Leakage guards: drop IP/port/time columns, dedupe before split, scaler fit on train only.
- Report macro-F1, per-class recall, benign FPR, epsilon at delta, HRR, attacker weight share.
- src/fedguard/aggregators.py is pure NumPy (no Flower import). Flower code lives in flower_app/.
- DataLoader num_workers=0 on Windows. Guard scripts with if __name__ == "__main__":.
- Never commit data/raw, data/processed, results/runs, *.pt.
- Every finding (bug caught, data surprise, measurement, decision, negative result) goes into
  Findings.md as a numbered entry: what we did / what we found (numbers) / evidence / decision.

## Machine
Everything (coding, dataset, training, experiments) runs on ONE Windows laptop: RTX 3050 4 GB,
16 GB RAM, driver 566.07. Repo: https://github.com/mayanksharma7804/FedGuard01 (branch main).
Measured: non-DP FedAvg round (5 clients, E=5) ~26 s; DP round ~65 s with 2 jobs on the GPU -> ~45 min per
40-round DP run. DP on CPU is ~5x slower - always use the GPU.

## Pinned versions (requirements.txt, made with uv pip compile --universal)
Python 3.11.17, torch 2.14.1 (CUDA build from the cu126 index - there is no cu128 build), opacus 1.6.0,
flwr 1.39.0. Do not upgrade mid-project.

## Commands
- Windows setup:   powershell -ExecutionPolicy Bypass -File scripts\windows\setup_windows.ps1
- Flower check:    powershell -ExecutionPolicy Bypass -File scripts\windows\flower_check.ps1
- GPU/DP timing:   .venv\Scripts\python.exe scripts\check_gpu_dp.py
- Linux setup+run: bash scripts/linux/bootstrap_and_train.sh   (--smoke, --setup-only, --push-results)
- Tests:           python -m pytest -q
- One experiment:  python scripts/run_experiments.py --config configs/<name>.yaml
- Sweep:           python scripts/run_experiments.py --sweep configs/sweeps/main.yaml --resume
- Plots:           python scripts/collect_results.py && python scripts/make_plots.py

## Flower 1.39 facts (see docs/flower_windows.md)
Control API is HTTP on 127.0.0.1:8000 (not 9093); SuperLink/SuperNode need .venv\Scripts on PATH
(they spawn flower-superexec); custom strategy = subclass serverapp.strategy.FedAvg and override
aggregate_train(server_round, replies).

## Current phase
Update this line at the start of every phase: Phase 5 - DP-SGD clients, E0, B2 IN PROGRESS. Phase 4 DONE (docs/phase_notes/P4_federated.md). Code for Phase 6/7 (attacks.py, autogm, fedguard) already written + unit-tested; their experiments come in Phases 6/7.

## Training facts (Phase 3)
- Fixed for ALL baselines and clients: plain SGD lr 0.2, batch 256, no class weights (chosen on validation, F19).
- B0 = ceiling: test macro-F1 0.597 +/- 0.004 (seeds 42/43/44), binary F1 0.912, benign FPR 0.068.
- `python scripts/run_centralised.py` (skips finished runs; live log in results/runs/<run>/train.log).
- Decide designs on the VALIDATION split only: `python scripts/compare_runs.py --glob "<pattern>"`.
- Run at most 2 trainings in parallel (laptop heats up, F21).

## Data facts (Phase 2)
- `python scripts/prepare_data.py` -> data/processed/nf_unsw.npz (+ _manifest.json); `python scripts/make_partitions.py` -> partitions/nf_unsw_alpha{a}_k{K}_seed{s}.npz.
- Load with `fedguard.data.load_processed("nf_unsw")` and `fedguard.partition.load_partition(path, fingerprint=data.fingerprint)`.
- 104,022 unique rows, 38 features, 10 classes (Benign = 0). Train 45,225 (cap 20k/class), val 8,322, test 20,805.
- Dropped identifiers: IPs, ports, DNS_QUERY_ID. log1p as float32 BEFORE de-duplication (prevents float collisions across splits).

## Federated facts (Phase 4)
- Engine: `src/fedguard/fl_sim.py`; runner: `scripts/run_experiments.py --config|--sweep [--shard i/n] [--set k=v] [--dry-run]`.
  Run dir name = name_a{alpha}_k{K}_e{E}[_eps{e}_c{C}][_{attack}{frac}]_s{seed}_{hash6}; finished = metrics_test.json.
- `set_flat` must COPY into parameters (never torch's vector_to_parameters: F24). Parameter vector always from
  model.parameters() (DPLSTM state_dict lists LSTM weights twice under aliases, F26).
- All federated baselines: E=5, patience 15, min 10 rounds, max 120 (B1) / 40 (DP runs: max_rounds = privacy budget).
- B1 (test macro-F1): alpha 100 0.603, alpha 0.5 0.537 +/- 0.076, alpha 0.1 0.435 (F29). Seed spread at alpha 0.5
  is large -> always compare methods on the same partitions (paired).
- Never run Flower SuperNodes next to training jobs (RAM, F25). Flower parity = 6.9e-17 (F26).
- Smoke check after big changes: `python scripts/run_experiments.py --sweep configs/sweeps/smoke.yaml --force`.

## DP facts (Phase 5)
- `src/fedguard/dp.py`: sigma per client calibrated ONCE for T_round * max_rounds steps (RDP, delta = 1/N);
  epsilon_spent for rounds actually run; noise_radius = lr*sigma*C*sqrt(T*d)/B; DP config needs dp.clip.
- make_private uses its own noise generator, so sigma and sigma=0 runs share batches and dropout (E0, determinism).
- Model must be in train mode before make_private (evaluate() leaves it in eval mode).
- Per-example grad norms with GradSampleModule need a MEAN loss (sum gives batch-size-times norms).

