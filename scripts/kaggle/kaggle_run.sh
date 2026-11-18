#!/usr/bin/env bash
# =============================================================================
# FedGuard on a Kaggle GPU notebook - one helper "machine" of the multi-machine queue (docs/multi_machine.md).
#
#   bash scripts/kaggle/kaggle_run.sh <machine indices> <number of machines> [hours] [data dir]
#   e.g. bash scripts/kaggle/kaggle_run.sh 2 3            # machine 2 of 3, stop starting runs after 10.5 h
#        bash scripts/kaggle/kaggle_run.sh 2,3 4          # GPU T4 x2: machines 2 and 3 of 4, one per GPU
#        bash scripts/kaggle/kaggle_run.sh smoke 1        # 2-round smoke test only (do this first)
#        ONLY=dp bash scripts/kaggle/kaggle_run.sh 1,2 3   # only runs WITH DP (when a CPU-only laptop takes the rest)
#
# What it does: Python 3.11.17 in an isolated .venv (uv), the project's pinned packages with the CUDA 12.6
# build of torch 2.14.1 (same build as the main laptop; the default Linux wheel needs a CUDA 13 driver),
# the tests, then the queue share(s), and finally a zip of all finished runs in /kaggle/working/ - download
# it from the notebook's Output tab and import it on the main laptop with scripts/import_runs.py.
# =============================================================================
set -euo pipefail
MACHINES_IDX="${1:?machine index (or comma list, or 'smoke')}"
N="${2:?number of machines}"
HOURS="${3:-10.5}"
ONLY="${ONLY:-all}"            # all | dp | nodp - must match how the other machines split the queue
REVERSE="${REVERSE:-0}"        # 1 = take runs from the END of the queue (main laptop goes from the front)
export FEDGUARD_MACHINE="${MACHINE_NAME:-kaggle}"   # label stored in every run's meta.json and the zip name
DATA="${4:-}"
cd "$(dirname "$0")/../.."
REPO="$(pwd)"
OUT=/kaggle/working
mkdir -p "$OUT" logs data/processed results/runs
say() { printf '\n==> %s\n' "$*"; }

say "GPU"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv || { echo "No GPU: in the notebook settings choose Accelerator = GPU T4 x2"; exit 1; }

say "Data"
if [[ -z "$DATA" ]]; then DATA="$(dirname "$(find /kaggle/input -name nf_unsw.npz -print -quit 2>/dev/null || true)")"; fi
if [[ ! -f "$DATA/nf_unsw.npz" ]]; then echo "nf_unsw.npz not found under /kaggle/input - add your private dataset to the notebook"; exit 1; fi
cp "$DATA/nf_unsw.npz" data/processed/
[[ -f "$DATA/nf_unsw_manifest.json" ]] && cp "$DATA/nf_unsw_manifest.json" data/processed/

say "Python 3.11.17 + pinned packages (about 5-10 minutes)"
pip install -q uv==0.12.23
uv python install 3.11.17
[[ -d .venv ]] || uv venv --python 3.11.17 .venv
PY="$REPO/.venv/bin/python"
# the lock file is complete (made with --universal): install it WITHOUT dependency resolution, minus torch and
# the CUDA 13 runtime wheels, then add torch 2.14.1 + its CUDA 12.6 runtime from PyTorch's own index
grep -viE '^(torch|triton|nvidia-|cuda-|pywin32|pywinpty|pypiwin32|wmi)' requirements.txt > /tmp/req_no_torch.txt
uv pip install --python "$PY" --no-deps -r /tmp/req_no_torch.txt
uv pip install --python "$PY" "torch==2.14.1" --index-url https://download.pytorch.org/whl/cu126
uv pip install --python "$PY" --no-deps -e .
"$PY" scripts/verify_setup.py

say "Tests (the run starts only if every test passes)"
"$PY" -m pytest -q -p no:warnings

export FEDGUARD_THREADS=4
if [[ "$MACHINES_IDX" == "smoke" ]]; then
  say "Smoke test (2 rounds, 2 configs)"
  "$PY" scripts/run_experiments.py --sweep configs/sweeps/smoke.yaml --force
  echo "SMOKE TEST PASSED - now run the real share with: bash scripts/kaggle/kaggle_run.sh <index> <machines>"
  exit 0
fi

if [[ -n "${SWEEPS_LIST:-}" ]]; then        # a chosen part of the queue, e.g. "configs/sweeps/a.yaml,configs/sweeps/b.yaml"
  IFS=',' read -ra SWEEPS <<< "$SWEEPS_LIST"
else
  mapfile -t SWEEPS < <(grep -vE '^\s*(#|$)' configs/sweeps/QUEUE.txt)
fi
SKIP=(); [[ -f results/done_runs.txt ]] && SKIP=(--skip-file results/done_runs.txt)
# runs reserved for OTHER machines (results/reserved/reserved_<labels>.txt): skip every list whose file name does not
# contain this machine's label (e.g. kaggle-2 keeps reserved_kaggle-2_kaggle-3_kaggle-4.txt, skips reserved_laptop.txt)
for f in results/reserved/reserved_*.txt; do
  [[ -f "$f" ]] || continue
  if [[ "$(basename "$f")" != *"$FEDGUARD_MACHINE"* ]]; then
    [[ ${#SKIP[@]} -eq 0 ]] && SKIP=(--skip-file)
    SKIP+=("$f"); echo "  skipping runs reserved for others: $f"
  fi
done
IFS=',' read -ra IDX <<< "$MACHINES_IDX"
NGPU=$(nvidia-smi -L | wc -l)
say "Queue share(s) ${IDX[*]} of $N (runs: $ONLY) on $NGPU GPU(s); no new run starts after $HOURS h"
pids=()
for k in "${!IDX[@]}"; do
  i="${IDX[$k]}"; gpu=$(( k % NGPU ))
  CUDA_VISIBLE_DEVICES=$gpu "$PY" -u scripts/run_experiments.py --sweep "${SWEEPS[@]}" --shard "$i/$N" \
      "${SKIP[@]}" --only "$ONLY" $([[ "$REVERSE" == 1 ]] && echo --reverse) --max-hours "$HOURS" > "logs/kaggle_shard$i.log" 2>&1 &
  pids+=($!)
  echo "  share $i/$N on GPU $gpu -> logs/kaggle_shard$i.log"
done
sleep 60; head -n 1 logs/kaggle_shard*.log      # the 'queue id' - must match the main laptop's
for p in "${pids[@]}"; do wait "$p" || true; done
tail -n 3 logs/kaggle_shard*.log

say "Export"
STAMP=$(date +%Y%m%d-%H%M)
"$PY" scripts/export_runs.py --out "$OUT/fedguard_runs_${FEDGUARD_MACHINE}_${STAMP}.zip"
cp logs/kaggle_shard*.log "$OUT/" 2>/dev/null || true
echo "DONE - download the zip from the Output tab and run on the main laptop:"
echo "  python scripts/import_runs.py <zip>"
