#!/usr/bin/env bash
# =============================================================================
# FedGuard - one-command setup + training on a Linux (Ubuntu/Debian) laptop
# -----------------------------------------------------------------------------
# First time on the friend's laptop:
#   git clone https://github.com/<your-username>/fedguard.git ~/fedguard
#   cd ~/fedguard
#   bash scripts/linux/bootstrap_and_train.sh
#
# Modes:
#   bash scripts/linux/bootstrap_and_train.sh                 setup + data + tests + full training
#   bash scripts/linux/bootstrap_and_train.sh --smoke         setup + data + tests + tiny 2-round run
#   bash scripts/linux/bootstrap_and_train.sh --setup-only    install everything, do not train
#   bash scripts/linux/bootstrap_and_train.sh --push-results  send results/master.csv + figures to GitHub
#
# Options (environment variables, all optional):
#   SWEEP=configs/sweeps/main.yaml   which experiment list to run (default shown)
#   DATA_URL=https://.../file.zip    direct download link for the dataset
#   KAGGLE_DATASET=owner/slug        or: download with the Kaggle API (needs ~/.kaggle/kaggle.json)
#   NO_TMUX=1                        run in this terminal instead of a background tmux session
#
# The script is safe to run again: finished steps are skipped and finished
# experiment runs are not repeated (run_experiments.py --resume).
# =============================================================================
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
PY_VERSION="3.11"
SWEEP="${SWEEP:-configs/sweeps/main.yaml}"
SESSION="fedguard"
LOG_DIR="$REPO_DIR/logs"

say()  { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m[warn] %s\033[0m\n' "$*"; }
die()  { printf '\033[1;31m[error] %s\033[0m\n' "$*"; exit 1; }
trap 'die "Stopped at line $LINENO. Fix the message above, then run the script again."' ERR

check_os() {
  [[ "$(uname -s)" == "Linux" ]] || die "This script is for Linux. On Windows use scripts/windows/setup_windows.ps1"
  command -v apt-get >/dev/null || warn "apt-get not found: install git, curl, tmux and unzip with your package manager."
}

install_system_packages() {
  local need=() p
  for p in git curl tmux unzip; do command -v "$p" >/dev/null || need+=("$p"); done
  if ((${#need[@]})); then
    say "Installing system packages: ${need[*]} (asks for the sudo password once)"
    sudo apt-get update -y
    sudo apt-get install -y "${need[@]}"
  fi
}

install_uv() {
  export PATH="$HOME/.local/bin:$PATH"
  if ! command -v uv >/dev/null; then
    say "Installing uv (downloads Python $PY_VERSION and packages)"
    curl -LsSf https://astral.sh/uv/install.sh | sh
  fi
  command -v uv >/dev/null || die "uv is not on PATH. Open a new terminal and run the script again."
}

update_repo() {
  cd "$REPO_DIR"
  if [[ -d .git ]]; then
    say "Getting the latest code (git pull)"
    git pull --ff-only || warn "git pull failed (local changes?). Continuing with the code already here."
  fi
}

make_env() {
  cd "$REPO_DIR"
  say "Creating the Python $PY_VERSION environment in .venv"
  [[ -d .venv ]] || uv venv --python "$PY_VERSION" .venv
  # shellcheck disable=SC1091
  source .venv/bin/activate

  local req="requirements.txt"
  [[ -f "$req" ]] || req="requirements.in"
  [[ -f "$req" ]] || die "No requirements.txt or requirements.in in the repo root."

  # A lock file made on Windows may pin torch with a "+cu128"/"+cpu" tag or list Windows-only
  # packages. Clean those lines so the same file works here. On Linux, the normal PyPI torch
  # wheel already includes CUDA support.
  local clean="$LOG_DIR/requirements.linux.txt"
  mkdir -p "$LOG_DIR"
  sed -E 's/^(torch[a-z]*==[0-9.]+)\+[A-Za-z0-9]+/\1/' "$req" \
    | grep -viE '^(pywin32|pywinpty|pypiwin32|wmi)([=<> ;]|$)' > "$clean"
  say "Installing packages from $req"
  uv pip install -r "$clean"
  if [[ -f pyproject.toml ]]; then
    uv pip install -e . >/dev/null || warn "Could not install the repo as a package (pip install -e .)."
  fi
}

check_gpu() {
  say "Checking PyTorch and the GPU"
  python - <<'PY'
import torch
print("torch", torch.__version__, "| CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("No GPU found: training will use the CPU (slower, still fine for this small model).")
PY
  if [[ -f scripts/verify_setup.py ]]; then python scripts/verify_setup.py; fi
}

get_data() {
  cd "$REPO_DIR"
  mkdir -p data/raw data/processed
  if [[ -n "$(find data/processed -name '*.npz' -print -quit)" ]]; then
    say "Processed data already present - skipping download"
    return
  fi
  if [[ -n "$(find data/raw -name '*.csv' -print -quit)" ]]; then
    say "Raw CSV already in data/raw - skipping download"
    return
  fi
  if [[ -n "${DATA_URL:-}" ]]; then
    say "Downloading the dataset from DATA_URL"
    curl -L --fail --retry 3 -o data/raw/dataset.download "$DATA_URL"
    if unzip -tq data/raw/dataset.download >/dev/null 2>&1; then
      unzip -oq data/raw/dataset.download -d data/raw && rm -f data/raw/dataset.download
    else
      mv data/raw/dataset.download data/raw/dataset.csv
    fi
  elif [[ -n "${KAGGLE_DATASET:-}" ]]; then
    [[ -f "$HOME/.kaggle/kaggle.json" ]] || die "Put your Kaggle API token in ~/.kaggle/kaggle.json first."
    say "Downloading $KAGGLE_DATASET with the Kaggle API"
    uv pip install kaggle >/dev/null
    kaggle datasets download -d "$KAGGLE_DATASET" -p data/raw --unzip
  else
    die "No dataset found. Do ONE of these, then run the script again:
   1) DATA_URL='<direct link>' bash scripts/linux/bootstrap_and_train.sh
   2) KAGGLE_DATASET='<owner/slug>' bash scripts/linux/bootstrap_and_train.sh
   3) From Windows PowerShell: scp .\\NF-UNSW-NB15-v2.csv <user>@<friend-ip>:~/fedguard/data/raw/"
  fi
  [[ -n "$(find data/raw -name '*.csv' -print -quit)" ]] || die "Download finished but no .csv file was found in data/raw."
}

prepare_data() {
  cd "$REPO_DIR"
  if [[ -n "$(find data/processed -name '*.npz' -print -quit)" ]]; then return; fi
  [[ -f scripts/prepare_data.py ]] || die "scripts/prepare_data.py is missing (it is built in Phase 2)."
  say "Cleaning the data (leakage-safe pipeline)"
  python scripts/prepare_data.py
  if [[ -f scripts/make_partitions.py ]]; then
    say "Making the non-IID client partitions"
    python scripts/make_partitions.py
  fi
}

run_tests() {
  cd "$REPO_DIR"
  if [[ -d tests ]]; then
    say "Running the unit tests (training only starts if they pass)"
    python -m pytest -q
  fi
}

start_training() {
  local mode="$1"
  cd "$REPO_DIR"
  mkdir -p "$LOG_DIR"
  [[ -f scripts/run_experiments.py ]] || die "scripts/run_experiments.py is missing (it is built in Phase 4). Use --setup-only until then."

  local sweep="$SWEEP" stamp log runner
  [[ "$mode" == "--smoke" ]] && sweep="configs/sweeps/smoke.yaml"
  [[ -f "$sweep" ]] || die "Sweep file $sweep not found."
  stamp="$(date +%Y%m%d_%H%M%S)"
  log="$LOG_DIR/train_$stamp.log"
  runner="$LOG_DIR/run_$stamp.sh"

  cat > "$runner" <<EOF
#!/usr/bin/env bash
set -o pipefail
cd "$REPO_DIR"
source .venv/bin/activate
echo "started: \$(date)" | tee -a "$log"
python scripts/run_experiments.py --sweep "$sweep" --resume 2>&1 | tee -a "$log"
status=\${PIPESTATUS[0]}
if [[ -f scripts/collect_results.py ]]; then python scripts/collect_results.py 2>&1 | tee -a "$log"; fi
if [[ -f scripts/make_plots.py ]]; then python scripts/make_plots.py 2>&1 | tee -a "$log"; fi
echo "finished: \$(date) (exit code \$status)" | tee -a "$log"
EOF

  # Keep the laptop awake (and ignore the lid switch) while training, if systemd allows it.
  local launcher="bash $runner"
  if systemd-inhibit --what=sleep:idle:handle-lid-switch --who=fedguard --why=test true >/dev/null 2>&1; then
    launcher="systemd-inhibit --what=sleep:idle:handle-lid-switch --who=fedguard --why=fedguard-training bash $runner"
  else
    warn "Could not block sleep automatically. Turn off sleep in Settings > Power and keep the charger plugged in."
  fi

  if [[ -n "${NO_TMUX:-}" ]]; then
    say "Training in this terminal (log: $log)"
    $launcher
    return
  fi
  if tmux has-session -t "$SESSION" 2>/dev/null; then
    die "A training session is already running. See it with: tmux attach -t $SESSION"
  fi
  tmux new-session -d -s "$SESSION" "$launcher; exec bash"
  say "Training started in the background."
  cat <<EOF

  Watch it live      : tmux attach -t $SESSION      (leave again with Ctrl+b then d)
  Log file           : $log
  Results per run    : results/runs/<run_id>/metrics.csv
  When it finishes   : bash scripts/linux/bootstrap_and_train.sh --push-results
  You can close this terminal or the SSH connection; training keeps running.
EOF
}

push_results() {
  cd "$REPO_DIR"
  [[ -f results/master.csv ]] || die "results/master.csv not found yet (training not finished?)."
  local branch
  branch="results/$(hostname)-$(date +%Y%m%d-%H%M)"
  say "Pushing results to the GitHub branch $branch"
  git checkout -b "$branch"
  git add -f results/master.csv
  [[ -d results/figures ]] && git add -f results/figures
  git commit -m "Experiment results from $(hostname) on $(date +%F)"
  git push -u origin "$branch"
  git checkout -
  cat <<EOF

  Done. On your Windows laptop run:
    git fetch
    git checkout origin/$branch -- results/
EOF
}

main() {
  local mode="${1:-train}"
  case "$mode" in
    -h|--help) sed -n '2,25p' "$0"; exit 0 ;;
    --push-results) push_results; exit 0 ;;
    train|--smoke|--setup-only) ;;
    *) die "Unknown option '$mode'. Use --help." ;;
  esac
  check_os
  install_system_packages
  install_uv
  update_repo
  make_env
  check_gpu
  if [[ "$mode" == "--setup-only" ]]; then say "Setup finished. Nothing was trained."; exit 0; fi
  get_data
  prepare_data
  run_tests
  start_training "$mode"
}

main "$@"; exit
