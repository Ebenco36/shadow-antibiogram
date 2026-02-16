#!/usr/bin/env bash
#SBATCH --job-name=run_all_scripts
#SBATCH --output=run_all_scripts.out
#SBATCH --error=run_all_scripts.err
#SBATCH --time=5-00:00:00
#SBATCH --partition=main
#SBATCH --nodelist=hpc-node06
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8          # increase CPU threads if needed
#SBATCH --gres=gpu:1               # request 1 GPU
#SBATCH --mem=1000G                 # shorthand for GB

set -euo pipefail
export PYTHONUNBUFFERED=1

# ---------------- helpers ----------------
is_writable_dir() {
  local d="$1"
  [[ -d "$d" ]] || return 1
  [[ -w "$d" ]] || return 1
  local t; t="$(mktemp -p "$d" .wtest.XXXX 2>/dev/null)" || return 1
  rm -f "$t"; return 0
}
have_cmd() { command -v "$1" >/dev/null 2>&1; }

runpy() {
  # Use srun if under Slurm and srun exists; else run python directly.
  if [[ -n "${SLURM_JOB_ID:-}" ]] && have_cmd srun; then
    srun -u "$PY" "$@"
  else
    "$PY" "$@"
  fi
}
runpymod() {
  if [[ -n "${SLURM_JOB_ID:-}" ]] && have_cmd srun; then
    srun -u "$PY" -m "$1"
  else
    "$PY" -m "$1"
  fi
}

# ---------------- working dir ----------------
# Prefer the directory the job was submitted from on Slurm; otherwise script dir.
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  cd "$SLURM_SUBMIT_DIR"
else
  cd "$(dirname "$0")"
fi
PROJECT_ROOT="${PROJECT_ROOT:-$PWD}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
SRC_DIR="${SRC_DIR:-$PROJECT_ROOT/src}"

# Guard: do not run from /var/spool
if [[ "$PROJECT_ROOT" == /var/spool/* ]]; then
  echo "PROJECT_ROOT is under /var/spool (read-only)."
  echo "   On Slurm, submit with:  sbatch --chdir=/path/to/your/repo run_anywhere.sh"
  exit 2
fi

# Ensure src exists
if [[ ! -d "$SRC_DIR" ]]; then
  echo "SRC_DIR does not exist: $SRC_DIR"
  echo "   Set --chdir to your repo root or export SRC_DIR=/path/to/repo/src"
  exit 2
fi

echo "Project root : $PROJECT_ROOT"
echo "Source dir   : $SRC_DIR"

# ---------------- threads / BLAS ----------------
# On Slurm use allocated CPUs; else local CPU count.
if [[ -n "${SLURM_CPUS_PER_TASK:-}" ]]; then
  CPU_COUNT="$SLURM_CPUS_PER_TASK"
else
  if have_cmd sysctl; then
    CPU_COUNT="$(sysctl -n hw.logicalcpu 2>/dev/null || echo 4)"
  else
    CPU_COUNT="${CPU_COUNT:-4}"
  fi
fi
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$CPU_COUNT}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$CPU_COUNT}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-$CPU_COUNT}"
export NUMEXPR_MAX_THREADS="${NUMEXPR_MAX_THREADS:-$CPU_COUNT}"

# ---------------- venv location ----------------
if [[ -z "${VENV_DIR:-}" ]]; then
  if is_writable_dir "$PROJECT_ROOT"; then
    VENV_DIR="$PROJECT_ROOT/.venv"
  else
    VENV_DIR="$HOME/.venvs/$(basename "$PROJECT_ROOT")"
  fi
fi
mkdir -p "$(dirname "$VENV_DIR")"
is_writable_dir "$(dirname "$VENV_DIR")" || { echo "Not writable: $(dirname "$VENV_DIR")"; exit 13; }

# ---------------- create venv (prefer --upgrade-deps if supported) ----------------
if ! "$PYTHON_BIN" -m venv --help 2>&1 | grep -q -- '--upgrade-deps'; then
  [[ -x "$VENV_DIR/bin/python" ]] || "$PYTHON_BIN" -m venv "$VENV_DIR"
else
  if [[ ! -x "$VENV_DIR/bin/python" ]]; then
    "$PYTHON_BIN" -m venv --upgrade-deps "$VENV_DIR" || "$PYTHON_BIN" -m venv "$VENV_DIR"
  fi
fi
PY="$VENV_DIR/bin/python"

echo "Venv dir     : $VENV_DIR"

# ---------------- bootstrap pip if missing ----------------
if ! "$PY" -m pip --version >/dev/null 2>&1; then
  echo "pip missing in venv → bootstrapping (ensurepip or get-pip.py)"
  if "$PY" -m ensurepip --upgrade >/dev/null 2>&1; then
    :
  else
    GP="${PROJECT_ROOT}/get-pip.py"
    if [[ ! -f "$GP" ]]; then
      # try to download if allowed
      if have_cmd curl; then
        curl -fsSL https://bootstrap.pypa.io/get-pip.py -o "$GP"
      elif have_cmd wget; then
        wget -qO "$GP" https://bootstrap.pypa.io/get-pip.py
      else
        echo "Need $PROJECT_ROOT/get-pip.py (no curl/wget)."
        exit 3
      fi
    fi
    "$PY" "$GP"
  fi
fi

# ---------------- pip policy ----------------
# default: upgrade packages each run
# PIP_REINSTALL=1 → force reinstall each run (no-deps)
PIP_FLAGS="--upgrade --no-cache-dir"
if [[ "${PIP_REINSTALL:-0}" == "1" ]]; then
  PIP_FLAGS="--force-reinstall --no-deps --no-cache-dir"
fi

REQ="$PROJECT_ROOT/requirements.txt"
# contourpy hot-fix (portable sed for mac/BSD)
if [[ -f "$REQ" ]] && grep -qE '^contourpy==1\.3\.1$' "$REQ"; then
  echo "Adjusting contourpy pin to 1.3.0"
  sed -i.bak 's/^contourpy==1\.3\.1$/contourpy==1.3.0/' "$REQ" && rm -f "$REQ.bak"
fi

# ---------------- upgrade tooling & install requirements ----------------
runpy -m pip install --upgrade pip setuptools wheel ${PIP_EXTRA:-}
if [[ -f "$REQ" ]]; then
  echo "Running pip install ${PIP_FLAGS} -r requirements.txt"
  runpy -m pip install $PIP_FLAGS -r "$REQ" ${PIP_EXTRA:-}
fi

# ---------------- debug ----------------
echo "PYTHON: $(runpy - <<'PY'
import sys; print(sys.executable)
PY
)"
runpy --version
runpy -m pip list | sed -n '1,40p'

# ---------------- pyarrow preflight ----------------
set +e
runpy - <<'PY'
import importlib, sys
try:
    pa = importlib.import_module("pyarrow")
    print("pyarrow from:", getattr(pa, "__file__", None))
    print("pyarrow version:", getattr(pa, "__version__", None))
    if not hasattr(pa, "__version__"):
        print("FATAL: Shadowed 'pyarrow' detected (local pyarrow.py/pyarrow/).", file=sys.stderr)
        sys.exit(2)
except Exception as e:
    print("WARNING: pyarrow import failed:", e, file=sys.stderr)
PY
rc=$?; set -e
if [[ $rc -ne 0 ]]; then
  echo "Preflight failed. Remove/rename any local 'pyarrow*' files/folders and re-run."
  exit $rc
fi

# ---------------- modules to run (edit as needed) ----------------
# MODULES=(
  # "src.runners.Preload" # optional data preloading. Not needed since we already have preprocessed data.
  # "src.runners.DataProcessing" # optional full data processing. Not needed since we already have preprocessed data.
  # "src.runners.GenerateTables"
  # "src.controllers.AMR.scripts.run_comprehensive_analysis"
  # "src.runners.CompleteTemporalAnalysis"
  # "src.runners.SummaryStatistics"
# )

MODULES=(
  # "src.runners.Preload"
  # "src.runners.DataProcessing"
  # "src.AntibioticResistanceAggregator"
  "src.controllers.AMR.scripts.run_comprehensive_analysis"
  # "src.controllers.AMR.evaluation.AMR_network_benchmark_runner"
)

for m in "${MODULES[@]}"; do
  echo "──────────────────────────────────────────────────────"
  echo "▶ Running: $m"
  if ! runpymod "$m"; then
    echo "$m failed. Aborting."
    exit 1
  fi
done

echo "All scripts executed successfully."
