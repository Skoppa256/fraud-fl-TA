#!/usr/bin/env bash
# Orchestrates the full RQ1 sweep:
#   centralized upper bounds → fedxgbllr → ffd → gbm → svm → lr
#
# Each sub-script tees stdout+stderr to results/logs/<model>/<run_name>.log
# and writes a summary CSV + per-round CSV via evaluation/results_writer.py.
#
# Usage:
#   bash experiments/run_all.sh                 # seed 42 only
#   SEEDS="42 123 2024" bash experiments/run_all.sh
#   SEEDS="42" SKIP_CENTRALIZED=1 bash experiments/run_all.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

: "${SEEDS:=42}"
: "${SKIP_CENTRALIZED:=0}"
: "${DATASET:=paysim}"

# ---------------------------------------------------------------------------
# Preflight checks
# ---------------------------------------------------------------------------
fail() { echo "[run_all] PREFLIGHT FAIL: $*" >&2; exit 1; }

echo "[run_all] === preflight ==="
echo "[run_all] cwd=${REPO_ROOT}"
echo "[run_all] SEEDS=${SEEDS}"
echo "[run_all] DATASET=${DATASET}"

# 1. Conda environment is active and named (or at least, python is available).
if [[ -z "${CONDA_DEFAULT_ENV:-}" ]]; then
  echo "[run_all] WARN: \$CONDA_DEFAULT_ENV is empty — is the conda env active?"
  echo "[run_all]       Expected: 'fraud-fl'. Continuing anyway."
else
  echo "[run_all] conda env: ${CONDA_DEFAULT_ENV}"
fi

# 2. Python and core deps importable.
python -c "import flwr, torch, xgboost, sklearn, pandas, imblearn, shap, wandb, hydra, yaml; print('[run_all] deps OK')" \
  || fail "Python imports failed — run: pip install -r requirements.txt && pip install -e models/fedxgbllr/"

# 3. Selected dataset CSV present.
[[ -f "data/${DATASET}/${DATASET}.csv" ]] \
  || fail "Dataset CSV missing at data/${DATASET}/${DATASET}.csv (DATASET=${DATASET}) — provide it before running."
echo "[run_all] dataset: data/${DATASET}/${DATASET}.csv (present)"

# 4. W&B login (only if any run uses --use_wandb true, which all of them do).
if ! python -c "import wandb, sys; sys.exit(0 if wandb.api.api_key else 1)" 2>/dev/null; then
  fail "W&B not logged in — run: wandb login"
fi
echo "[run_all] W&B: logged in"

# 5. Results directory tree exists (created by setup but re-assert here).
mkdir -p results/logs/"${DATASET}"/{ffd,bert_fraud,fedxgbllr,lr,svm,gbm,centralized}
echo "[run_all] log dirs: ready"

echo "[run_all] === preflight OK ==="
echo

# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------
export SEEDS DATASET

if [[ "${SKIP_CENTRALIZED}" != "1" ]]; then
  echo "[run_all] >>> centralized baselines (upper bound reference)"
  bash "${SCRIPT_DIR}/run_centralized.sh"
fi

echo "[run_all] >>> FedXGBllr (the heaviest FL run — run early)"
bash "${SCRIPT_DIR}/run_fedxgbllr.sh"

echo "[run_all] >>> FFD"
bash "${SCRIPT_DIR}/run_ffd.sh"

echo "[run_all] >>> GBM best-model selection"
bash "${SCRIPT_DIR}/run_gbm.sh"

echo "[run_all] >>> FedAvg-SVM"
bash "${SCRIPT_DIR}/run_svm.sh"

echo "[run_all] >>> FedAvg-LR"
bash "${SCRIPT_DIR}/run_lr.sh"

echo
echo "[run_all] === sweep complete ==="
python -m experiments.status || true
echo
python -m experiments.collect_results
