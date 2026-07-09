#!/usr/bin/env bash
# Combined orchestration for ONE dataset: every model under THREE scenarios —
#   1) centralized            (no partitioning; one run per centralized arm)
#   2) federated IID
#   3) federated Dirichlet, alpha=1.0
# One run each (no seed/oversampler sweep). Fixed shared config, passed
# EXPLICITLY on every command (never relying on defaults):
#   oversampling=smote  sampling_strategy=0.01  num_clients=5 (federated only)
#   num_rounds=20 (federated only)  random_seed=42  use_wandb=true
#
# Standalone addition: it only calls the existing run entrypoints through
# run_one() from _run_helpers.sh (no model code / defaults / helpers touched).
# Runs sequentially — creditcard is large, so no parallelism. Logs land in
# results/logs/<dataset>/<model>/ (federated) and
# results/logs/<dataset>/centralized/ (centralized), via the dataset-nesting
# already built into run_one's log_subdir.
#
# Dry-run (print every command that WOULD run, execute nothing):
#   DRYRUN=1 bash experiments/run_all_centralized_iid_dirichlet1.0_creditcard.sh

source "$(dirname "${BASH_SOURCE[0]}")/_run_helpers.sh"

# --- Dataset for THIS script. Authoritative: overrides the _run_helpers.sh
#     default so the script is self-contained and not env-dependent.
DATASET="creditcard"

# --- Fixed shared config (single value each; no sweeps here).
OS="smote"
SAMPLING_STRATEGY="0.01"
NUM_CLIENTS=5
NUM_ROUNDS=20
SEED=42
ALPHA="1.0"

# --- Preflight: the dataset CSV must exist. Abort with a clear message if not
#     (but keep going under DRYRUN so the command list can still be eyeballed).
CSV="${REPO_ROOT}/data/${DATASET}/${DATASET}.csv"
if [[ ! -f "${CSV}" ]]; then
  echo "[preflight] dataset CSV not found: ${CSV}" >&2
  if [[ "${DRYRUN:-0}" == "1" ]]; then
    echo "[preflight] continuing anyway because DRYRUN=1." >&2
  else
    echo "[preflight] aborting — place the CSV there before running." >&2
    exit 1
  fi
fi

# --- Progress + dry-run aware launcher. Wraps run_one (does NOT modify the
#     helper). Echoes a progress line before each run; under DRYRUN it prints
#     the fully-quoted command instead of executing it.
STEP=0
TOTAL=18
launch() {
  local run_name="$1"; shift
  local log_subdir="$1"; shift
  [[ "${1:-}" == "--" ]] || { echo "launch: expected -- before command" >&2; return 64; }
  shift
  STEP=$((STEP + 1))
  echo
  echo ">>> [${STEP}/${TOTAL}] ${DATASET} | ${run_name}  (log: results/logs/${log_subdir}/)"
  if [[ "${DRYRUN:-0}" == "1" ]]; then
    printf '    '; printf '%q ' "$@"; printf '\n'
  else
    run_one "${run_name}" "${log_subdir}" -- "$@"
  fi
}

# --- Reusable federated argparse arm: one IID run + one Dirichlet(alpha) run.
#     $1 = MODEL token (matches existing run_*.sh log/run_name convention)
#     $2 = python module for `python -m`
fed_argparse() {
  local MODEL="$1" MODULE="$2"
  launch "${MODEL}_iid_alpha-_${OS}_seed${SEED}" "${DATASET}/${MODEL}" -- \
    python -m "${MODULE}" \
      --dataset "${DATASET}" \
      --scheme iid --num_rounds "${NUM_ROUNDS}" --num_clients "${NUM_CLIENTS}" \
      --oversampling "${OS}" --sampling_strategy "${SAMPLING_STRATEGY}" \
      --random_seed "${SEED}" --use_wandb true
  launch "${MODEL}_dirichlet_alpha${ALPHA}_${OS}_seed${SEED}" "${DATASET}/${MODEL}" -- \
    python -m "${MODULE}" \
      --dataset "${DATASET}" \
      --scheme dirichlet --alpha "${ALPHA}" \
      --num_rounds "${NUM_ROUNDS}" --num_clients "${NUM_CLIENTS}" \
      --oversampling "${OS}" --sampling_strategy "${SAMPLING_STRATEGY}" \
      --random_seed "${SEED}" --use_wandb true
}

echo "=== run_all: ${DATASET} | centralized + federated(IID, Dirichlet alpha=${ALPHA}) ==="
echo "=== shared: oversampling=${OS} sampling_strategy=${SAMPLING_STRATEGY} num_clients=${NUM_CLIENTS} num_rounds=${NUM_ROUNDS} seed=${SEED} wandb=true ==="

# ---------------------------------------------------------------------------
# 1) CENTRALIZED (cheapest; no partition / clients / rounds).
#    Arms: lr, svm, gbm, xgb, ffd, bert_fraud
# ---------------------------------------------------------------------------
for ARM in lr svm gbm xgb ffd bert_fraud; do
  launch "centralized_${ARM}_${OS}_seed${SEED}" "${DATASET}/centralized" -- \
    python -m "experiments.centralized_baseline.run_${ARM}" \
      --dataset "${DATASET}" \
      --oversampling "${OS}" --sampling_strategy "${SAMPLING_STRATEGY}" \
      --random_seed "${SEED}" --use_wandb true
done

# ---------------------------------------------------------------------------
# 2) FEDERATED — cheap linear / tree arms first (lr, svm, gbm).
# ---------------------------------------------------------------------------
fed_argparse lr  models.fedavg_lr.run
fed_argparse svm models.fedavg_svm.run
fed_argparse gbm models.gbm_bestmodel.run

# ---------------------------------------------------------------------------
# 3) FEDERATED — NN arms (ffd, bert_fraud). They read batch_size/lr from their
#    own conf/base.yaml; we pass only the shared-config flags.
# ---------------------------------------------------------------------------
fed_argparse ffd        models.ffd.run
fed_argparse bert_fraud models.bert_fraud.run

# ---------------------------------------------------------------------------
# 4) FedXGBllr LAST (the ~90 min arm). Hydra overrides, NOT argparse.
#    num_clients stays 5 via clients=<ds>_5_clients.
# ---------------------------------------------------------------------------
FX_MODEL="fedxgbllr"
launch "${FX_MODEL}_iid_alpha-_${OS}_seed${SEED}" "${DATASET}/${FX_MODEL}" -- \
  python -m hfedxgboost.main \
    dataset="${DATASET}" clients="${DATASET}_5_clients" \
    run_experiment.num_rounds="${NUM_ROUNDS}" \
    dataset.oversampling.method="${OS}" \
    dataset.oversampling.sampling_strategy="${SAMPLING_STRATEGY}" \
    random_seed="${SEED}" use_wandb=true
launch "${FX_MODEL}_dirichlet_alpha${ALPHA}_${OS}_seed${SEED}" "${DATASET}/${FX_MODEL}" -- \
  python -m hfedxgboost.main \
    dataset="${DATASET}" clients="${DATASET}_5_clients" \
    run_experiment.num_rounds="${NUM_ROUNDS}" \
    dataset.non_iid.enabled=true dataset.non_iid.alpha="${ALPHA}" \
    dataset.oversampling.method="${OS}" \
    dataset.oversampling.sampling_strategy="${SAMPLING_STRATEGY}" \
    random_seed="${SEED}" use_wandb=true

echo
echo "[run_all_${DATASET}] done (${STEP} runs). Aggregate with: python -m experiments.collect_results"
