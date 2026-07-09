#!/usr/bin/env bash
# Centralized upper-bound baselines for all model classes.
# (3 oversamplers × 6 models × |SEEDS| = 18 runs per seed.)

source "$(dirname "${BASH_SOURCE[0]}")/_run_helpers.sh"

LOG_DIR="${DATASET}/centralized"

for SEED in ${SEEDS}; do
  for OS in smote adasyn none; do
    run_one "centralized_lr_${OS}_seed${SEED}" "${LOG_DIR}" -- \
      python -m experiments.centralized_baseline.run_lr \
        --dataset "${DATASET}" --oversampling "${OS}" --random_seed "${SEED}" --use_wandb true

    run_one "centralized_svm_${OS}_seed${SEED}" "${LOG_DIR}" -- \
      python -m experiments.centralized_baseline.run_svm \
        --dataset "${DATASET}" --oversampling "${OS}" --random_seed "${SEED}" --use_wandb true

    run_one "centralized_gbm_${OS}_seed${SEED}" "${LOG_DIR}" -- \
      python -m experiments.centralized_baseline.run_gbm \
        --dataset "${DATASET}" --oversampling "${OS}" --random_seed "${SEED}" --use_wandb true

    run_one "centralized_xgb_${OS}_seed${SEED}" "${LOG_DIR}" -- \
      python -m experiments.centralized_baseline.run_xgb \
        --dataset "${DATASET}" --oversampling "${OS}" --random_seed "${SEED}" --use_wandb true

    run_one "centralized_ffd_${OS}_seed${SEED}" "${LOG_DIR}" -- \
      python -m experiments.centralized_baseline.run_ffd \
        --dataset "${DATASET}" --oversampling "${OS}" --random_seed "${SEED}" --use_wandb true

    run_one "centralized_bert_fraud_${OS}_seed${SEED}" "${LOG_DIR}" -- \
      python -m experiments.centralized_baseline.run_bert_fraud \
        --dataset "${DATASET}" --oversampling "${OS}" --random_seed "${SEED}" --use_wandb true
  done
done

echo
echo "[run_centralized] done."
