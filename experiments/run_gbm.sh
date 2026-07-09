#!/usr/bin/env bash
# Full GBM best-model-selection sweep (12 conditions × |SEEDS|).

source "$(dirname "${BASH_SOURCE[0]}")/_run_helpers.sh"

MODEL="gbm"
NUM_ROUNDS=10

for SEED in ${SEEDS}; do
  for OS in smote adasyn none; do
    run_one "${MODEL}_iid_alpha-_${OS}_seed${SEED}" "${DATASET}/${MODEL}" -- \
      python -m models.gbm_bestmodel.run \
        --dataset "${DATASET}" \
        --scheme iid --num_rounds "${NUM_ROUNDS}" --num_clients 5 \
        --oversampling "${OS}" \
        --random_seed "${SEED}" --use_wandb true
  done
  for ALPHA in 0.5 1.0 5.0; do
    for OS in smote adasyn none; do
      run_one "${MODEL}_dirichlet_alpha${ALPHA}_${OS}_seed${SEED}" "${DATASET}/${MODEL}" -- \
        python -m models.gbm_bestmodel.run \
        --dataset "${DATASET}" \
          --scheme dirichlet --alpha "${ALPHA}" \
          --num_rounds "${NUM_ROUNDS}" --num_clients 5 \
          --oversampling "${OS}" \
          --random_seed "${SEED}" --use_wandb true
    done
  done
done

echo
echo "[run_gbm] done."
