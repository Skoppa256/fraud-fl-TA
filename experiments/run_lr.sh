#!/usr/bin/env bash
# Full FedAvg-LR sweep (12 conditions × |SEEDS|).
# See experiments/run_ffd.sh header for output paths and SEEDS override.

source "$(dirname "${BASH_SOURCE[0]}")/_run_helpers.sh"

MODEL="lr"
NUM_ROUNDS=20

for SEED in ${SEEDS}; do
  for OS in smote adasyn none; do
    run_one "${MODEL}_iid_alpha-_${OS}_seed${SEED}" "${DATASET}/${MODEL}" -- \
      python -m models.fedavg_lr.run \
        --dataset "${DATASET}" \
        --scheme iid --num_rounds "${NUM_ROUNDS}" --num_clients 5 \
        --local_epochs 1 --oversampling "${OS}" \
        --random_seed "${SEED}" --use_wandb true
  done
  for ALPHA in 0.5 1.0 5.0; do
    for OS in smote adasyn none; do
      run_one "${MODEL}_dirichlet_alpha${ALPHA}_${OS}_seed${SEED}" "${DATASET}/${MODEL}" -- \
        python -m models.fedavg_lr.run \
        --dataset "${DATASET}" \
          --scheme dirichlet --alpha "${ALPHA}" \
          --num_rounds "${NUM_ROUNDS}" --num_clients 5 \
          --local_epochs 1 --oversampling "${OS}" \
          --random_seed "${SEED}" --use_wandb true
    done
  done
done

echo
echo "[run_lr] done."
