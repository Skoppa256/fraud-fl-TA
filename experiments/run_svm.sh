#!/usr/bin/env bash
# Full FedAvg-SVM sweep (12 conditions × |SEEDS|).

source "$(dirname "${BASH_SOURCE[0]}")/_run_helpers.sh"

MODEL="svm"
NUM_ROUNDS=20

for SEED in ${SEEDS}; do
  for OS in smote adasyn none; do
    run_one "${MODEL}_iid_alpha-_${OS}_seed${SEED}" "${MODEL}" -- \
      python -m models.fedavg_svm.run \
        --scheme iid --num_rounds "${NUM_ROUNDS}" --num_clients 5 \
        --local_epochs 1 --oversampling "${OS}" \
        --random_seed "${SEED}" --use_wandb true
  done
  for ALPHA in 0.5 1.0 5.0; do
    for OS in smote adasyn none; do
      run_one "${MODEL}_dirichlet_alpha${ALPHA}_${OS}_seed${SEED}" "${MODEL}" -- \
        python -m models.fedavg_svm.run \
          --scheme dirichlet --alpha "${ALPHA}" \
          --num_rounds "${NUM_ROUNDS}" --num_clients 5 \
          --local_epochs 1 --oversampling "${OS}" \
          --random_seed "${SEED}" --use_wandb true
    done
  done
done

echo
echo "[run_svm] done."
