#!/usr/bin/env bash
# Full FFD sweep (12 conditions × |SEEDS|).
#
# Usage:
#   bash experiments/run_ffd.sh                # seed 42 only
#   SEEDS="42 123 2024" bash experiments/run_ffd.sh
#
# Every run writes:
#   results/logs/ffd/<run_name>.log            (tee'd stdout+stderr)
#   results/logs/ffd/<run_name>.csv            (summary; written by run.py)
#   results/logs/ffd/<run_name>_rounds.csv     (per-round metrics)

source "$(dirname "${BASH_SOURCE[0]}")/_run_helpers.sh"

MODEL="ffd"

for SEED in ${SEEDS}; do
  # IID × {smote, adasyn, none}
  for OS in smote adasyn none; do
    run_one "${MODEL}_iid_alpha-_${OS}_seed${SEED}" "${DATASET}/${MODEL}" -- \
      python -m models.ffd.run \
        --dataset "${DATASET}" \
        --scheme iid --oversampling "${OS}" \
        --random_seed "${SEED}" --use_wandb true
  done
  # Dirichlet × {0.5, 1.0, 5.0} × {smote, adasyn, none}
  for ALPHA in 0.5 1.0 5.0; do
    for OS in smote adasyn none; do
      run_one "${MODEL}_dirichlet_alpha${ALPHA}_${OS}_seed${SEED}" "${DATASET}/${MODEL}" -- \
        python -m models.ffd.run \
        --dataset "${DATASET}" \
          --scheme dirichlet --alpha "${ALPHA}" --oversampling "${OS}" \
          --random_seed "${SEED}" --use_wandb true
    done
  done
done

echo
echo "[run_ffd] done. Aggregate with: python -m experiments.collect_results"
