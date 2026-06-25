#!/usr/bin/env bash
# Full bert_fraud sweep (12 conditions × |SEEDS|).
#
# Usage:
#   bash experiments/run_bert_fraud.sh                # seed 42 only
#   SEEDS="42 123 2024" bash experiments/run_bert_fraud.sh
#
# Every run writes:
#   results/logs/bert_fraud/<run_name>.log            (tee'd stdout+stderr)
#   results/logs/bert_fraud/<run_name>.csv            (summary; written by run.py)
#   results/logs/bert_fraud/<run_name>_rounds.csv     (per-round metrics)

source "$(dirname "${BASH_SOURCE[0]}")/_run_helpers.sh"

MODEL="bert_fraud"

for SEED in ${SEEDS}; do
  # IID × {smote, adasyn, none}
  for OS in smote adasyn none; do
    run_one "${MODEL}_iid_alpha-_${OS}_seed${SEED}" "${MODEL}" -- \
      python -m models.bert_fraud.run \
        --scheme iid --oversampling "${OS}" \
        --random_seed "${SEED}" --use_wandb true
  done
  # Dirichlet × {0.5, 1.0, 5.0} × {smote, adasyn, none}
  for ALPHA in 0.5 1.0 5.0; do
    for OS in smote adasyn none; do
      run_one "${MODEL}_dirichlet_alpha${ALPHA}_${OS}_seed${SEED}" "${MODEL}" -- \
        python -m models.bert_fraud.run \
          --scheme dirichlet --alpha "${ALPHA}" --oversampling "${OS}" \
          --random_seed "${SEED}" --use_wandb true
    done
  done
done

echo
echo "[run_bert_fraud] done. Aggregate with: python -m experiments.collect_results"
