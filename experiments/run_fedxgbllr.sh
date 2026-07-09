#!/usr/bin/env bash
# Full FedXGBllr sweep (12 conditions × |SEEDS|). Uses Hydra CLI overrides
# instead of argparse — see models/fedxgbllr/hfedxgboost/conf/ for the keys.

source "$(dirname "${BASH_SOURCE[0]}")/_run_helpers.sh"

MODEL="fedxgbllr"
NUM_ROUNDS=50

for SEED in ${SEEDS}; do
  # IID × {smote, adasyn, none}: non_iid.enabled=false.
  for OS in smote adasyn none; do
    run_one "${MODEL}_iid_alpha-_${OS}_seed${SEED}" "${DATASET}/${MODEL}" -- \
      python -m hfedxgboost.main \
        dataset="${DATASET}" clients="${DATASET}_5_clients" \
        run_experiment.num_rounds="${NUM_ROUNDS}" \
        dataset.oversampling.method="${OS}" \
        random_seed="${SEED}" \
        use_wandb=true
  done
  # Dirichlet × {0.5, 1.0, 5.0} × {smote, adasyn, none}.
  for ALPHA in 0.5 1.0 5.0; do
    for OS in smote adasyn none; do
      run_one "${MODEL}_dirichlet_alpha${ALPHA}_${OS}_seed${SEED}" "${DATASET}/${MODEL}" -- \
        python -m hfedxgboost.main \
          dataset="${DATASET}" clients="${DATASET}_5_clients" \
          run_experiment.num_rounds="${NUM_ROUNDS}" \
          dataset.non_iid.enabled=true \
          dataset.non_iid.alpha="${ALPHA}" \
          dataset.oversampling.method="${OS}" \
          random_seed="${SEED}" \
          use_wandb=true
    done
  done
done

echo
echo "[run_fedxgbllr] done."
