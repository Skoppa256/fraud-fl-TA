# Centralized Baseline

Non-federated reference runs for the four models studied in this thesis
(LR, SVM, GBM, XGBoost). These provide the upper-bound numbers that the
federated arms in `models/fedavg_lr/`, `models/fedavg_svm/`,
`models/gbm_bestmodel/`, and `models/fedxgbllr/` are compared against
for RQ1.

## Purpose

For each model, the centralized baseline answers the question:
*if this same model were trained on the entire training set without any
partitioning or aggregation, what performance would we get?* The gap
between the centralized number and its federated counterpart is the
cost of federation.

## Methodological note

Each script uses the **identical model class and hyperparameters** as
its federated counterpart. The only differences are operational:

- No Flower simulation, no client splitting, no aggregation.
- The model trains directly on the full 70% train split returned by
  `preprocessing.paysim.load_paysim`.
- Evaluation is on the 15% val and 15% test splits, with the same
  metrics (AUPRC primary; F1, Precision, Recall at threshold 0.5).

| Model    | Class                                | Key hyperparameters                                            |
|----------|--------------------------------------|----------------------------------------------------------------|
| LR       | `sklearn.linear_model.LogisticRegression` | `C=1.0, max_iter=1000`                                    |
| SVM      | `sklearn.svm.LinearSVC`              | `C=1.0, max_iter=1000` (AUPRC via `decision_function`)         |
| GBM      | `sklearn.ensemble.HistGradientBoostingClassifier` | `max_iter=100, learning_rate=0.1, max_depth=6`    |
| XGBoost  | `xgboost.XGBClassifier`              | `n_estimators=50, max_depth=6, learning_rate=0.1, subsample=0.8` |

## Global oversampling

In the federated arms, SMOTE or ADASYN is applied **per client** after
Dirichlet partitioning. Per-client resampling is constrained by each
client's local minority pool (it can fail entirely when a client has
fewer than `k_neighbors+1` fraud samples, or — for ADASYN — when the
local density landscape contains no candidates).

In the centralized baseline, the chosen sampler is applied **once
globally** on the full training set using

```
imblearn.over_sampling.SMOTE (sampling_strategy="auto", k_neighbors=5)
imblearn.over_sampling.ADASYN(sampling_strategy="auto", n_neighbors=5)
```

which oversamples the minority class to roughly 1:1. This is strictly
better than per-client resampling because the synthetic neighbours are
drawn from the full minority distribution, not a single client's
subset. The centralized arm therefore represents the upper bound for
imbalance handling.

## How to run

Every script lives at module path
`experiments.centralized_baseline.run_<model>` and accepts the same
four flags:

```bash
--oversampling   (str,  default="smote", one of: smote, adasyn, none)
--random_seed    (int,  default=42)
--use_wandb      (bool, default=false)
--wandb_project  (str,  default="fraud-fl-TA")
```

Run each from the project root (`fraud-fl-TA/`):

```bash
# Logistic Regression — SMOTE / ADASYN / no oversampling
python -m experiments.centralized_baseline.run_lr \
  --oversampling smote  --random_seed 42 --use_wandb false
python -m experiments.centralized_baseline.run_lr \
  --oversampling adasyn --random_seed 42 --use_wandb false
python -m experiments.centralized_baseline.run_lr \
  --oversampling none   --random_seed 42 --use_wandb false

# Linear SVM
python -m experiments.centralized_baseline.run_svm \
  --oversampling smote --random_seed 42 --use_wandb false

# HistGBM
python -m experiments.centralized_baseline.run_gbm \
  --oversampling smote --random_seed 42 --use_wandb false

# XGBoost
python -m experiments.centralized_baseline.run_xgb \
  --oversampling smote --random_seed 42 --use_wandb false
```

To log to W&B, pass `--use_wandb true`; the run name is
`centralized_<model>_seed<seed>` and the default project is
`fraud-fl-TA`.

## Results

Fill in once runs complete. `time` is wall-clock training time in
seconds, as printed by each script.

| Model   | oversampling | test_auprc | test_f1 | test_precision | test_recall | time |
|---------|--------------|------------|---------|----------------|-------------|------|
| LR      | smote        |            |         |                |             |      |
| LR      | adasyn       |            |         |                |             |      |
| LR      | none         |            |         |                |             |      |
| SVM     | smote        |            |         |                |             |      |
| SVM     | adasyn       |            |         |                |             |      |
| SVM     | none         |            |         |                |             |      |
| GBM     | smote        |            |         |                |             |      |
| GBM     | adasyn       |            |         |                |             |      |
| GBM     | none         |            |         |                |             |      |
| XGBoost | smote        |            |         |                |             |      |
| XGBoost | adasyn       |            |         |                |             |      |
| XGBoost | none         |            |         |                |             |      |

When `--oversampling none`, some linear models can produce
`test_auprc=0.0` or `test_recall=0.0` purely from the class imbalance
(threshold 0.5 with a heavily skewed prior). This is expected and is
the reason the SMOTE / ADASYN arms exist.
