# Federated Learning for Financial Fraud Detection — Comparative Study

> **Undergraduate Thesis (Tugas Akhir / TA)** — Institut Teknologi Sepuluh
> Nopember (ITS), Surabaya. This repository contains the source code,
> configurations, and experiment scaffolding for the thesis comparing
> federated learning approaches on mobile-money fraud detection.

A research project benchmarking five model classes — four federated and
their centralized upper bounds — on the PaySim mobile-money fraud dataset.
The goal is to compare classical FL (FedAvg over linear models), tree-ensemble
FL with learnable aggregation (FedXGBllr), a model-selection variant
(GBM with best-model promotion), and a CNN-based fraud detector (FFD)
under non-IID partitioning and extreme class imbalance.

The study targets **AUPRC** (Area Under the Precision–Recall Curve) as the
primary metric, because the PaySim fraud rate is ~0.13% and accuracy is
uninformative under this skew. F1, Precision, and Recall are reported as
supporting metrics. Explanations are produced per-client via SHAP.

---

## Tech stack

| Layer                       | Tool                                  |
|-----------------------------|---------------------------------------|
| Federated learning          | **Flower (`flwr`) 1.5.0** + Ray simulation |
| Deep learning               | **PyTorch 2.8** (+ torchmetrics) — CNN aggregator in FedXGBllr, FFD model |
| Classical ML                | **scikit-learn 1.5** (LR / SVM / HistGBM), **XGBoost 2.0** |
| Configuration               | **Hydra 1.3** + OmegaConf (FedXGBllr), YAML + argparse (others) |
| Class imbalance             | **imbalanced-learn** (SMOTE, ADASYN)  |
| Explainability              | **SHAP**                              |
| Experiment tracking         | **Weights & Biases (`wandb` 0.15)**   |
| Data / numerics             | pandas, NumPy, SciPy                  |
| Plotting                    | matplotlib, seaborn                   |
| Python                      | **3.10.x** (constraint from FedXGBllr) |

---

## Project structure

```
fraud-fl-TA/
├── data/
│   └── paysim/
│       └── paysim.csv             # PaySim mobile-money fraud CSV (~6.3M rows)
│
├── preprocessing/
│   ├── paysim.py                  # PaySim cleaning / feature engineering
│   ├── smote.py                   # Per-client local SMOTE wrapper
│   ├── adasyn.py                  # Per-client local ADASYN wrapper
│   └── oversampling.py            # Dispatch between SMOTE / ADASYN / none
│
├── partitioning/
│   └── dirichlet.py               # Dirichlet non-IID client partition (TBD)
│
├── models/
│   ├── fedxgbllr/                 # FedXGBllr (Flower hfedxgboost — Hydra CLI)
│   ├── fedavg_lr/                 # Logistic Regression + FedAvg
│   ├── fedavg_svm/                # Linear SVM + FedAvg
│   ├── gbm_bestmodel/             # HistGBM with server-side best-model selection
│   └── ffd/                       # FFD — Conv1D fraud detector (Yang et al., 2019)
│
├── evaluation/
│   ├── metrics.py                 # AUPRC/F1/Precision/Recall helpers (TBD)
│   └── shap_analysis.py           # SHAP explainability driver (TBD)
│
├── experiments/
│   ├── centralized_baseline/      # Upper-bound non-FL runs (LR/SVM/GBM/XGB/FFD)
│   ├── configs/                   # Hydra configs per experiment (TBD)
│   └── run_all.sh                 # Orchestrates the runs (TBD)
│
├── results/                       # Auto-generated CSVs, plots, model artifacts
│   └── visualizations/            # PCA plots per partition × oversampler
├── notebooks/                     # Exploratory and analysis (kaggle_setup.ipynb, pca_visualization.py)
│
├── requirements.txt               # Shared dependency lockset for all models
├── README.md                      # This file
└── .gitignore
```

---

## Environment setup

Python 3.10 is required (fedxgbllr pins `>=3.10.0, <3.11.0`). The canonical
setup uses conda for the interpreter and `pip` for packages:

```bash
conda create -n fraud-fl python=3.10 -y
conda activate fraud-fl
pip install -r requirements.txt
pip install -e models/fedxgbllr/
```

Verify the environment:

```bash
python -c "import flwr, torch, xgboost, sklearn, pandas, imblearn, shap, wandb, hydra; print('OK')"
```

**Note on `setuptools`**: `ray==2.6.3` (pulled in by `flwr[simulation]==1.5.0`)
still imports the legacy `pkg_resources` module, which `setuptools>=81` removed.
The requirements file pins `setuptools<81` to keep Ray's worker startup intact.

---

## Dataset

**PaySim** is a synthetic mobile-money transaction dataset created to study
financial fraud detection without exposing real customer data. It contains
~6.3 million transactions across 11 columns, of which roughly 8,000 are
labelled fraud (≈0.13% positive rate).

- **Source**: Kaggle — `ealaxi/paysim1`
  (https://www.kaggle.com/datasets/ealaxi/paysim1)
- **Place the file at**: `data/paysim/paysim.csv`
- **Schema** (11 columns):
  `step, type, amount, nameOrig, oldbalanceOrg, newbalanceOrig,
   nameDest, oldbalanceDest, newbalanceDest, isFraud, isFlaggedFraud`

The CSV is large (~493 MB) and is git-ignored — download it once locally.

---

## Models compared

1. **FedXGBllr** — Federated XGBoost with Learnable Learning Rates.
   Each client trains a local XGBoost ensemble; trees are exchanged with the
   server and concatenated into a global forest. A small 1-D CNN, trained
   federatedly under FedAvg, learns per-tree weights so the global ensemble
   produces calibrated predictions. (Baseline; complete in `models/fedxgbllr/`.)

2. **Logistic Regression + FedAvg** — Each client fits a standard logistic
   regression on its local partition. Model weights are averaged across
   clients each round using vanilla **FedAvg** aggregation. Linear, fast,
   and the natural FL baseline.

3. **Linear SVM + FedAvg** — Linear support-vector classifier (hinge loss),
   weights aggregated via **FedAvg**. Captures a different decision boundary
   from LR while staying within the FedAvg protocol so the comparison
   isolates *model class* from *aggregation strategy*.

4. **GBM with Best-Model Selection** — Each client trains a local Gradient
   Boosting Machine. Instead of averaging, the server evaluates submitted
   models on a held-out validation slice and **promotes the single best-AUPRC
   model** as the new global. This explores whether model selection beats
   parameter averaging when local distributions diverge sharply.

5. **FFD (Conv1D Fraud Detector)** — A small 1-D CNN architecture from
   Yang et al. (2019), adapted to PaySim's 13-feature tabular input.
   Lives in `models/ffd/`; weights are averaged via **FedAvg**.

All five FL arms use **Dirichlet partitioning** (`α` ∈ {0.5, 1.0, 5.0}) to induce
non-IID client splits, and per-client oversampling — either **SMOTE** or
**ADASYN**, selected via `--oversampling {smote, adasyn, none}` — to soften
the fraud-rate imbalance before training.

---

## Running experiments

Every run — federated or centralized — emits the same structured output:

| Artifact | Path | Schema |
|----------|------|--------|
| Stdout/stderr log | `results/logs/<model>/<run_name>.log` | Tee'd from the python process by the shell script |
| Summary CSV | `results/logs/<model>/<run_name>.csv` | `model, scheme, alpha, oversampling, random_seed, num_rounds, num_clients, best_round, best_val_*, test_*, timestamp, duration_seconds, run_name` |
| Per-round CSV | `results/logs/<model>/<run_name>_rounds.csv` | `round, val_auprc, val_f1, val_precision, val_recall, train_loss` (FL only) |
| W&B run | `https://wandb.ai/<entity>/hfedxgboost-paysim/runs/<id>` | Per-round loss/AUPRC curves + run summary |

The summary schema is defined once in [evaluation/results_writer.py](evaluation/results_writer.py) and every model calls into it. The canonical `<run_name>` is:

| Kind | Format |
|------|--------|
| FL | `<model>_<scheme>_alpha<alpha\|->_<oversampling>_seed<seed>` |
| Centralized | `centralized_<model>_<oversampling>_seed<seed>` |

`<model>` is the short canonical name: `ffd`, `lr`, `svm`, `gbm`, `fedxgbllr` (for FL); `lr`, `svm`, `gbm`, `xgb`, `ffd` (for centralized).

### Single experiment

Federated (argparse-style — FFD, LR, SVM, GBM):
```bash
python -m models.ffd.run \
  --scheme dirichlet --alpha 0.5 --oversampling smote \
  --random_seed 42 --use_wandb true
```

Federated (Hydra-style — FedXGBllr):
```bash
python -m hfedxgboost.main \
  dataset=paysim clients=paysim_5_clients \
  run_experiment.num_rounds=50 \
  dataset.non_iid.enabled=true dataset.non_iid.alpha=0.5 \
  dataset.oversampling.method=smote \
  random_seed=42 use_wandb=true
```

Centralized upper bound:
```bash
python -m experiments.centralized_baseline.run_ffd \
  --oversampling smote --random_seed 42 --use_wandb true
```

### Full sweep

Per-model:
```bash
bash experiments/run_ffd.sh           # 12 runs (1 seed)
bash experiments/run_fedxgbllr.sh
bash experiments/run_lr.sh
bash experiments/run_svm.sh
bash experiments/run_gbm.sh
bash experiments/run_centralized.sh   # 15 runs (5 models × 3 oversamplers)

# Multi-seed:
SEEDS="42 123 2024" bash experiments/run_ffd.sh
```

End-to-end orchestration (preflight check + every script in order):
```bash
bash experiments/run_all.sh                              # seed 42 only
SEEDS="42 123 2024" bash experiments/run_all.sh         # full 3-seed sweep
SKIP_CENTRALIZED=1 bash experiments/run_all.sh          # skip upper-bound passes
```

`run_all.sh` checks before kickoff: conda env active, `data/paysim/paysim.csv` exists, W&B logged in, and that core Python deps import. It aborts if any precondition fails.

### Collecting and reviewing results

```bash
python -m experiments.status                      # done vs pending
python -m experiments.status --pending-only       # what's left
python -m experiments.status --print-commands     # rerun commands for the pending set

python -m experiments.collect_results             # writes results/summary_table.csv
                                                  # AND prints a Markdown table to stdout
python -m experiments.collect_results --markdown-only  # skip CSV, just print MD
```

The Markdown table is paste-ready for the [Results table](#results-table) section below. The expected planned sweep lives in [experiments/registry.yaml](experiments/registry.yaml); `status.py` cross-references it against existing CSVs.

---

## RQ1 Initial Scan — Running the Three Models

### Prerequisites
- conda env activated: `conda activate fraud-fl`
- working directory: `fraud-fl-TA/`
- W&B logged in: `wandb login`
- PaySim CSV at: `data/paysim/paysim.csv`

---

### Run Configuration

| Parameter | Value |
|-----------|-------|
| Clients (K) | 5 |
| Seeds | 42, 123, 2024 |
| Oversampling | `smote` / `adasyn` / `none` (set per run via `--oversampling`) |
| LR/SVM rounds | 20 |
| GBM rounds | 10 |
| Schemes | IID, Dirichlet α=0.5, α=1.0, α=5.0 |

The `--oversampling` flag selects the per-client (federated) or global
(centralized baseline) resampler. Both `smote` and `adasyn` target a 1:1
fraud:non-fraud ratio with `k_neighbors=5` / `n_neighbors=5`. Clients with
fewer than 6 fraud samples skip oversampling and train on their raw data.

---

### FedAvg-LR

```bash
# With SMOTE (IID)
python -m models.fedavg_lr.run \
  --scheme iid --num_rounds 20 --num_clients 5 \
  --local_epochs 1 --oversampling smote \
  --random_seed 42 --use_wandb true

# With ADASYN (IID)
python -m models.fedavg_lr.run \
  --scheme iid --num_rounds 20 --num_clients 5 \
  --local_epochs 1 --oversampling adasyn \
  --random_seed 42 --use_wandb true

# Without oversampling (IID)
python -m models.fedavg_lr.run \
  --scheme iid --num_rounds 20 --num_clients 5 \
  --local_epochs 1 --oversampling none \
  --random_seed 42 --use_wandb true

# Dirichlet sweeps — swap --scheme/--alpha and --oversampling as needed
python -m models.fedavg_lr.run \
  --scheme dirichlet --alpha 0.5 --num_rounds 20 --num_clients 5 \
  --local_epochs 1 --oversampling smote \
  --random_seed 42 --use_wandb true

python -m models.fedavg_lr.run \
  --scheme dirichlet --alpha 1.0 --num_rounds 20 --num_clients 5 \
  --local_epochs 1 --oversampling smote \
  --random_seed 42 --use_wandb true

python -m models.fedavg_lr.run \
  --scheme dirichlet --alpha 5.0 --num_rounds 20 --num_clients 5 \
  --local_epochs 1 --oversampling smote \
  --random_seed 42 --use_wandb true
```

---

### FedAvg-SVM

```bash
# With SMOTE (IID)
python -m models.fedavg_svm.run \
  --scheme iid --num_rounds 20 --num_clients 5 \
  --local_epochs 1 --oversampling smote \
  --random_seed 42 --use_wandb true

# With ADASYN (IID)
python -m models.fedavg_svm.run \
  --scheme iid --num_rounds 20 --num_clients 5 \
  --local_epochs 1 --oversampling adasyn \
  --random_seed 42 --use_wandb true

# Without oversampling (IID)
python -m models.fedavg_svm.run \
  --scheme iid --num_rounds 20 --num_clients 5 \
  --local_epochs 1 --oversampling none \
  --random_seed 42 --use_wandb true

# Dirichlet sweeps — swap --scheme/--alpha and --oversampling as needed
python -m models.fedavg_svm.run \
  --scheme dirichlet --alpha 0.5 --num_rounds 20 --num_clients 5 \
  --local_epochs 1 --oversampling smote \
  --random_seed 42 --use_wandb true

python -m models.fedavg_svm.run \
  --scheme dirichlet --alpha 1.0 --num_rounds 20 --num_clients 5 \
  --local_epochs 1 --oversampling smote \
  --random_seed 42 --use_wandb true

python -m models.fedavg_svm.run \
  --scheme dirichlet --alpha 5.0 --num_rounds 20 --num_clients 5 \
  --local_epochs 1 --oversampling smote \
  --random_seed 42 --use_wandb true
```

---

### GBM Best-Model Selection

```bash
# With SMOTE (IID)
python -m models.gbm_bestmodel.run \
  --scheme iid --num_rounds 10 --num_clients 5 \
  --oversampling smote \
  --random_seed 42 --use_wandb true

# With ADASYN (IID)
python -m models.gbm_bestmodel.run \
  --scheme iid --num_rounds 10 --num_clients 5 \
  --oversampling adasyn \
  --random_seed 42 --use_wandb true

# Without oversampling (IID)
python -m models.gbm_bestmodel.run \
  --scheme iid --num_rounds 10 --num_clients 5 \
  --oversampling none \
  --random_seed 42 --use_wandb true

# Dirichlet sweeps — swap --scheme/--alpha and --oversampling as needed
python -m models.gbm_bestmodel.run \
  --scheme dirichlet --alpha 0.5 --num_rounds 10 --num_clients 5 \
  --oversampling smote \
  --random_seed 42 --use_wandb true

python -m models.gbm_bestmodel.run \
  --scheme dirichlet --alpha 1.0 --num_rounds 10 --num_clients 5 \
  --oversampling smote \
  --random_seed 42 --use_wandb true

python -m models.gbm_bestmodel.run \
  --scheme dirichlet --alpha 5.0 --num_rounds 10 --num_clients 5 \
  --oversampling smote \
  --random_seed 42 --use_wandb true
```

---

### FedXGBllr (Flower/Hydra CLI — different from the other three models)

FedXGBllr selects the oversampler via the Hydra override
`dataset.oversampling.method=<smote|adasyn|none>` (default: `smote` in
`conf/dataset/paysim.yaml`).

```bash
# With SMOTE (IID)
python -m hfedxgboost.main \
  dataset=paysim clients=paysim_5_clients \
  run_experiment.num_rounds=20 \
  dataset.oversampling.method=smote \
  use_wandb=true

# With ADASYN (IID)
python -m hfedxgboost.main \
  dataset=paysim clients=paysim_5_clients \
  run_experiment.num_rounds=20 \
  dataset.oversampling.method=adasyn \
  use_wandb=true

# Without oversampling (IID)
python -m hfedxgboost.main \
  dataset=paysim clients=paysim_5_clients \
  run_experiment.num_rounds=20 \
  dataset.oversampling.method=none \
  use_wandb=true

# Dirichlet sweeps — swap dataset.non_iid.alpha and dataset.oversampling.method
python -m hfedxgboost.main \
  dataset=paysim clients=paysim_5_clients \
  run_experiment.num_rounds=20 \
  dataset.non_iid.enabled=true dataset.non_iid.alpha=0.5 \
  dataset.oversampling.method=smote \
  use_wandb=true

python -m hfedxgboost.main \
  dataset=paysim clients=paysim_5_clients \
  run_experiment.num_rounds=20 \
  dataset.non_iid.enabled=true dataset.non_iid.alpha=1.0 \
  dataset.oversampling.method=smote \
  use_wandb=true

python -m hfedxgboost.main \
  dataset=paysim clients=paysim_5_clients \
  run_experiment.num_rounds=20 \
  dataset.non_iid.enabled=true dataset.non_iid.alpha=5.0 \
  dataset.oversampling.method=smote \
  use_wandb=true
```

---

### Centralized Baselines

The centralized baselines train a single global model on the full
training set — no client splitting, no aggregation — and act as the
theoretical upper bound for each FL model class. They use the same
`--oversampling` flag, but applied **globally** (one resampling pass on
the full `x_train`) instead of per-client.

```bash
# Logistic Regression (centralized LR upper bound)
python -m experiments.centralized_baseline.run_lr --oversampling smote \
  --random_seed 42 --use_wandb true
python -m experiments.centralized_baseline.run_lr --oversampling adasyn \
  --random_seed 42 --use_wandb true
python -m experiments.centralized_baseline.run_lr --oversampling none \
  --random_seed 42 --use_wandb true

# Linear SVM (centralized SVM upper bound)
python -m experiments.centralized_baseline.run_svm --oversampling smote \
  --random_seed 42 --use_wandb true

# HistGBM (centralized GBM upper bound)
python -m experiments.centralized_baseline.run_gbm --oversampling smote \
  --random_seed 42 --use_wandb true

# XGBoost (centralized FedXGBllr upper bound)
python -m experiments.centralized_baseline.run_xgb --oversampling smote \
  --random_seed 42 --use_wandb true

# FFD (centralized Conv1D upper bound)
python -m experiments.centralized_baseline.run_ffd --oversampling smote \
  --random_seed 42 --use_wandb true
```

---

### Results Table

Auto-populated by `python -m experiments.collect_results` — paste the output of that command in place of the rows below. The pre-filled template is for `seed=42` only; for multi-seed runs the collector emits one row per seed.

| Model | Scheme | Oversampling | Seed | test_auprc | test_f1 | test_precision | test_recall | best_round | duration (s) |
|-------|--------|--------------|------|------------|---------|----------------|-------------|------------|--------------|
| ffd | IID | smote | 42 | - | - | - | - | - | - |
| ffd | IID | adasyn | 42 | - | - | - | - | - | - |
| ffd | IID | none | 42 | - | - | - | - | - | - |
| ffd | Dirichlet α=0.5 | smote | 42 | - | - | - | - | - | - |
| ffd | Dirichlet α=0.5 | adasyn | 42 | - | - | - | - | - | - |
| ffd | Dirichlet α=0.5 | none | 42 | - | - | - | - | - | - |
| ffd | Dirichlet α=1.0 | smote | 42 | - | - | - | - | - | - |
| ffd | Dirichlet α=1.0 | adasyn | 42 | - | - | - | - | - | - |
| ffd | Dirichlet α=1.0 | none | 42 | - | - | - | - | - | - |
| ffd | Dirichlet α=5.0 | smote | 42 | - | - | - | - | - | - |
| ffd | Dirichlet α=5.0 | adasyn | 42 | - | - | - | - | - | - |
| ffd | Dirichlet α=5.0 | none | 42 | - | - | - | - | - | - |
| fedxgbllr | IID | smote | 42 | - | - | - | - | - | - |
| fedxgbllr | IID | adasyn | 42 | - | - | - | - | - | - |
| fedxgbllr | IID | none | 42 | - | - | - | - | - | - |
| fedxgbllr | Dirichlet α=0.5 | smote | 42 | - | - | - | - | - | - |
| fedxgbllr | Dirichlet α=0.5 | adasyn | 42 | - | - | - | - | - | - |
| fedxgbllr | Dirichlet α=0.5 | none | 42 | - | - | - | - | - | - |
| fedxgbllr | Dirichlet α=1.0 | smote | 42 | - | - | - | - | - | - |
| fedxgbllr | Dirichlet α=1.0 | adasyn | 42 | - | - | - | - | - | - |
| fedxgbllr | Dirichlet α=1.0 | none | 42 | - | - | - | - | - | - |
| fedxgbllr | Dirichlet α=5.0 | smote | 42 | - | - | - | - | - | - |
| fedxgbllr | Dirichlet α=5.0 | adasyn | 42 | - | - | - | - | - | - |
| fedxgbllr | Dirichlet α=5.0 | none | 42 | - | - | - | - | - | - |
| lr | IID | smote | 42 | - | - | - | - | - | - |
| lr | IID | adasyn | 42 | - | - | - | - | - | - |
| lr | IID | none | 42 | - | - | - | - | - | - |
| lr | Dirichlet α=0.5 | smote | 42 | - | - | - | - | - | - |
| lr | Dirichlet α=0.5 | adasyn | 42 | - | - | - | - | - | - |
| lr | Dirichlet α=0.5 | none | 42 | - | - | - | - | - | - |
| lr | Dirichlet α=1.0 | smote | 42 | - | - | - | - | - | - |
| lr | Dirichlet α=1.0 | adasyn | 42 | - | - | - | - | - | - |
| lr | Dirichlet α=1.0 | none | 42 | - | - | - | - | - | - |
| lr | Dirichlet α=5.0 | smote | 42 | - | - | - | - | - | - |
| lr | Dirichlet α=5.0 | adasyn | 42 | - | - | - | - | - | - |
| lr | Dirichlet α=5.0 | none | 42 | - | - | - | - | - | - |
| svm | IID | smote | 42 | - | - | - | - | - | - |
| svm | IID | adasyn | 42 | - | - | - | - | - | - |
| svm | IID | none | 42 | - | - | - | - | - | - |
| svm | Dirichlet α=0.5 | smote | 42 | - | - | - | - | - | - |
| svm | Dirichlet α=0.5 | adasyn | 42 | - | - | - | - | - | - |
| svm | Dirichlet α=0.5 | none | 42 | - | - | - | - | - | - |
| svm | Dirichlet α=1.0 | smote | 42 | - | - | - | - | - | - |
| svm | Dirichlet α=1.0 | adasyn | 42 | - | - | - | - | - | - |
| svm | Dirichlet α=1.0 | none | 42 | - | - | - | - | - | - |
| svm | Dirichlet α=5.0 | smote | 42 | - | - | - | - | - | - |
| svm | Dirichlet α=5.0 | adasyn | 42 | - | - | - | - | - | - |
| svm | Dirichlet α=5.0 | none | 42 | - | - | - | - | - | - |
| gbm | IID | smote | 42 | - | - | - | - | - | - |
| gbm | IID | adasyn | 42 | - | - | - | - | - | - |
| gbm | IID | none | 42 | - | - | - | - | - | - |
| gbm | Dirichlet α=0.5 | smote | 42 | - | - | - | - | - | - |
| gbm | Dirichlet α=0.5 | adasyn | 42 | - | - | - | - | - | - |
| gbm | Dirichlet α=0.5 | none | 42 | - | - | - | - | - | - |
| gbm | Dirichlet α=1.0 | smote | 42 | - | - | - | - | - | - |
| gbm | Dirichlet α=1.0 | adasyn | 42 | - | - | - | - | - | - |
| gbm | Dirichlet α=1.0 | none | 42 | - | - | - | - | - | - |
| gbm | Dirichlet α=5.0 | smote | 42 | - | - | - | - | - | - |
| gbm | Dirichlet α=5.0 | adasyn | 42 | - | - | - | - | - | - |
| gbm | Dirichlet α=5.0 | none | 42 | - | - | - | - | - | - |
| lr  | centralized | smote  | 42 | - | - | - | - | — | - |
| lr  | centralized | adasyn | 42 | - | - | - | - | — | - |
| lr  | centralized | none   | 42 | - | - | - | - | — | - |
| svm | centralized | smote  | 42 | - | - | - | - | — | - |
| svm | centralized | adasyn | 42 | - | - | - | - | — | - |
| svm | centralized | none   | 42 | - | - | - | - | — | - |
| gbm | centralized | smote  | 42 | - | - | - | - | — | - |
| gbm | centralized | adasyn | 42 | - | - | - | - | — | - |
| gbm | centralized | none   | 42 | - | - | - | - | — | - |
| xgb | centralized | smote  | 42 | - | - | - | - | — | - |
| xgb | centralized | adasyn | 42 | - | - | - | - | — | - |
| xgb | centralized | none   | 42 | - | - | - | - | — | - |
| ffd | centralized | smote  | 42 | - | - | - | - | — | - |
| ffd | centralized | adasyn | 42 | - | - | - | - | — | - |
| ffd | centralized | none   | 42 | - | - | - | - | — | - |

---

### Notes
- Run sequentially to avoid memory issues with PaySim (~6.3M rows)
- Each LR/SVM run ≈ X minutes, each GBM run ≈ X minutes
  (fill in after first run completes)
- W&B dashboard: https://wandb.ai — project: hfedxgboost-paysim
- All runs use seed=42 for this initial scan
  (full 3-seed runs with seeds 42, 123, 2024 come after FedXGBllr integration)

---

## Experiment tracking — Weights & Biases

Training runs log to W&B (`wandb==0.15.12`). Log in once before the first
run:

```bash
wandb login
```

Per-run metrics (AUPRC, F1, Precision, Recall, per-round client counts) and
artefacts (SHAP summary plots, confusion matrices, model checkpoints) are
streamed to the W&B project. Offline runs are still written to `./wandb/`
and can be synced later with `wandb sync`.

---

## Research metrics

| Metric        | Role       | Why                                                                 |
|---------------|------------|---------------------------------------------------------------------|
| **AUPRC**     | Primary    | Threshold-independent; only rewards correctly ranked positives.     |
| F1            | Supporting | Single-threshold harmonic mean of precision and recall.             |
| Precision     | Supporting | Cost of false alarms (analyst workload).                            |
| Recall        | Supporting | Cost of missed fraud (financial loss).                              |

Accuracy is **deliberately omitted** as a headline number: predicting
"never fraud" on PaySim achieves ~99.87% accuracy and zero recall.

---

## Academic context

This repository is part of an undergraduate thesis (**Tugas Akhir / TA**)
submitted to **Institut Teknologi Sepuluh Nopember (ITS), Surabaya**.
The contribution is empirical: a head-to-head comparison of federated
learning strategies on a heavily imbalanced fraud-detection task, with a
focus on whether the choice of aggregation (FedAvg vs. best-model
selection vs. FedXGBllr's learnable aggregation) matters more than the
choice of base learner under non-IID partitioning.

If you reference this work, please cite the upstream baselines it
builds on — FedXGBllr (Ma et al.) for the tree-ensemble FL pathway, the
PaySim dataset (Lopez-Rojas et al.) for the synthetic data, and the
Flower framework for the FL runtime.

---
