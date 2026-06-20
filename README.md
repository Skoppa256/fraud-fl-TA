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
│   └── dirichlet.py               # Dirichlet non-IID client partition
│
├── models/
│   ├── fedxgbllr/                 # FedXGBllr (Flower hfedxgboost — Hydra CLI)
│   ├── fedavg_lr/                 # Logistic Regression + FedAvg
│   ├── fedavg_svm/                # Linear SVM + FedAvg
│   ├── gbm_bestmodel/             # HistGBM with server-side best-model selection
│   └── ffd/                       # FFD — Conv1D fraud detector (Yang et al., 2019)
│
├── evaluation/
│   ├── results_writer.py          # Unified summary/per-round CSV schema
│   ├── metrics.py                 # AUPRC/F1/Precision/Recall helpers (TBD)
│   └── shap_analysis.py           # SHAP explainability driver (TBD)
│
├── experiments/
│   ├── centralized_baseline/      # Upper-bound non-FL runs (LR/SVM/GBM/XGB/FFD)
│   ├── _run_helpers.sh            # Shared run_one helper sourced by run_*.sh
│   ├── run_<model>.sh             # Per-model sweep drivers (ffd/fedxgbllr/lr/svm/gbm/centralized)
│   ├── run_all.sh                 # End-to-end orchestration + preflight checks
│   ├── registry.yaml              # Planned sweep declaration (model × scheme × oversampling × seed)
│   ├── status.py                  # Cross-checks registry vs results/logs/
│   └── collect_results.py         # Aggregates per-run CSVs → master CSV + Markdown table
│
├── results/                       # Auto-generated CSVs, plots, model artifacts
│   ├── logs/<model>/              # Per-run .log, .csv, _rounds.csv (written by every run)
│   └── visualizations/            # PCA plots per partition × oversampler
├── notebooks/                     # kaggle_setup.ipynb, pca_visualization.py, tsne_visualization.py
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

3. **Linear SVM + FedAvg** — Linear support-vector classifier (hinge loss)
   trained by SGD (`SGDClassifier(loss="hinge")`), weights aggregated via
   **FedAvg**. SGD rather than `LinearSVC` is deliberate: a batch SVM solver
   re-solves to its local optimum every round and ignores the aggregated
   weights, leaving the federated metric curves flat after round 1. SGD
   takes a few local steps *from* the aggregated weights each round, so
   FedAvg makes incremental progress. Captures a different decision boundary
   from LR while staying within the FedAvg protocol so the comparison
   isolates *model class* from *aggregation strategy*. (The centralized SVM
   upper bound still uses batch `LinearSVC` — it fits once and never needs
   aggregating.)

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

### Prerequisites

- conda env active: `conda activate fraud-fl`
- working directory: repo root (`fraud-fl-TA/`)
- W&B logged in: `wandb login`
- PaySim CSV at: `data/paysim/paysim.csv`

### Output artefacts

Every run — federated or centralized — emits the same structured output:

| Artifact | Path | Schema |
|----------|------|--------|
| Stdout/stderr log | `results/logs/<model>/<run_name>.log` | Tee'd by the shell driver |
| Summary CSV | `results/logs/<model>/<run_name>.csv` | `model, scheme, alpha, oversampling, random_seed, num_rounds, num_clients, best_round, best_val_*, test_*, timestamp, duration_seconds, run_name` |
| Per-round CSV | `results/logs/<model>/<run_name>_rounds.csv` | `round, val_auprc, val_f1, val_precision, val_recall, train_loss` (FL only) |
| W&B run | `https://wandb.ai/<entity>/fraud-fl-TA/runs/<id>` | Per-round loss/AUPRC curves + run summary |

The summary schema is defined once in [evaluation/results_writer.py](evaluation/results_writer.py) and every model calls into it. The canonical `<run_name>` is:

| Kind | Format |
|------|--------|
| FL | `<model>_<scheme>_alpha<alpha\|->_<oversampling>_seed<seed>` |
| Centralized | `centralized_<model>_<oversampling>_seed<seed>` |

`<model>` is the short canonical name: `ffd`, `lr`, `svm`, `gbm`, `fedxgbllr` (for FL); `lr`, `svm`, `gbm`, `xgb`, `ffd` (for centralized).

### Run configuration (RQ1 initial scan)

| Parameter | Value |
|-----------|-------|
| Clients (K) | 5 |
| Seeds | 42, 123, 2024 |
| Schemes | IID, Dirichlet α ∈ {0.5, 1.0, 5.0} |
| Oversampling | `smote` / `adasyn` / `none` |
| FFD rounds | 50 |
| FedXGBllr rounds | 50 |
| LR / SVM rounds | 20 |
| GBM rounds | 10 |

`--oversampling` selects the per-client (federated) or global (centralized) resampler. `smote` and `adasyn` both target a 1:1 fraud:non-fraud ratio with `k_neighbors=5` / `n_neighbors=5`. Clients with fewer than 6 fraud samples skip oversampling and train on their raw data.

---

### CLI reference

CLI flags override the corresponding key in the model's `conf/base.yaml` (Hydra equivalent: `conf/dataset/paysim.yaml`); omit a flag to keep the YAML default.

#### Federated — argparse (FFD / LR / SVM / GBM)

```
usage: python -m models.<model>.run [-h]
       [--scheme {iid,dirichlet}] [--alpha ALPHA]
       [--num_rounds N] [--num_clients K] [--local_epochs E]
       [--oversampling {smote,adasyn,none}]
       [--sampling_strategy {auto,FLOAT}]
       [--random_seed SEED] [--use_wandb {true,false}]
       [--wandb_project NAME]
       [--batch_size B] [--lr LR]                       # ffd only
       [--max_iter N] [--max_depth D] [--learning_rate LR]   # gbm_bestmodel only
```

`<model>` ∈ {`ffd`, `fedavg_lr`, `fedavg_svm`, `gbm_bestmodel`}.

| Flag | Type | Choices / range | YAML default | Notes |
|------|------|-----------------|--------------|-------|
| `--scheme` | str | `iid`, `dirichlet` | `iid` | Partition strategy. |
| `--alpha` | float | > 0 | `null` | Dirichlet concentration; required when `--scheme dirichlet`. |
| `--num_rounds` | int | ≥ 1 | `50` | FL communication rounds. Sweep scripts pass `20` (LR/SVM) and `10` (GBM). |
| `--num_clients` | int | ≥ 1 | `5` | Total clients (K). |
| `--local_epochs` | int | ≥ 1 | `1` (LR/SVM/GBM), `5` (FFD) | Local passes per round. |
| `--oversampling` | str | `smote`, `adasyn`, `none` | `smote` | Per-client resampler. |
| `--sampling_strategy` | str / float | `auto` or float ∈ (0, 1] | `auto` | Passed to imblearn's `sampling_strategy`. `auto` = 1:1 fraud:non-fraud. Float = post-resample minority/majority ratio (e.g. `0.01` → 1:100). |
| `--random_seed` | int | — | `42` | Used by partitioning, samplers, model init. |
| `--use_wandb` | bool | `true` / `false` | `false` | Stream metrics to W&B. |
| `--wandb_project` | str | — | `fraud-fl-TA` | W&B project name. |
| `--batch_size` *(FFD)* | int | ≥ 1 | `80` | Mini-batch size for Conv1D. |
| `--lr` *(FFD)* | float | > 0 | `0.01` | Adam learning rate. |
| `--max_iter` *(GBM)* | int | ≥ 1 | `100` | HistGBM boosting iters. |
| `--max_depth` *(GBM)* | int | ≥ 1 | `6` | HistGBM tree depth. |
| `--learning_rate` *(GBM)* | float | > 0 | `0.1` | HistGBM shrinkage. |

Examples (substitute `ffd` → `fedavg_lr` / `fedavg_svm` / `gbm_bestmodel`):

```bash
# IID + SMOTE
python -m models.ffd.run --scheme iid --oversampling smote \
    --random_seed 42 --use_wandb true

# Dirichlet α=0.5 + ADASYN
python -m models.ffd.run --scheme dirichlet --alpha 0.5 --oversampling adasyn \
    --random_seed 42 --use_wandb true

# No oversampling, multi-seed (one invocation per seed)
python -m models.ffd.run --scheme iid --oversampling none \
    --random_seed 2024 --use_wandb true
```

#### Federated — Hydra (FedXGBllr)

```
usage: python -m hfedxgboost.main [HYDRA_OVERRIDE [HYDRA_OVERRIDE ...]]
```

Hydra overrides are `key=value` pairs chained on the command line.

| Override | Choices / range | Default | Notes |
|----------|-----------------|---------|-------|
| `dataset` | `paysim` | — | Selects `conf/dataset/paysim.yaml`. |
| `clients` | `paysim_5_clients` | — | Client count config. |
| `run_experiment.num_rounds` | int ≥ 1 | `50` | FL rounds. |
| `dataset.non_iid.enabled` | `true` / `false` | `false` | Switch to Dirichlet partitioning. |
| `dataset.non_iid.alpha` | float > 0 | `1.0` | Dirichlet α. |
| `dataset.oversampling.method` | `smote` / `adasyn` / `none` | `smote` | Per-client resampler. |
| `dataset.oversampling.sampling_strategy` | `auto` or float | `auto` | imblearn `sampling_strategy`. `auto` = 1:1; float = post-resample minority/majority ratio (e.g. `0.01` → 1:100). |
| `random_seed` | int | `42` | Affects partition + sampler + model init. |
| `use_wandb` | `true` / `false` | `false` | Stream metrics to W&B. |

Examples:

```bash
# IID + SMOTE
python -m hfedxgboost.main \
    dataset=paysim clients=paysim_5_clients \
    run_experiment.num_rounds=50 \
    dataset.oversampling.method=smote \
    random_seed=42 use_wandb=true

# Dirichlet α=0.5 + ADASYN
python -m hfedxgboost.main \
    dataset=paysim clients=paysim_5_clients \
    run_experiment.num_rounds=50 \
    dataset.non_iid.enabled=true dataset.non_iid.alpha=0.5 \
    dataset.oversampling.method=adasyn \
    random_seed=42 use_wandb=true
```

#### Centralized upper bounds (LR / SVM / GBM / XGB / FFD)

```
usage: python -m experiments.centralized_baseline.run_<model> [-h]
       [--oversampling {smote,adasyn,none}]
       [--sampling_strategy {auto,FLOAT}]
       [--random_seed SEED] [--use_wandb {true,false}]
       [--wandb_project NAME]
       [--num_epochs N] [--batch_size B] [--lr LR]      # run_ffd only
```

`<model>` ∈ {`lr`, `svm`, `gbm`, `xgb`, `ffd`}. Oversampling is applied **globally** to the full `x_train` (one resampling pass) rather than per-client.

| Flag | Type | Choices / range | Default | Notes |
|------|------|-----------------|---------|-------|
| `--oversampling` | str | `smote`, `adasyn`, `none` | `smote` | Global resampler. |
| `--sampling_strategy` | str / float | `auto` or float ∈ (0, 1] | `auto` | Passed to imblearn's `sampling_strategy`. `auto` = 1:1 fraud:non-fraud. Float = post-resample minority/majority ratio (e.g. `0.01` → 1:100). |
| `--random_seed` | int | — | `42` | |
| `--use_wandb` | bool | `true` / `false` | `false` | |
| `--wandb_project` | str | — | `fraud-fl-TA` | |
| `--num_epochs` *(FFD)* | int | ≥ 1 | `20` | Centralized training epochs. |
| `--batch_size` *(FFD)* | int | ≥ 1 | `80` | |
| `--lr` *(FFD)* | float | > 0 | `0.01` | |

Examples:

```bash
python -m experiments.centralized_baseline.run_lr  --oversampling smote  --random_seed 42 --use_wandb true
python -m experiments.centralized_baseline.run_svm --oversampling adasyn --random_seed 42 --use_wandb true
python -m experiments.centralized_baseline.run_gbm --oversampling none   --random_seed 42 --use_wandb true
python -m experiments.centralized_baseline.run_xgb --oversampling smote  --random_seed 42 --use_wandb true
python -m experiments.centralized_baseline.run_ffd --oversampling smote  --num_epochs 30 --random_seed 42 --use_wandb true
```

#### Picking a non-1:1 oversampling ratio

`--sampling_strategy` (argparse) / `dataset.oversampling.sampling_strategy` (Hydra) is passed straight through to `imblearn.over_sampling.SMOTE(sampling_strategy=...)`. Default `auto` resamples the minority class to match the majority count (1:1). Pass a float in `(0, 1]` to set the post-resample minority/majority ratio:

| Value | Meaning | Resulting fraud share |
|-------|---------|-----------------------|
| `auto` *(default)* | 1:1 fraud:non-fraud | 50.00% |
| `0.5` | 1:2 | 33.33% |
| `0.1` | 1:10 | 9.09% |
| `0.05` | 1:20 | 4.76% |
| `0.01` | 1:100 | 0.99% |

Examples — centralized FFD with SMOTE 1:100, FL FFD with SMOTE 1:100, FedXGBllr with SMOTE 1:20:

```bash
python -m experiments.centralized_baseline.run_ffd \
    --oversampling smote --sampling_strategy 0.01 \
    --random_seed 42 --use_wandb true

python -m models.ffd.run --scheme iid \
    --oversampling smote --sampling_strategy 0.01 \
    --random_seed 42 --use_wandb true

python -m hfedxgboost.main \
    dataset=paysim clients=paysim_5_clients \
    run_experiment.num_rounds=50 \
    dataset.oversampling.method=smote \
    dataset.oversampling.sampling_strategy=0.05 \
    random_seed=42 use_wandb=true
```

The startup log echoes the resolved value and the post-resample fraud ratio so you can sanity-check.

---

### Sweep drivers (shell scripts)

```
usage: [SEEDS="<seed> [seed ...]"] bash experiments/run_<model>.sh
usage: [SEEDS="..."] [SKIP_CENTRALIZED={0,1}] bash experiments/run_all.sh
```

Env-var overrides:

| Variable | Default | Applies to | Notes |
|----------|---------|------------|-------|
| `SEEDS` | `42` | all `run_*.sh` | Space-separated list, e.g. `"42 123 2024"`. |
| `SKIP_CENTRALIZED` | `0` | `run_all.sh` only | `1` skips the centralized upper-bound passes. |

Per-model sweeps — each covers IID + Dirichlet ∈ {0.5, 1.0, 5.0} × {SMOTE, ADASYN, none}:

```bash
bash experiments/run_ffd.sh           # 12 runs / seed
bash experiments/run_fedxgbllr.sh     # 12 runs / seed
bash experiments/run_lr.sh            # 12 runs / seed
bash experiments/run_svm.sh           # 12 runs / seed
bash experiments/run_gbm.sh           # 12 runs / seed
bash experiments/run_centralized.sh   # 15 runs / seed (5 models × 3 oversamplers)

# Multi-seed
SEEDS="42 123 2024" bash experiments/run_ffd.sh
```

End-to-end orchestration (preflight check + every script in order):

```bash
bash experiments/run_all.sh                          # seed 42, all stages
SEEDS="42 123 2024" bash experiments/run_all.sh      # full 3-seed sweep
SKIP_CENTRALIZED=1 bash experiments/run_all.sh       # skip upper-bound passes
```

`run_all.sh` aborts on first preflight failure: conda env active, `data/paysim/paysim.csv` exists, W&B logged in, and core Python deps import.

---

### Collecting and reviewing results

```
usage: python -m experiments.status            [--registry PATH] [--logs-root PATH]
                                               [--pending-only] [--print-commands]
usage: python -m experiments.collect_results   [--logs-root PATH] [--out PATH]
                                               [--markdown-only]
```

| Flag | Belongs to | Default | Notes |
|------|------------|---------|-------|
| `--registry` | `status` | `experiments/registry.yaml` | Planned sweep definition. |
| `--logs-root` | both | `results/logs` | Directory walked for per-run CSVs. |
| `--pending-only` | `status` | off | Only print runs still missing. |
| `--print-commands` | `status` | off | Emit re-run commands for the pending set. |
| `--out` | `collect_results` | `results/summary_table.csv` | Master CSV destination. |
| `--markdown-only` | `collect_results` | off | Skip CSV write; only print Markdown. |

```bash
python -m experiments.status                       # done vs pending
python -m experiments.status --pending-only        # just the gaps
python -m experiments.status --print-commands      # re-run commands for pending

python -m experiments.collect_results              # writes results/summary_table.csv
                                                   # AND prints a paste-ready Markdown table
python -m experiments.collect_results --markdown-only
```

The Markdown table is paste-ready for the [Results table](#results-table) section below. The planned sweep lives in [experiments/registry.yaml](experiments/registry.yaml); `status.py` cross-references it against existing CSVs.

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
- Run sequentially to avoid memory issues with PaySim (~6.3M rows).
- Per-run wall-clock times — fill in after first sweep completes.
- W&B dashboard: https://wandb.ai — project: `fraud-fl-TA`.
- The single-seed (`SEEDS=42`) sweep is the initial scan; the full study runs
  seeds `42 123 2024` via `SEEDS="42 123 2024" bash experiments/run_all.sh`.

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
