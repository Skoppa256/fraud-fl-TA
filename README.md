# Federated Learning for Financial Fraud Detection — Comparative Study

A research project benchmarking several model classes — six federated arms and
their centralized upper bounds — on the PaySim mobile-money fraud dataset.
The goal is to compare classical FL (FedAvg over linear models: LR, SVM),
tree-ensemble FL with learnable aggregation (FedXGBllr), a model-selection
variant (GBM with best-model promotion), a CNN-based fraud detector (FFD), and
a from-scratch tabular Transformer (BERT/FT-Transformer) under non-IID
partitioning and extreme class imbalance.

> **Aggregation is not uniform across arms.** Only LR and SVM use vanilla
> FedAvg. FFD and BERT use `AccuracyWeightedFedAvg` (weight by data size ×
> local AUPRC); GBM uses server-side best-model selection (no averaging); and
> FedXGBllr uses a hybrid (frozen trees concatenated, CNN via sample-count-weighted FedAvg). See the
> [Project Pipeline](#project-pipeline) aggregation table for the per-model
> truth taken from code.

The study targets **AUPRC** (Area Under the Precision–Recall Curve) as the
primary metric, because the PaySim fraud rate is ~0.13% and accuracy is
uninformative under this skew. F1, Precision, and Recall are reported as
supporting metrics. Explanations are produced per-client via SHAP.

---
## Documentation (`buku/`)

`buku/` is the undergraduate thesis (Tugas Akhir) source, written in Typst — it is
an authored deliverable, not generated output. Chapters 1–3 document this
pipeline; Chapters 4–5 are intentional stubs (no results yet).

```bash
typst compile buku/thesis.typ buku/thesis.pdf   # one-shot build
typst watch   buku/thesis.typ buku/thesis.pdf   # live rebuild
```

**Documentation updates ship with the code change that causes them** — if you
change the pipeline, models, hyperparameters, dataset handling, or dependencies,
update `buku/content.typ` in the same change and confirm the thesis still
compiles. The enforceable rule and the code→section mapping live in
[`CLAUDE.md`](CLAUDE.md).

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
│   ├── fedavg_svm/                # Linear SVM (SGD-hinge) + FedAvg
│   ├── gbm_bestmodel/             # HistGBM with server-side best-model selection
│   ├── ffd/                       # FFD — Conv1D fraud detector (Yang et al., 2019)
│   └── bert_fraud/                # FT-Transformer (Gorishniy et al., 2021) — NOT HuggingFace BERT
│
├── evaluation/
│   ├── results_writer.py          # Unified summary/per-round CSV schema
│   ├── metrics.py                 # AUPRC + val-tuned-threshold F1/Precision/Recall helpers
│   └── shap_analysis.py           # SHAP explainability driver (stub — not yet implemented)
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

### Second dataset — ULB credit-card

A second, parallel dataset can be selected at run time (see [CLI reference](#cli-reference)); PaySim remains the default, and both share the same models, aggregation, and metrics.

**Credit Card Fraud Detection** — real European card transactions over two days,
already dimensionality-reduced with PCA. 284,807 transactions across 31 columns,
of which 492 are fraud (≈0.172% positive rate).

- **Source**: Kaggle — `mlg-ulb/creditcardfraud`
  (https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Place the file at**: `data/creditcard/creditcard.csv`
- **Schema** (31 columns): `Time, V1 … V28, Amount, Class` — label is `Class`.
- **Preprocessing** (`preprocessing/creditcard.py`): no identifier drop, no one-hot,
  no balance engineering. Only `Time` and `Amount` are standardized (scaler fit on
  train only); `V1 … V28` are already PCA-standardized and left untouched. Same
  stratified 70/15/15 split and seed handling as PaySim → a **30-feature** vector.

Like PaySim, the CSV is git-ignored — download it once locally. Selecting a dataset
namespaces every output under `results/logs/<dataset>/…` and stamps a `dataset`
column into each summary CSV, so PaySim and creditcard runs never collide on disk.

---

## Models compared

1. **FedXGBllr** — Federated XGBoost with Learnable Learning Rates
   (Ma et al., 2023). Each client fits a local XGBoost ensemble **once**
   (50 trees/client → 250 for K=5) and then **freezes** it. Trees are
   **concatenated** across clients (not averaged, not retrained) into a global
   forest. A small 1-D CNN consumes the per-tree margin outputs of that frozen
   forest and learns per-tree weights ("learnable learning rates"); the CNN
   weights are aggregated via **sample-count-weighted FedAvg** (weighted by each
   client's local sample count N_k) each round while the frozen trees are
   **concatenated, not averaged**, riding along in the broadcast as fixed
   context. (Baseline; complete in
   `models/fedxgbllr/`.) See [Project Pipeline](#project-pipeline) for the exact
   mechanics.

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

4. **GBM with Best-Model Selection** — Each client trains a local
   **`HistGradientBoostingClassifier`** (histogram-based GBM, chosen for speed
   on ~4.4M rows) from scratch each round, ignoring the global model. Instead
   of averaging, the server evaluates submitted models on a held-out
   validation slice and **promotes the single best-AUPRC model** as the new
   global (Aljunaid et al., 2025). This explores whether model selection beats
   parameter averaging when local distributions diverge sharply.

5. **FFD (Conv1D Fraud Detector)** — A small 1-D CNN architecture from
   Yang et al. (2019), adapted to PaySim's 13-feature tabular input.
   Lives in `models/ffd/`; weights are aggregated via
   **`AccuracyWeightedFedAvg`** — each client's update is weighted by
   `n_c · local_AUPRC`, not by data size alone (falls back to plain FedAvg at
   cold start when all local AUPRCs are 0).

6. **BERT/FT-Transformer** *(extra — not part of the original 4-model
   proposal, but fully wired: registry, sweep script, centralized baseline)* —
   A from-scratch **Feature-Tokenizer Transformer** (Gorishniy et al., 2021),
   **not** a HuggingFace/pretrained BERT and with **no tokenizer/text**. Each
   of the 13 scalar features gets a learned linear projection to a token; a
   learnable `[CLS]` token + self-attention pool them for classification. Lives
   in `models/bert_fraud/`; aggregated via the same **`AccuracyWeightedFedAvg`**
   as FFD. "BERT" is a naming convention only (CLS-token + attention), so treat
   the label with care.

All six FL arms use **Dirichlet partitioning** (`α` ∈ {0.5, 1.0, 5.0}) to induce
non-IID client splits, and per-client oversampling — either **SMOTE** or
**ADASYN**, selected via `--oversampling {smote, adasyn, none}` — to soften
the fraud-rate imbalance before training.

---

## Project Pipeline

End-to-end, every run follows the same five stages. Stages 1–3 and 5 are
shared code; only stage 4 (local training + aggregation) differs per model.
All line citations are to source as of this writing.

```
 raw paysim.csv (~6.3M rows, 11 cols)
        │
        ▼
 (1) PREPROCESS  preprocessing/paysim.py
     drop nameOrig/nameDest/isFlaggedFraud · engineer errorBalanceOrig/Dest
     one-hot `type` (5 cols) · stratified 70/15/15 split · StandardScaler FIT ON TRAIN ONLY
        │  → 13 float32 features, int32 label; x_val/x_test kept on the server
        ▼
 (2) PARTITION  partitioning/dirichlet.py   (train split only)
     IID (shuffle + K-way split)  |  Dirichlet(α) per-class draw → K clients
     small α ⇒ a client can get ZERO fraud (intentional)
        │
        ▼
 (3) OVERSAMPLE  preprocessing/{oversampling,smote,adasyn}.py   (per client)
     SMOTE | ADASYN | none · SKIP a client when n_fraud < k_neighbors+1 (i.e. < 6)
        │
        ▼
 (4) LOCAL TRAIN + FEDERATED AGGREGATE  models/<name>/   (K clients × R rounds)
     per-model — see aggregation table below
        │
        ▼
 (5) EVALUATE on the central test set  evaluation/metrics.py
     AUPRC (primary, threshold-free) · F1/Precision/Recall at a threshold
     TUNED on the validation set (max-F1) then applied unchanged to test
     (per-client SHAP stability is planned — evaluation/shap_analysis.py is a stub)
```

**Shared 13-feature input** (`preprocessing/paysim.py:114-180`), in order:
`step, amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest,
errorBalanceOrig, errorBalanceDest, type_CASH_IN, type_CASH_OUT, type_DEBIT,
type_PAYMENT, type_TRANSFER`. Every model consumes exactly this vector — **with
one twist: FedXGBllr's CNN does not see these features directly.** Its input is
the `(N, 1, 250)` tree-margin tensor (see the FedXGBllr box below).

### Per-model local training + aggregation (from code)

| Model | Local estimator (what the client trains) | Parameter exchanged | Aggregation | Source |
|-------|------------------------------------------|---------------------|-------------|--------|
| **LR** | `LogisticRegression` (warm-start, resumes from aggregated coef) | `[coef_(1,13), intercept_(1,)]` | **Vanilla FedAvg** (sample-count weighted) | `fedavg_lr/client.py`, `strategy.py:38-49` |
| **SVM** | `SGDClassifier(loss="hinge")` — linear SVM by SGD, warm-start, `tol=None` | `[coef_(1,13), intercept_(1,)]` | **Vanilla FedAvg** | `fedavg_svm/client.py`, `strategy.py:38-49` |
| **GBM** | `HistGradientBoostingClassifier`, retrained from scratch each round | whole model, pickled → uint8 array | **Best-model selection** (argmax val AUPRC; no averaging) | `gbm_bestmodel/client.py`, `strategy.py:124-188` |
| **FedXGBllr** | 50 XGBoost trees/client (fitted once, frozen) **+** shared 1-D CNN | `[CNN weights, this client's tree ensemble]` | **Hybrid**: CNN via sample-count-weighted FedAvg (by local N_k); trees **concatenated** (kept whole, sorted by cid) | `fedxgbllr/hfedxgboost/strategy.py:45-79` |
| **FFD** | 1-D CNN on the 13 features (`(B,1,13)`), SGD, 5 local epochs | all CNN weights | **`AccuracyWeightedFedAvg`** (n_c · local AUPRC) | `ffd/strategy.py:46-106` |
| **BERT** | FT-Transformer (per-feature token + `[CLS]` + 2× attention), AdamW, 3 epochs | all Transformer weights | **`AccuracyWeightedFedAvg`** (same as FFD) | `bert_fraud/strategy.py:36-92` |

Only **LR** and **SVM** use stock FedAvg. **FFD** and **BERT** use
`AccuracyWeightedFedAvg` (data size × local AUPRC). **GBM** does not average at
all. **FedXGBllr** is hybrid.

### FedXGBllr mechanics (the most-misunderstood model)

Read `models/fedxgbllr/hfedxgboost/{models.py, utils.py:265-329, strategy.py}`.

- **Trees are fitted ONCE** on each client's train partition (50 trees/client
  → 250 total for K=5), then **FROZEN**. There is no per-round tree retraining.
- **"Tree aggregation" = concatenation/collection** of whole ensembles (each
  tagged by cid and sorted, `utils.py:311`), **not averaging** and **not
  retraining**.
- **The `(N, 1, 250)` CNN input** is built by pushing samples through the
  frozen trees with `output_margin=True` — one raw pre-sigmoid vote per tree —
  laid out in **contiguous per-client blocks of 50** (`utils.py:305-322`). The
  tabular row is *not* fed to the CNN.
- Because the trees are frozen, **this margin tensor is identical every round**;
  only the CNN weights change round to round.
- **CNN** = `Conv1d(1→64, kernel=stride=50)` so each non-overlapping window is
  exactly one client's 50-tree block → `(B,64,5)` → flatten `(B,320)` →
  `Linear(320→1)` → `Sigmoid`, trained with **BCELoss**. The conv filters *are*
  the "learnable learning rates" (`models.py` `CNN`).
- **CNN weights are aggregated by sample-count-weighted FedAvg** across clients
  each round — flwr's `Σ(wₖ·nₖ)/Σ(nₖ)` where `nₖ` is each client's local
  partition size N_k (pre-SMOTE row count, reported by `client.py` `fit`;
  `strategy.py:59-66`). This is data-proportional FedAvg (Ma et al. §3.1/§3.4):
  under unequal Dirichlet non-IID splits the per-client weights differ, while
  under equal splits it coincides with a plain uniform mean. **This weighting
  applies to the CNN aggregator only** — the frozen trees are concatenated (not
  averaged) and merely ride along in the broadcast as fixed context.
- **At test time**, test transactions are pushed through the **same
  train-fitted frozen trees** to build their `(N,1,250)` tensor, then scored by
  the single averaged CNN. This is standard train→test application, **not
  leakage** (the trees never saw test labels; they were fit only on train).

### Neural-network summary

| | FedXGBllr CNN | FFD CNN | BERT / FT-Transformer |
|---|---|---|---|
| Input to net | tree margins `(B,1,250)` | features `(B,1,13)` | features `(B,13)` → tokens `(B,14,64)` |
| Core | `Conv1d(1→64, k=50, s=50)` | 2× `Conv1d`+`MaxPool` | 2× `TransformerEncoderLayer` (Pre-LN, GELU) |
| Output | `(B,1)` **Sigmoid** | `(B,2)` logits → softmax | `(B,2)` logits → softmax |
| Loss / optimizer | BCELoss / Adam 5e-4 | CrossEntropy / SGD 1e-2 | CrossEntropy / AdamW 1e-3 |
| Aggregation | sample-count-weighted FedAvg (CNN) + tree concat | AccuracyWeightedFedAvg | AccuracyWeightedFedAvg |

### Documentation notes / open questions

Items found while re-deriving these docs from source. Per this task's scope,
**no logic was changed** — these are recorded, not fixed:

1. **README framing was stale on model count and aggregation.** The intro said
   "five model classes / four federated"; the code has **six** FL arms
   (`experiments/registry.yaml:12-18`) including a fully-wired BERT arm. FFD was
   described as "averaged via FedAvg" but uses `AccuracyWeightedFedAvg`. Both
   fixed in this doc pass.
2. **"BERT" is a misnomer.** `models/bert_fraud/` is a from-scratch
   FT-Transformer (Gorishniy 2021) with per-feature linear tokenization — no
   pretrained weights, no tokenizer, no text (`bert_fraud/model.py:1-101`).
   Renaming the module is a code change and was left alone.
3. **`num_rounds` default vs. sweep value.** LR/SVM/GBM `conf/base.yaml` default
   `num_rounds: 50`, but the effective sweep values are LR/SVM = 20, GBM = 10
   (`experiments/registry.yaml:52-58`; the shell drivers pass the override). The
   YAML default is therefore not what the sweep runs.
4. **GBM clients ignore the global model.** `set_parameters` is a no-op and each
   round retrains from scratch (`gbm_bestmodel/client.py:104-142`); "best-model
   selection" keeps the single best client model server-side with no
   cross-client knowledge transfer into local training. Intentional per the
   in-code note, but worth stating.
5. **Pickle-over-the-wire in GBM** is flagged unsafe for untrusted networks
   (arbitrary-code-execution risk) and is fine only because the whole
   simulation runs on one machine (`gbm_bestmodel/client.py:8-15`). Not for
   production as-is.
6. **`models/fedxgbllr/README.md`** is the upstream Flower baseline README
   (a9a/cod-rna/ijcnn1, `n_estimators=500`, Poetry setup) and does **not**
   describe the PaySim configuration used here (50 trees/client, 50 CNN
   iterations — `conf/clients/paysim_5_clients.yaml`). Read it as upstream
   provenance, not as this project's run config.
7. **SHAP is not implemented.** `evaluation/shap_analysis.py` is a stub, so the
   per-client SHAP-stability analysis mentioned as a research goal is not yet
   wired into any run.
8. **FedXGBllr CNN weighting is data-proportional, and only because of what the
   client reports.** flwr's `aggregate` weights each client's CNN update by the
   `num_examples` it returns. The client reports its local partition size N_k
   (pre-SMOTE row count; `client.py` `fit`), so the CNN aggregation is genuine
   sample-count-weighted FedAvg per Ma et al. §3.1/§3.4. Note the sensitivity:
   an earlier revision reported `num_examples = num_iterations × batch_size`
   (identical for every client), which silently collapsed the same
   `Σ(wₖ·nₖ)/Σ(nₖ)` into a uniform mean — the two only diverge under unequal
   (Dirichlet non-IID) splits. This concerns the CNN aggregator only; the trees
   are concatenated, never averaged.

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
| BERT rounds | 50 |
| FedXGBllr rounds | 50 |
| LR / SVM rounds | 20 |
| GBM rounds | 10 |

`--oversampling` selects the per-client (federated) or global (centralized) resampler. `smote` and `adasyn` both target a 1:1 fraud:non-fraud ratio with `k_neighbors=5` / `n_neighbors=5`. Clients with fewer than 6 fraud samples skip oversampling and train on their raw data.

---

### CLI reference

CLI flags override the corresponding key in the model's `conf/base.yaml` (Hydra equivalent: `conf/dataset/paysim.yaml`); omit a flag to keep the YAML default.

**Choosing the dataset.** Every entry point takes a dataset selector that defaults to `paysim`, so existing commands are unchanged. Pass `--dataset creditcard` (argparse) or `dataset=creditcard clients=creditcard_5_clients` (Hydra) to run the ULB credit-card dataset instead. Feature count (13 for PaySim, 30 for creditcard) is read from the data at run time — no other flag changes.

#### Federated — argparse (FFD / BERT / LR / SVM / GBM)

```
usage: python -m models.<model>.run [-h]
       [--dataset {paysim,creditcard}]
       [--scheme {iid,dirichlet}] [--alpha ALPHA]
       [--num_rounds N] [--num_clients K] [--local_epochs E]
       [--oversampling {smote,adasyn,none}]
       [--sampling_strategy {auto,FLOAT}]
       [--random_seed SEED] [--use_wandb {true,false}]
       [--wandb_project NAME]
       [--batch_size B] [--lr LR]                       # ffd / bert_fraud
       [--weight_decay WD]                              # bert_fraud only
       [--max_iter N] [--max_depth D] [--learning_rate LR]   # gbm_bestmodel only
```

`<model>` ∈ {`ffd`, `bert_fraud`, `fedavg_lr`, `fedavg_svm`, `gbm_bestmodel`}.
`bert_fraud` defaults: `local_epochs=3`, `batch_size=64`, `lr=0.001`,
`weight_decay=1e-4` (`models/bert_fraud/conf/base.yaml`).

| Flag | Type | Choices / range | YAML default | Notes |
|------|------|-----------------|--------------|-------|
| `--dataset` | str | `paysim`, `creditcard` | `paysim` | Dataset to load. Sets the feature count (13 / 30) and the `results/logs/<dataset>/…` namespace. |
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

Examples (substitute `ffd` → `bert_fraud` / `fedavg_lr` / `fedavg_svm` / `gbm_bestmodel`):

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

# Same run on the ULB credit-card dataset (30 features, auto-detected)
python -m models.ffd.run --dataset creditcard --scheme iid --oversampling smote \
    --random_seed 42 --use_wandb true
```

#### Federated — Hydra (FedXGBllr)

```
usage: python -m hfedxgboost.main [HYDRA_OVERRIDE [HYDRA_OVERRIDE ...]]
```

Hydra overrides are `key=value` pairs chained on the command line.

| Override | Choices / range | Default | Notes |
|----------|-----------------|---------|-------|
| `dataset` | `paysim`, `creditcard` | `paysim` | Selects `conf/dataset/<dataset>.yaml`. |
| `clients` | `paysim_5_clients`, `creditcard_5_clients`, `*_2_clients` | `paysim_5_clients` | Client-count config; match the dataset (e.g. `creditcard_5_clients`). |
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

# Credit-card dataset (swap both dataset= and clients= together)
python -m hfedxgboost.main \
    dataset=creditcard clients=creditcard_5_clients \
    run_experiment.num_rounds=50 \
    dataset.oversampling.method=smote \
    random_seed=42 use_wandb=true
```

#### Centralized upper bounds (LR / SVM / GBM / XGB / FFD)

```
usage: python -m experiments.centralized_baseline.run_<model> [-h]
       [--dataset {paysim,creditcard}]
       [--oversampling {smote,adasyn,none}]
       [--sampling_strategy {auto,FLOAT}]
       [--random_seed SEED] [--use_wandb {true,false}]
       [--wandb_project NAME]
       [--num_epochs N] [--batch_size B] [--lr LR]      # run_ffd only
```

`<model>` ∈ {`lr`, `svm`, `gbm`, `xgb`, `ffd`, `bert_fraud`}. Oversampling is applied **globally** to the full `x_train` (one resampling pass) rather than per-client.

| Flag | Type | Choices / range | Default | Notes |
|------|------|-----------------|---------|-------|
| `--dataset` | str | `paysim`, `creditcard` | `paysim` | Dataset to load; also sets the `results/logs/<dataset>/centralized/…` namespace. |
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
python -m experiments.centralized_baseline.run_lr  --oversampling smote --sampling_strategy 0.01 --random_seed 42 --use_wandb true
python -m experiments.centralized_baseline.run_svm --oversampling smote --sampling_strategy 0.01 --random_seed 42 --use_wandb true
python -m experiments.centralized_baseline.run_gbm --oversampling smote --sampling_strategy 0.01 --random_seed 42 --use_wandb true
python -m experiments.centralized_baseline.run_xgb --oversampling smote  --sampling_strategy 0.01 --random_seed 42 --use_wandb true
python -m experiments.centralized_baseline.run_ffd --oversampling smote  --num_epochs 30 --random_seed 42 --use_wandb true

# Credit-card dataset
python -m experiments.centralized_baseline.run_lr  --dataset creditcard --oversampling smote --random_seed 42 --use_wandb true
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
usage: [SEEDS="<seed> [seed ...]"] [DATASET={paysim,creditcard}] bash experiments/run_<model>.sh
usage: [SEEDS="..."] [DATASET=...] [SKIP_CENTRALIZED={0,1}] bash experiments/run_all.sh
```

Env-var overrides:

| Variable | Default | Applies to | Notes |
|----------|---------|------------|-------|
| `SEEDS` | `42` | all `run_*.sh` | Space-separated list, e.g. `"42 123 2024"`. |
| `DATASET` | `paysim` | all `run_*.sh` | Dataset to sweep; passed through as `--dataset` / `dataset=` and namespaces logs + CSVs under `results/logs/<DATASET>/`. |
| `SKIP_CENTRALIZED` | `0` | `run_all.sh` only | `1` skips the centralized upper-bound passes. |

Per-model sweeps — each covers IID + Dirichlet ∈ {0.5, 1.0, 5.0} × {SMOTE, ADASYN, none}:

```bash
bash experiments/run_ffd.sh           # 12 runs / seed
bash experiments/run_bert_fraud.sh    # 12 runs / seed
bash experiments/run_fedxgbllr.sh     # 12 runs / seed
bash experiments/run_lr.sh            # 12 runs / seed
bash experiments/run_svm.sh           # 12 runs / seed
bash experiments/run_gbm.sh           # 12 runs / seed
bash experiments/run_centralized.sh   # 18 runs / seed (6 models × 3 oversamplers)

# Multi-seed
SEEDS="42 123 2024" bash experiments/run_ffd.sh

# Sweep the credit-card dataset instead of PaySim
DATASET=creditcard bash experiments/run_ffd.sh
```

End-to-end orchestration (preflight check + every script in order):

```bash
bash experiments/run_all.sh                          # seed 42, all stages
SEEDS="42 123 2024" bash experiments/run_all.sh      # full 3-seed sweep
SKIP_CENTRALIZED=1 bash experiments/run_all.sh       # skip upper-bound passes
DATASET=creditcard bash experiments/run_all.sh       # run the full sweep on creditcard
```

`run_all.sh` aborts on first preflight failure: conda env active, the selected dataset's CSV (`data/${DATASET}/${DATASET}.csv`, default `data/paysim/paysim.csv`) exists, W&B logged in, and core Python deps import.

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