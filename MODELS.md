# Federated Learning Fraud-Detection Codebase — Model-by-Model Technical Report

**Scope & method.** Everything below is taken from source in `fraud-fl-TA/`. The
shared pipeline and every model's `model.py`/`client.py`/`server.py`/`strategy.py`/`conf/base.yaml`
were read directly (the two neural nets and the whole FedXGBllr stack line-by-line).
The binary `.docx` proposal files were **not** opened, so the "proposal" column
reflects the model list stated in the task prompt and the repo `README.md`, not the
Word documents themselves — that caveat matters for the discrepancy section.

---

## 1. Codebase inventory

| FL "model" | Implementing files | Status vs. proposal |
|---|---|---|
| **Logistic Regression + FedAvg** | `models/fedavg_lr/` — `client.py`, `server.py`, `strategy.py`, `run.py`, `conf/base.yaml` | ✅ Implemented (in proposal) |
| **Linear SVM + FedAvg** | `models/fedavg_svm/` — same layout | ✅ Implemented (in proposal) |
| **GBM + best-model selection** | `models/gbm_bestmodel/` — same layout | ✅ Implemented (in proposal) |
| **FedXGBllr** (XGBoost trees + 1-D CNN) | `models/fedxgbllr/hfedxgboost/` — `models.py`, `client.py`, `server.py`, `strategy.py`, `utils.py`, `dataset.py`, `dataset_preparation.py`, `conf/**` (Hydra) | ✅ Implemented (in proposal) |
| **FFD — 1-D Conv fraud detector** (Yang et al. 2019) | `models/ffd/` — `model.py`, `client.py`, `server.py`, `strategy.py`, `run.py`, `conf/base.yaml` | ⚠️ Extra vs. the 4 proposal models; listed in README |
| **"BERT" fraud model** (actually an FT-Transformer) | `models/bert_fraud/` — `model.py`, `client.py`, `server.py`, `strategy.py`, `run.py`, `conf/base.yaml` + centralized `run_bert_fraud.py` | 🔺 **Extra** — not in README's 5-model list, not in proposal |

**Shared infrastructure** (used by all the argparse models — LR/SVM/GBM/FFD/BERT):
- `preprocessing/paysim.py` — the single PaySim cleaning/feature pipeline (13 features).
- `partitioning/dirichlet.py` — IID + Dirichlet non-IID client splits.
- `preprocessing/smote.py`, `adasyn.py`, `oversampling.py` — per-client resampling.
- `evaluation/metrics.py` — AUPRC + val-tuned-threshold F1/precision/recall.

FedXGBllr has its own Hydra data stack but calls the shared `get_partition` for
Dirichlet and the shared SMOTE/ADASYN for oversampling (see §5).

**FL framework wiring.** All models are Flower (`flwr` 1.5.0). LR/SVM/GBM/FFD/BERT
are `NumPyClient`/`Client` subclasses launched from a per-model `run.py` with
argparse; FedXGBllr is a Hydra CLI (`python -m hfedxgboost.main`) with a custom
`fl.server.Server` subclass (`FlServer`) and custom strategy (`FedXgbNnAvg`).

---

## 2. Shared tabular → feature-vector transformation (applies to every model)

All models consume the **same 13-dimensional feature vector** produced by
`preprocessing/paysim.py:53-111`.

Steps, in order:
1. **Drop 3 columns** — `nameOrig`, `nameDest`, `isFlaggedFraud` (`paysim.py:24`, `:114-116`).
2. **Engineer 2 balance-error features** (`paysim.py:119-128`):
   ```python
   df["errorBalanceOrig"] = df["newbalanceOrig"] - df["oldbalanceOrg"] + df["amount"]
   df["errorBalanceDest"] = df["oldbalanceDest"] + df["amount"] - df["newbalanceDest"]
   ```
3. **One-hot the transaction `type`** into 5 columns (`CASH_IN, CASH_OUT, DEBIT, PAYMENT, TRANSFER`),
   `type` dropped (`paysim.py:131-139`, categories at `:25-31`).
4. **Stratified 70/15/15 train/val/test** split (`paysim.py:150-167`).
5. **StandardScaler fit on train only**, applied to val/test (`paysim.py:170-180`).

**Final feature vector: 13 float32 columns, in this order** (8 numeric + 5 one-hot):
`step, amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest,
errorBalanceOrig, errorBalanceDest, type_CASH_IN, type_CASH_OUT, type_DEBIT,
type_PAYMENT, type_TRANSFER`. Target `isFraud` is int32. The `N_FEATURES = 13`
constant is hard-coded in the LR/SVM/GBM clients and confirmed dynamically by the two NNs.

**Partitioning.** `get_partition(scheme, alpha, num_clients, random_state)` — IID does
a seeded shuffle + `np.array_split` (`dirichlet.py:24-53`); Dirichlet draws a `Dir(alpha)`
proportion vector *per class* and routes class indices accordingly (`dirichlet.py:56-113`).
Small α can leave a client with **zero fraud** — intentional (`dirichlet.py:84-87`).

**Per-client oversampling.** SMOTE skips a client when `n_fraud < k_neighbors+1` (i.e. < 6)
or when a float target ratio is already met, else
`SMOTE(sampling_strategy, k_neighbors=5, random_state=base_seed+client_id)`
(`smote.py:61-125`). Dispatch across `{smote, adasyn, none}` in `oversampling.py:19-72`.

---

## 3. Logistic Regression + FedAvg

**Algorithm.** Single sklearn `LogisticRegression` per client (not composite).
```python
# models/fedavg_lr/client.py:33-40
LogisticRegression(C=1.0, max_iter=1000, warm_start=True, random_state=seed)
```
Default `lbfgs` solver, `l2` penalty (log loss).

**Local training.** Server weights are pushed into the estimator, then `.fit()` is called
`local_epochs` times; `warm_start=True` means each round resumes from the aggregated
coefficients (`client.py:79-84` `set_parameters`, `:86-113` `fit`). Clients with one class
in their partition skip fitting. Local AUPRC is computed from `predict_proba(...)[:,1]`
and reported as a metric.

**Parameter exchange.** `[coef_ (1,13), intercept_ (1,)]` as float32 NDArrays (`client.py:73-77`).

**Aggregation — vanilla FedAvg** (sample-count weighted average, built into Flower):
```python
# models/fedavg_lr/strategy.py:37-49
fl.server.strategy.FedAvg(fraction_fit=1.0, fraction_evaluate=0.0,
    min_fit_clients=num_clients, ..., evaluate_fn=server_eval_fn,
    initial_parameters=<zeros>, fit_metrics_aggregation_fn=weighted_average, ...)
```
`weighted_average` (`strategy.py:11-24`) only aggregates *reported metrics*; the parameter
averaging is Flower's stock FedAvg.

**Hyperparameters** (`conf/base.yaml`): `num_clients=5, num_rounds=50, local_epochs=1,
seed=42, C=1.0, max_iter=1000, warm_start=true, oversampling=smote,
sampling_strategy=auto, partition=iid`.

---

## 4. Linear SVM + FedAvg

**Algorithm.** Single `SGDClassifier(loss="hinge")` per client — a linear SVM trained by
SGD (chosen over `LinearSVC` precisely so FedAvg can make incremental progress).
```python
# models/fedavg_svm/client.py:62-71
SGDClassifier(loss="hinge", alpha=1e-4, max_iter=5, tol=None,
    learning_rate="optimal", eta0=0.0, random_state=seed, warm_start=True)
```

**Local training.** `.fit()` (not `partial_fit`) called `local_epochs` times with
`warm_start=True` so SGD's `coef_init` is seeded from the aggregated weights; `tol=None`
forces a fixed number of passes rather than re-solving to the local optimum
(`client.py:107-140`). Scores come from `decision_function` (hinge → no `predict_proba`);
metrics later use a val-tuned threshold, not 0.

**Parameter exchange / aggregation.** Identical shape/protocol to LR:
`[coef_ (1,13), intercept_ (1,)]` float32 (`client.py:94-105`); vanilla Flower **FedAvg**
(`strategy.py:27-49`).

**Hyperparameters** (`conf/base.yaml`): `num_clients=5, num_rounds=50, local_epochs=1,
seed=42, alpha=1e-4, max_iter=5, learning_rate=optimal, eta0=0.0, oversampling=smote,
partition=iid`.

---

## 5. FedXGBllr — composite: XGBoost tree ensembles + 1-D CNN aggregator (CRITICAL)

This is the one genuinely composite FL model. It is Ma et al. 2023, "Federated XGBoost
with learnable learning rates" (`strategy.py:20-24`).

### 5.1 The two sub-components and how they connect

**Sub-component A — per-client XGBoost ensembles.** Each client fits a local
`XGBClassifier` on its partition:
```python
# models/fedxgbllr/hfedxgboost/models.py:44-52  (paysim → else-branch)
tree = instantiate(config.XGBoost)   # for paysim (dataset_name != "all")
tree.fit(x_train, y_train)
```
`config.XGBoost` resolves (`conf/base.yaml:17-30`) to
`XGBClassifier(objective="binary:logistic", learning_rate=0.1, max_depth=6,
n_estimators=50, subsample=0.8, alpha=5, gamma=5, num_parallel_tree=1,
min_child_weight=1)`. So **each client → 50 trees**; with 5 clients the server holds
**250 trees** total.

**Sub-component B — 1-D CNN aggregator** (the "learnable learning rates"). Defined in
`models.py:55-105` (details in §5.4).

**The connection — tree outputs become the CNN's input** (`utils.py:265-329`). This is the
key transform: the tabular row is **not** fed to the CNN. Instead, every tree emits a raw
margin prediction per sample, and those per-tree margins are stacked:
```python
# utils.py:305-322
preds_from_all_trees_from_all_clients = np.zeros(
    (x_train.shape[0], client_num * n_estimators_client), dtype=np.float32)   # (N, 250)
...
for i, _ in enumerate(temp_trees):            # per client tree-ensemble
    for j in range(n_estimators_client):      # per tree in that ensemble
        preds[:, i*n_estimators_client + j] = single_tree_prediction(temp_trees[i], j, x_train)
# single_tree_prediction → tree.predict(dataset, iteration_range=(j,j+1), output_margin=True)  (utils.py:260-262)
preds = torch.from_numpy(np.expand_dims(preds, axis=1))   # (N, 1, 250)
```
So the vector the Conv1d consumes has **length = client_num × n_estimators_client
= 5 × 50 = 250** and **1 channel**. Each of the 250 positions is one tree's margin output;
trees are laid out in **contiguous per-client blocks** of 50.

### 5.2 Local training (client side)

Per round, each client (`FlClient`, `client.py:31-342`):
1. In `get_parameters`, optionally applies **local SMOTE/ADASYN** to its raw partition
   (`client.py:170-231`, invoked `:256-263`), then fits its 50-tree XGBoost (`:265-271`).
2. In `fit`, sets the incoming CNN weights, receives the **aggregated list of all clients'
   trees**, converts its local data to the `(N,1,250)` tree-output tensor via
   `single_tree_preds_from_each_client`, and trains the CNN (`client.py:290-327`).

**CNN training loop** (`client.py:88-141`): optimizer **Adam**, `lr = clients.CNN.lr = 0.0005`,
`betas=(0.5, 0.999)`; criterion **`torch.nn.BCELoss`** (`task/Binary_Classification.yaml:9-10`);
one loop of `num_iterations` mini-batch updates per round (batch_size 32).

### 5.3 Federated aggregation — CNN via FedAvg plus tree concatenation

Custom strategy `FedXgbNnAvg(FedAvg)` (`strategy.py:45-79`):
```python
# CNN weights: sample-count weighted FedAvg
weights_results = [(parameters_to_ndarrays(fit_res.parameters[0].parameters),
                    fit_res.num_examples) for _, fit_res in results]
parameters_aggregated = ndarrays_to_parameters(aggregate(weights_results))
# Trees: NOT averaged — collected/concatenated across clients
trees_aggregated = [fit_res.parameters[1] for _, fit_res in results]
return [parameters_aggregated, trees_aggregated], metrics_aggregated
```
So aggregation is **hybrid**: the CNN is FedAvg-averaged; the trees are **concatenated**
(each client's ensemble kept whole, tagged with its cid and sorted `utils.py:311`). The
server broadcasts `[averaged_CNN, all_trees]` back (`server.py:156-175`). Server-side eval
builds the `(N,1,250)` tensor and runs the CNN, scoring AUPRC/F1 with a val-tuned threshold
(`server.py:406-472`).

### 5.4 Neural-network shapes — FedXGBllr Conv1d (CRITICAL)

Layer definitions (`models.py:58-81`):
```python
self.conv1d = nn.Conv1d(in_channels=1, out_channels=n_channel,       # n_channel=64
                        kernel_size=n_estimators_client,             # =50
                        stride=n_estimators_client, padding=0)       # =50
self.layer_direct = nn.Linear(n_channel * client_num, 1)            # =Linear(320,1)
self.relu = nn.ReLU();  self.final_layer = nn.Sigmoid()             # BINARY
```
Because `kernel_size == stride == n_estimators_client == 50`, the convolution slides in
**non-overlapping windows of exactly one client's tree block** → one window per client →
output length = `client_num = 5`.

Forward pass (`models.py:89-105`):

| Step | Op | Tensor shape | Meaning of dims |
|---|---|---|---|
| input | tree-margin stack | `(B, 1, 250)` | channels=1; length=250 = 5 clients × 50 trees |
| `conv1d` | Conv1d(1→64, k=50, s=50) | `(B, 64, 5)` | 5 = one window per client; 64 learned filters |
| `flatten(start_dim=1)` | — | `(B, 320)` | 320 = n_channel(64) × client_num(5) |
| `relu` | ReLU | `(B, 320)` | — |
| `layer_direct` | Linear(320→1) | `(B, 1)` | — |
| `final_layer` | Sigmoid | `(B, 1)` | fraud probability ∈ (0,1) |

**What the CNN learns.** Each conv filter is a weighted combination over a client's 50
tree-margin outputs — i.e. **learnable per-tree contribution weights ("learnable learning
rates")**. The 64 filters give 64 features per client; concatenation over 5 clients (320)
is linearly mapped and squashed by sigmoid into a calibrated fraud probability. BCELoss
trains directly on that sigmoid output; `preds = probs ≥ 0.5` for the accuracy metric,
while AUPRC uses the raw probabilities (`utils.py:359-370`).

### 5.5 Hyperparameters (as set in code, PaySim)

| Param | Value | Source |
|---|---|---|
| trees per client (`n_estimators_client`) | **50** | `clients/paysim_5_clients.yaml:1` |
| clients | 5 | same:3 |
| FL rounds | 50 | same:2 |
| CNN updates/round (`num_iterations`) | **50** | same:8 |
| `xgb.max_depth` | 6 | same:9-10 |
| CNN `lr` | **0.0005** (Adam) | same:11-12 |
| XGBoost `learning_rate` | 0.1 | `base.yaml:20` |
| subsample / alpha / gamma | 0.8 / 5 / 5 | `base.yaml:23-28` |
| Conv `out_channels` (`n_channel`) | 64 | `models.py:58` |
| CNN batch_size | 32 | `base.yaml:56` |
| random_seed | 42 | `base.yaml:15` |

Note the comment at `paysim_5_clients.yaml:4-7`: `num_iterations` was reduced from 500→50
because 500 over-trained the aggregator and drifted AUPRC down (0.97→0.69).

---

## 6. GBM with best-model selection

**Algorithm.** Single sklearn **`HistGradientBoostingClassifier`** per client (chosen over
`GradientBoostingClassifier` for speed on ~4.4M rows — not composite):
```python
# models/gbm_bestmodel/client.py:91-97
HistGradientBoostingClassifier(max_iter=100, learning_rate=0.1, max_depth=6,
    early_stopping=False, random_state=self.seed)
```

**Local training.** Each round the client trains a **fresh** model from scratch on its
(oversampled) partition and **ignores the incoming global model** — there is no warm start
and no `local_epochs` concept (`client.py:108-142`). It reports its local AUPRC.

**Aggregation — best-model selection, not averaging** (Aljunaid et al. 2025,
`W* = argmax_i AUPRC(W_i, V)`):
- Whole sklearn models are shipped as bytes:
  `np.frombuffer(pickle.dumps(model), np.uint8)` → Flower `Parameters` (`client.py:35-51`).
- Server unpickles each client model, scores it on the held-out **server validation set**,
  and promotes the argmax:
```python
# models/gbm_bestmodel/strategy.py (aggregate_fit)
winner = max(eligible, key=lambda c: c["val_auprc"])   # argmax over clients
...
return winner["params"], aggregated_metrics             # broadcast the winning model verbatim
# per-client eval: model.predict_proba(self.x_val)[:,1]; average_precision_score(y_val, scores)
```
(`strategy.py:124-188` selection, `:193-227` evaluation). The winner is broadcast; clients
discard it and retrain anyway.

**Hyperparameters** (`conf/base.yaml`): `num_clients=5, num_rounds=50, seed=42, max_iter=100,
learning_rate=0.1, max_depth=6, early_stopping=False, oversampling=smote, partition=iid`.
Pickle-over-the-wire is flagged unsafe for production (`client.py:8-14`).

---

## 7. FFD — 1-D Conv fraud detector (Yang et al. 2019)

**Algorithm.** A small PyTorch 1-D CNN, `input_dim=13`, 2-class output (`model.py:29-77`).
Not composite; the tabular row **is** the CNN input (contrast FedXGBllr).

### 7.1 Neural-network shapes — FFD

Layer defs (`model.py:34-53`); forward (`:68-77`) — input `(B, 13)` → `unsqueeze(1)`
→ `(B, 1, 13)` (channels=1, length=13 features):

| Step | Op | Shape | Length calc |
|---|---|---|---|
| input | features | `(B, 13)` | — |
| unsqueeze(1) | → conv format | `(B, 1, 13)` | — |
| conv1 | Conv1d(1→32,k=3,p=0)→ReLU | `(B, 32, 11)` | (13−3)+1=11 |
| | MaxPool1d(2) | `(B, 32, 5)` | ⌊(11−2)/2⌋+1=5 |
| conv2 | Conv1d(32→64,k=3,p=0)→ReLU | `(B, 64, 3)` | (5−3)+1=3 |
| | MaxPool1d(2) | `(B, 64, 1)` | ⌊(3−2)/2⌋+1=1 |
| flatten | Flatten | `(B, 64)` | 64×1 (computed dynamically, `:61-66`) |
| fc | Linear(64→512)→ReLU→Dropout(0.5) | `(B, 512)` | — |
| out | Linear(512→2) | `(B, 2)` | 2-class **logits** |

Output = raw logits; fraud probability via `softmax(logits)[:,1]` at inference (`model.py:88-104`).
Note the "length" dimension is the 13 features treated as an ordered 1-D sequence —
unconventional for tabular data, but faithful to the Yang et al. adaptation (original: 30
PCA features).

**Local training** (`client.py:75-99`): **`nn.CrossEntropyLoss`**, **`torch.optim.SGD`**,
`lr=0.01`, `batch_size=80`, `local_epochs=5`. Oversampling is applied **per-round inside the
client** (`client.py:107-149`), not once up front. Local AUPRC reported for aggregation.

**Aggregation — `AccuracyWeightedFedAvg`, NOT vanilla FedAvg** (`strategy.py:46-106`):
```
w_{t+1} = Σ_c (n_c/n · α_c · w_c) / Σ_c (n_c/n · α_c),   α_c = client c's local AUPRC
```
with fallback to plain FedAvg (data-size weights only) when all `α_c = 0` at cold start
(`strategy.py:72-78`). This differs from the README's "weights are averaged via FedAvg."

**Hyperparameters** (`conf/base.yaml`): `num_clients=5, num_rounds=50, local_epochs=5,
batch_size=80, lr=0.01, seed=42, oversampling=smote, partition=iid`.

---

## 8. "BERT" fraud model — actually an FT-Transformer (extra, not real BERT)

**What it really is.** Not a HuggingFace/pretrained BERT and **no tokenizer** — a from-scratch
PyTorch **Feature-Tokenizer Transformer** (Gorishniy et al. 2021), BERT-*style* only in that
it uses a learnable `[CLS]` token + self-attention (`model.py:1-20, 31-101`).

**Tabular → sequence.** No bucketization, no text serialization of column names. Each of the
13 scalar features gets its own learned projection to a token:
```python
# model.py:48-54, 91-95
self.feature_weights = nn.Parameter(torch.empty(13, 64))   # W_i ∈ R^64 per feature
self.feature_biases  = nn.Parameter(torch.zeros(13, 64))
self.cls_token       = nn.Parameter(torch.zeros(1, 1, 64))
...
tokens = x.unsqueeze(-1) * self.feature_weights + self.feature_biases  # (B,13,64)
tokens = torch.cat([cls.expand(B,-1,-1), tokens], dim=1)               # (B,14,64)
```

### 8.1 Neural-network shapes — FT-Transformer

| Step | Op | Shape |
|---|---|---|
| input | features | `(B, 13)` |
| tokenize | `x_i·W_i + b_i` | `(B, 13, 64)` |
| prepend `[CLS]` | concat | `(B, 14, 64)` |
| encoder | 2× `TransformerEncoderLayer` (d_model=64, nhead=4, ff=256, GELU, Pre-LN, batch_first) | `(B, 14, 64)` |
| take `[CLS]` | `out[:,0,:]` | `(B, 64)` |
| head | LayerNorm→Linear(64→64)→GELU→Dropout→Linear(64→2) | `(B, 2)` logits |

Config (`conf/base.yaml`): `d_model=64, nhead=4 (16 dims/head), num_layers=2,
dim_feedforward=256, dropout=0.1`; **max sequence length = 14** (1 CLS + 13 features),
implicit. Layer/head defs at `model.py:57-75`.

**Local training** (`client.py:84-112`): `CrossEntropyLoss`, **AdamW**, `lr=0.001`,
`weight_decay=1e-4`, `batch_size=64`, `local_epochs=3`. **Aggregation = the same
`AccuracyWeightedFedAvg`** as FFD (`strategy.py`). Hyperparameters otherwise mirror the
others (`num_clients=5, num_rounds=50, seed=42, oversampling=smote`).

---

## 9. Neural-network summary (all NNs in one place)

| | FedXGBllr CNN | FFD CNN | BERT/FT-Transformer |
|---|---|---|---|
| Input to net | tree margins `(B,1,250)` | features `(B,1,13)` | features `(B,13)`→tokens `(B,14,64)` |
| Core | Conv1d(1→64,k=50,s=50) | 2× Conv1d+MaxPool | 2× TransformerEncoderLayer |
| Output | `(B,1)` **Sigmoid** | `(B,2)` logits→softmax | `(B,2)` logits→softmax |
| Loss | **BCELoss** | CrossEntropyLoss | CrossEntropyLoss |
| Optimizer / lr | Adam / 5e-4 | SGD / 1e-2 | AdamW / 1e-3 |
| Aggregation | FedAvg (CNN) + tree concat | AccuracyWeightedFedAvg | AccuracyWeightedFedAvg |

---

## 10. Proposal vs. code discrepancies

> Compared against the model list in the task prompt and the repo `README.md` (the `.docx`
> proposals are binary and were not opened). Flag accordingly.

1. **"BERT" is not BERT, and is an undocumented extra.** `models/bert_fraud/` exists and is
   fully wired, but the README's "Models compared" lists only 5 (no BERT). And it is a
   from-scratch **FT-Transformer** (Gorishniy 2021) with per-feature linear tokenization —
   no pretrained weights, no tokenizer, no text. Anyone expecting HuggingFace BERT will be
   misled (`model.py:1-20`).

2. **FFD (and BERT) do not use plain FedAvg.** README §5 says FFD "weights are averaged via
   FedAvg," but the code uses **`AccuracyWeightedFedAvg`** — weights by `n_c · local_AUPRC`
   (`ffd/strategy.py:46-106`). Only LR and SVM use stock FedAvg.

3. **`num_rounds` config defaults vs. README sweep values.** LR/SVM/GBM `conf/base.yaml` all
   default `num_rounds: 50`, but README's run table says LR/SVM=20 and GBM=10 rounds. The
   sweep shell scripts override at launch, so the *effective* rounds depend on the driver,
   not the YAML default.

4. **GBM is `HistGradientBoostingClassifier`, not textbook GBM**, and clients **ignore the
   global model** and retrain from scratch each round — "best-model selection" here means the
   server keeps the single best client model; there is no cross-client knowledge transfer
   into local training (`gbm_bestmodel/client.py:91-142`).

5. **FedXGBllr tree count.** Code uses **50 trees/client** (`n_estimators_client=50`) → 250
   total, with `max_depth=6`, XGBoost `learning_rate=0.1`, CNN `lr=0.0005`, `num_iterations=50`.
   If the proposal quoted different numbers (e.g. 100 trees / 500 CNN iters), the code was
   deliberately tuned down — see the inline note that 500 iterations drifted AUPRC 0.97→0.69
   (`paysim_5_clients.yaml:4-7`).

6. **SVM uses SGD, not a batch SVM** (`SGDClassifier(loss="hinge")`), a deliberate deviation
   so FedAvg can progress; the centralized SVM upper bound uses batch `LinearSVC`.

7. **Two distinct "1-D CNN" models exist** and are easy to conflate: the FedXGBllr aggregator
   CNN (consumes 250 tree margins, sigmoid+BCE) and the FFD detector CNN (consumes 13 raw
   features, softmax+CE). They share no code.
