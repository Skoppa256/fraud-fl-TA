# hfedxgboost — Technical Analysis & PaySim Integration Report

---

## TASK 1 — PROJECT STRUCTURE INVENTORY

### Directory layout

```
hfedxgboost/
├── LICENSE
├── README.md
├── pyproject.toml
├── results.csv                          # Empty/log file for FL experiments
├── results_centralized.csv              # Empty/log file for centralized experiments
└── hfedxgboost/                         # Python package
    ├── __init__.py                      # Package marker (one-line docstring)
    ├── main.py                          # Hydra entrypoint
    ├── client.py                        # Flower client (FlClient)
    ├── server.py                        # Custom Flower server + serverside_eval
    ├── strategy.py                      # FedXgbNnAvg strategy (extends FedAvg)
    ├── models.py                        # CNN + fit_xgboost
    ├── dataset.py                       # load_single_dataset, divide_dataset_between_clients
    ├── dataset_preparation.py           # download_data, datafiles_fusion, train_test_split, modify_labels
    ├── utils.py                         # CNN test loop, single-tree prediction, EarlyStop, ResultsWriter, etc.
    ├── sweep.yaml                       # W&B sweep config
    └── conf/                            # Hydra config tree
        ├── base.yaml                    # default FL config
        ├── Centralized_Baseline.yaml    # centralized-XGBoost config
        ├── centralized_basline_all_datasets_paper_config.yaml
        ├── dataset/                     # {a9a,cod_rna,ijcnn1,space_ga,abalone,cpusmall,YearPredictionMSD}.yaml
        │   └── task/{Binary_Classification,Regression}.yaml
        ├── clients/                     # 24 files: <dataset>_{2,5,10}_clients.yaml + paper_{2,5,10}_clients.yaml
        ├── xgboost_params_centralized/  # paper_, abalone_, cpusmall_, YearPredictionMSD_ ...
        └── wandb/default.yaml
```

### Source-file purpose summaries

- **pyproject.toml** — Poetry build. Pins Python `>=3.10.0, <3.11.0`. Dependencies: `flwr[simulation]==1.5.0`, `hydra-core==1.3.2`, `torch==2.8.0`, `scikit-learn==1.5.0`, `xgboost==2.0.0`, `torchmetrics==1.8.2`, `tqdm==4.66.3`, `torchvision==0.23.0`, `wandb==0.15.12`. Dev: black, isort, mypy, pylint, flake8, pytest, ruff. (No `imblearn`/SMOTE; no `pandas`.)
- **main.py** — Hydra entry. Loads config; if `cfg.centralized`, runs centralized XGBoost via `run_centralized`; otherwise loads data, partitions IID, instantiates strategy, defines `client_fn`, and calls `fl.simulation.start_simulation` with a custom `FlServer`.
- **client.py** — `FlClient(fl.client.Client)`. Owns a `CNN`. `get_parameters` trains the local XGBoost ensemble and returns `(cnn_weights, (tree, cid))`. `fit` receives aggregated CNN weights + aggregated trees from the server, expands each tree into per-tree margin predictions via `single_tree_preds_from_each_client`, then trains the CNN on those tree-output features.
- **server.py** — `FlServer(fl.server.Server)` overrides `fit`, `fit_round`, `_get_initial_parameters`, `evaluate_round` to handle the `(Parameters, trees_list)` tuple format. `serverside_eval(server_round, parameters, ...)` rebuilds the CNN with aggregated weights, transforms the centralized test loader through the aggregated trees, and runs `test()`.
- **strategy.py** — `FedXgbNnAvg(FedAvg)`. `aggregate_fit` does FedAvg on the CNN weight vectors (weighted by `num_examples`) AND **concatenates** every client's XGBoost tree (`trees_aggregated = [fit_res.parameters[1] for _, fit_res in results]`); it does NOT average the trees.
- **models.py** — `fit_xgboost(config, task_type, x_train, y_train, n_estimators)` builds either an `XGBClassifier` or `XGBRegressor` from Hydra and calls `.fit`. `CNN(nn.Module)` is a 1-D conv (`kernel=n_estimators_client`, `stride=n_estimators_client`, `out_channels=64`) followed by ReLU → Linear(64×client_num, 1) → Sigmoid for BINARY / Identity for REG.
- **dataset.py** — `load_single_dataset` calls the downloader, fuses files, splits train/test, and normalizes labels. `divide_dataset_between_clients` does **uniform IID `random_split`** of the train set across `pool_size` clients with optional val split.
- **dataset_preparation.py** — `download_data` is a giant `match` over hard-coded dataset names that downloads LIBSVM/.bz2 files into `./dataset/<name>/`. `datafiles_fusion` uses `sklearn.datasets.load_svmlight_file` only. `train_test_split` is a hand-rolled shuffle+slice using `np.random.seed(2023)`. `modify_labels` flips `-1 → 0` for binary classification.
- **utils.py** — Dataset-task dict; `evaluate` (accuracy or MSE); `run_single_exp`/`run_centralized` for centralized XGBoost experiments; `local_clients_performance` (per-client XGBoost-only sanity check); `single_tree_prediction`/`single_tree_preds_from_each_client` which iterate each tree in each client's ensemble to produce `(N, client_num * n_estimators_client)` margin features used as the CNN input; `test` runs CNN evaluation; `EarlyStop`; `ResultsWriter`/`CentralizedResultsWriter`/`create_res_csv`.
- **conf/base.yaml** — defaults: dataset=cpusmall, clients=cpusmall_5_clients; sets `batch_size: "whole"`, `val_ratio: 0.0`, `centralized: False`, the XGBoost block, server resources, FedXgbNnAvg strategy, and `run_experiment.num_rounds` etc.
- **conf/dataset/\*.yaml** — `defaults: task: Binary_Classification|Regression`, `dataset_name`, `train_ratio` (0.75), `early_stop_patience_rounds`.
- **conf/dataset/task/Binary_Classification.yaml** — `task_type: BINARY`, `torchmetrics.Accuracy`, `torch.nn.BCELoss`, `xgboost.XGBClassifier`.
- **conf/dataset/task/Regression.yaml** — `task_type: REG`, `MeanSquaredError`, `MSELoss`, `XGBRegressor`.
- **conf/clients/\*.yaml** — `n_estimators_client`, `num_rounds`, `client_num`, `num_iterations`, `xgb.max_depth`, `CNN.lr`. Example: `a9a_5_clients`: 100/30/5/500/8/0.0005.
- **conf/xgboost_params_centralized/\*.yaml** — centralized XGBoost hyperparams.
- **conf/wandb/default.yaml** — wandb project name `p1`.
- **sweep.yaml** — W&B random sweep over `n_estimators_client`, `num_iterations`, `xgb.max_depth`, `CNN.lr`.
- **README.md** — overview of paper, datasets, hyperparams, environment, run commands.

---

## TASK 2 — ENVIRONMENT SETUP

The README mandates Poetry + pyenv. Step-by-step:

### 1. Python version
- **`>=3.10.0, <3.11.0`** (from `pyproject.toml:40`). README pins `3.10.6`.

### 2. Virtual environment (Poetry path, as in README)
```bash
cd /Users/deo/Documents/Kuliah/6/TA/hfedxgboost
pyenv local 3.10.6
poetry env use 3.10.6
poetry install
poetry shell
```

### 2-alt. Plain venv (if you don't use Poetry)
```bash
cd /Users/deo/Documents/Kuliah/6/TA/hfedxgboost
python3.10 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

### 3. Exact pip install (pulling the actual list from `pyproject.toml:39-49`)
```bash
pip install \
  "flwr[simulation]==1.5.0" \
  "hydra-core==1.3.2" \
  "torch==2.8.0" \
  "scikit-learn==1.5.0" \
  "xgboost==2.0.0" \
  "torchmetrics==1.8.2" \
  "tqdm==4.66.3" \
  "torchvision==0.23.0" \
  "wandb==0.15.12"
```

(For PaySim integration you will also need `pandas` and `imbalanced-learn`; not in the baseline.)

### 4. Additional setup
- **Datasets download automatically** the first time you run a config — `download_data()` in `dataset_preparation.py:20` creates `./dataset/<name>/` and `urllib.request.urlretrieve`s LIBSVM files from `csie.ntu.edu.tw`. No env var needed.
- **W&B**: only if you set `use_wandb: True`. Then run `wandb login`. The default project is `p1` (`conf/wandb/default.yaml:2`).
- No other env variables.

### 5. Smoke test
```bash
# Quick centralized run on a tiny dataset (downloads ~140KB):
python -m hfedxgboost.main --config-name Centralized_Baseline dataset=abalone xgboost_params_centralized=abalone_xgboost_centralized

# Or a tiny FL run (2 clients):
python -m hfedxgboost.main dataset=a9a clients=a9a_2_clients run_experiment.num_rounds=2
```
A successful run prints config YAML, `FL starting`, per-round `fit progress` lines, and a `History` object.

---

## TASK 3 — SYSTEM ARCHITECTURE & DATA FLOW

### A) Entry point and execution trace

`python -m hfedxgboost.main ...` → `main.py:34` is `@hydra.main(config_path="conf", config_name="base")`. Trace:

1. `OmegaConf.to_yaml(cfg)` printed (`main.py:44`).
2. Branch on `cfg.centralized` (`main.py:46-56`): if true, call `run_centralized` and write to `results_centralized.csv`.
3. Otherwise (FL):
   - `EarlyStop(cfg)` instantiated (`main.py:62`).
   - `load_single_dataset(...)` → `(x_train, y_train, x_test, y_test)` (`main.py:63-67`).
   - `divide_dataset_between_clients(...)` → `trainloaders, valloaders, testloader` (`main.py:69-75`).
   - Strategy `FedXgbNnAvg` instantiated via Hydra, with `evaluate_fn = serverside_eval` (`main.py:94-105`).
   - `FlServer(cfg, SimpleClientManager(), early_stopper, strategy)` (`main.py:119-123`).
   - `fl.simulation.start_simulation(client_fn=..., server=..., num_clients=..., strategy=...)` (`main.py:117-129`).
   - After it ends, `ResultsWriter(cfg).write_res("results.csv")`.

### B) FL training loop

#### Server init
`FlServer.fit(num_rounds, timeout)` (`server.py:212-266`):
- `_get_initial_parameters` samples one random client and calls its `get_parameters` (`server.py:329-347`); the client fits a brand-new XGBoost (`n_estimators=100`) on its local data (`client.py:185-194`) and returns `(CNN weights, (tree, cid))`. That tuple becomes `self.parameters`.
- Initial evaluation at round 0 via `self.strategy.evaluate` → `serverside_eval`.

#### Round k (k=1..num_rounds)
- `check_res_cen(current_round, ...)` calls `fit_round`:
  - `strategy.configure_fit` sends each client the current `(cnn_params, trees)` tuple.
  - `fit_clients(...)` executes each client's `fit` in parallel.
  - **Inside the client `fit`** (`client.py:196-261`):
    - Load new CNN weights from `ins.parameters[0]`.
    - Extract aggregated trees list from `ins.parameters[1]`.
    - `single_tree_preds_from_each_client` (`utils.py:264-328`) iterates each tree from each client's ensemble and produces an `(N, 1, client_num * n_estimators_client)` tensor of single-tree margin predictions. This becomes the new `trainloader`/`valloader`.
    - CNN trains for `num_iterations` (Adam, `betas=(0.5, 0.999)`, lr from `clients.CNN.lr`) (`client.py:119-136`).
    - `get_parameters` is called again — **a fresh local XGBoost is fit each round** (n_estimators=100, hardcoded) (`client.py:188-190`).
    - Returns `(CNN weights, (new tree, cid))` plus `num_examples` and metrics.
  - Server aggregates results via `strategy.aggregate_fit` (see C/Aggregation).
- `strategy.evaluate(round, parameters)` → `serverside_eval` (`server.py:350-406`) projects the centralized `testloader` through aggregated trees and runs `test()`.
- `EarlyStop.early_stop` checks loss patience; if `patience_rounds` rounds without improvement, fit returns early.
- `evaluate_round` is also called but `fraction_evaluate=0.0` in `base.yaml:43` → no clients sample, no-op.

#### Aggregation (`FedXgbNnAvg.aggregate_fit`, `strategy.py:45-79`)
```python
weights_results = [(parameters_to_ndarrays(fit_res.parameters[0].parameters), fit_res.num_examples) for _, fit_res in results]
parameters_aggregated = ndarrays_to_parameters(aggregate(weights_results))    # FedAvg on CNN
trees_aggregated = [fit_res.parameters[1] for _, fit_res in results]           # LIST of (tree, cid)
return [parameters_aggregated, trees_aggregated], metrics_aggregated
```
The CNN weights are weighted-averaged; the trees are **not averaged but collected as a list**, so each round every client gets every other client's freshly-trained XGBoost ensemble.

#### Configurables
- `clients.num_rounds` → server rounds.
- `clients.client_num` → number of clients (and `min_available_clients` in strategy).
- `clients.num_iterations` → CNN updates per round (per client).
- `clients.n_estimators_client` → trees per client.
- Local "epochs" don't exist literally; training is parameterized by `num_iterations` mini-batch updates (note `batch_size: "whole"` in base.yaml ⇒ 1 update == 1 pass).

### C) Model — `FedXGBllr`

1. **Local XGBoost training (per client)** — In `FlClient.get_parameters` (`client.py:185-194`) the client iterates its `trainloader_original` (always a single batch because `batch_size="whole"`) and calls `fit_xgboost(self.config, task_type, data, label, 100)`. The `100` is hardcoded and overrides `cfg.n_estimators_client`. The XGB hyperparams come from `cfg.XGBoost` (objective, max_depth, lr, etc.).
2. **Tree ensembles sent to server** — Each `FitRes.parameters` is a 2-tuple `(CNN weights as Parameters, (XGBClassifier|XGBRegressor, cid))`. Server collects them in a list (`strategy.py:69`).
3. **CNN with FedAvg learning per-tree learning rates** — `CNN` (`models.py:55-105`) has a single `Conv1d(in=1, out=64, kernel=n_estimators_client, stride=n_estimators_client)` over the per-tree margin predictions (shape `(N, 1, client_num*n_estimators_client)`), so each non-overlapping window of one client's `n_estimators_client` trees is convolved into a 64-channel vector — this is the *per-tree learnable weighting*. Then `Flatten → ReLU → Linear(64·client_num, 1) → Sigmoid/Identity`. Optimizer is `torch.optim.Adam(lr=cfg.clients.CNN.lr, betas=(0.5, 0.999))` (`client.py:119-121`). CNN weights are aggregated by classical FedAvg in `aggregate_fit`.

### D) Configuration system

**Hydra + OmegaConf**. Config tree under `conf/` with `defaults:` composition. Override syntax on the CLI: `dataset=a9a`, `clients=a9a_5_clients`, `clients.CNN.lr=0.0003`, `--multirun`, etc. No argparse anywhere.

#### Configurable parameters and defaults
From `base.yaml`:

| Key | Default |
|---|---|
| `centralized` | `False` |
| `use_wandb` | `False` |
| `show_each_client_performance_on_its_local_data` | `False` |
| `val_ratio` | `0.0` |
| `batch_size` | `"whole"` |
| `n_estimators_client` | `${clients.n_estimators_client}` |
| `task_type` | `${dataset.task.task_type}` |
| `client_num` | `${clients.client_num}` |
| `XGBoost.learning_rate` | `0.1` |
| `XGBoost.max_depth` | `${clients.xgb.max_depth}` (=8) |
| `XGBoost.n_estimators` | `${clients.n_estimators_client}` |
| `XGBoost.subsample` | `0.8` |
| `XGBoost.colsample_{bylevel,bynode,bytree}` | `1` |
| `XGBoost.alpha`, `gamma` | `5`, `5` |
| `XGBoost.num_parallel_tree`, `min_child_weight` | `1`, `1` |
| `server.max_workers` | `"None"` (string!) |
| `server.device` | `"cpu"` |
| `client_resources.num_cpus` | `1` |
| `client_resources.num_gpus` | `0.0` |
| `strategy._target_` | `hfedxgboost.strategy.FedXgbNnAvg` |
| `strategy.fraction_fit` | `1.0` |
| `strategy.fraction_evaluate` | `0.0` |
| `strategy.min_fit_clients` | `1` |
| `strategy.min_available_clients` | `${client_num}` |
| `strategy.accept_failures` | `False` |
| `run_experiment.num_rounds` | `${clients.num_rounds}` |
| `run_experiment.batch_size` | `32` |
| `run_experiment.fit_config.num_iterations` | `${clients.num_iterations}` |

Per-client-count file (e.g., `a9a_5_clients.yaml`) sets `n_estimators_client`, `num_rounds`, `client_num`, `num_iterations`, `xgb.max_depth`, `CNN.lr`.

Per-dataset file (e.g., `a9a.yaml`) sets `dataset_name`, `train_ratio`, `early_stop_patience_rounds`, and pulls the task type.

### E) Data loading

- Format: **LIBSVM only**. `datafiles_fusion` (`dataset_preparation.py:174-196`) uses `sklearn.datasets.load_svmlight_file(path, zero_based=False)`. No CSV, no parquet, no HF datasets.
- Pipeline call site: `main.py:63-75` → `load_single_dataset` → `download_data` → `datafiles_fusion` → `train_test_split` → optional `modify_labels` → `TensorDataset(torch.from_numpy(x), torch.from_numpy(y))` → `divide_dataset_between_clients(... random_split ...)`.
- **No normalization / scaling**: data is passed straight from `load_svmlight_file` (often already scaled in LIBSVM datasets) to `np.ndarray` → torch tensors. There is **no `StandardScaler`, no one-hot encoder, no preprocessing** anywhere in the code.
- **Partitioning is strictly uniform IID via `torch.utils.data.random_split` with seed 0** (`dataset.py:127`). **No Dirichlet, no label-skew, no quantity-skew.**
- The README also mentions a `.bz2` handler and the warning that CSVs need `datafiles_fusion` to be altered.

---

## TASK 4 — INTEGRATING THE PaySim DATASET

### 4.1 — Which files handle data loading/partitioning?

- **Loading** — `hfedxgboost/dataset_preparation.py` (`download_data`, `datafiles_fusion`, `train_test_split`, `modify_labels`).
- **Pipeline orchestrator** — `hfedxgboost/dataset.py` (`load_single_dataset`, `divide_dataset_between_clients`, `get_dataloader`).
- **Train tensor → TensorDataset bridging + train/test driving** — `hfedxgboost/main.py:63-75`.
- **XGBoost fit per round** — `hfedxgboost/client.py:185-194` (`get_parameters`).

Relevant code (the IID partitioner):

```python
# dataset.py:120-143
trainset_length = len(trainset)
lengths = [trainset_length // pool_size] * pool_size
if sum(lengths) != trainset_length:
    lengths[-1] = trainset_length - sum(lengths[0:-1])
datasets = random_split(trainset, lengths, torch.Generator().manual_seed(0))
```

### 4.2 — Exact changes to replace dataset with PaySim CSV

#### A) Add `pandas` + `imbalanced-learn` to `pyproject.toml`
**Before** (`pyproject.toml:39-49`):
```toml
[tool.poetry.dependencies]
python = ">=3.10.0, <3.11.0"
flwr = { extras = ["simulation"], version = "1.5.0" }
hydra-core = "1.3.2"
torch = "2.8.0"
scikit-learn = "1.5.0"
xgboost = "2.0.0"
torchmetrics = "1.8.2"
tqdm = "4.66.3"
torchvision = "0.23.0"
wandb = "0.15.12"
```
**After**:
```toml
[tool.poetry.dependencies]
python = ">=3.10.0, <3.11.0"
flwr = { extras = ["simulation"], version = "1.5.0" }
hydra-core = "1.3.2"
torch = "2.8.0"
scikit-learn = "1.5.0"
xgboost = "2.0.0"
torchmetrics = "1.8.2"
tqdm = "4.66.3"
torchvision = "0.23.0"
wandb = "0.15.12"
pandas = "2.2.2"
imbalanced-learn = "0.12.3"
```

#### B) Add the PaySim case to `download_data` (`dataset_preparation.py:37-171`)
You need to put `PS_20174392719_1491204439457_log.csv` (the Kaggle file) at `./dataset/paysim/paysim.csv` (no automatic download — Kaggle requires auth).
Add a new case **before** the `case _:` line (`dataset_preparation.py:169`):
```python
case "paysim":
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(
            f"Place PaySim CSV manually at {dataset_path}/paysim.csv"
            " (download from Kaggle: ealaxi/paysim1)."
        )
    return_list = [os.path.join(dataset_path, "paysim.csv")]
```

#### C) Replace `datafiles_fusion` (or branch on extension) to support CSV + preprocessing
**Before** (`dataset_preparation.py:174-196`):
```python
def datafiles_fusion(data_paths):
    data = load_svmlight_file(data_paths[0], zero_based=False)
    X = data[0].toarray()
    Y = data[1]
    for i in range(1, len(data_paths)):
        data = load_svmlight_file(data_paths[i], zero_based=False, n_features=X.shape[1])
        X = np.concatenate((X, data[0].toarray()), axis=0)
        Y = np.concatenate((Y, data[1]), axis=0)
    return X, Y
```
**After**:
```python
import pandas as pd
from sklearn.preprocessing import StandardScaler

PAYSIM_DROP = ["nameOrig", "nameDest", "isFlaggedFraud"]
PAYSIM_CAT  = ["type"]

def _load_paysim_csv(path):
    df = pd.read_csv(path)
    df = df.drop(columns=PAYSIM_DROP, errors="ignore")
    # Feature engineering
    df["errorBalanceOrig"] = df["newbalanceOrig"] - df["oldbalanceOrg"] + df["amount"]
    df["errorBalanceDest"] = df["oldbalanceDest"] + df["amount"] - df["newbalanceDest"]
    # One-hot encode `type`
    df = pd.get_dummies(df, columns=PAYSIM_CAT, drop_first=False, dtype=np.float32)
    y = df["isFraud"].to_numpy(dtype=np.float32)
    X = df.drop(columns=["isFraud"]).to_numpy(dtype=np.float32)
    return X, y

def datafiles_fusion(data_paths):
    first = data_paths[0]
    if first.endswith(".csv"):
        return _load_paysim_csv(first)
    # ... existing svmlight path ...
    data = load_svmlight_file(first, zero_based=False)
    X, Y = data[0].toarray(), data[1]
    for i in range(1, len(data_paths)):
        data = load_svmlight_file(data_paths[i], zero_based=False, n_features=X.shape[1])
        X = np.concatenate((X, data[0].toarray()), axis=0)
        Y = np.concatenate((Y, data[1]), axis=0)
    return X, Y
```
Note `StandardScaler` is applied **after** the train/test split (see 4.5).

#### D) Stratified split — replace `train_test_split` for PaySim (`dataset_preparation.py:199-241`)
**Before**:
```python
def train_test_split(X, y, train_ratio=0.75):
    np.random.seed(2023)
    y = np.expand_dims(y, axis=1)
    full = np.concatenate((X, y), axis=1)
    np.random.shuffle(full)
    ...
```
**After**: use sklearn's stratified split (so the 0.13% fraud rate survives):
```python
from sklearn.model_selection import train_test_split as sk_split

def train_test_split(X, y, train_ratio=0.75, stratify=False):
    if stratify:
        x_train, x_test, y_train, y_test = sk_split(
            X, y, train_size=train_ratio, random_state=2023, stratify=y
        )
    else:
        np.random.seed(2023)
        y_ = np.expand_dims(y, axis=1)
        full = np.concatenate((X, y_), axis=1)
        np.random.shuffle(full)
        ...
    for arr in (x_train, y_train, x_test, y_test):
        arr.flags.writeable = True
    return x_train, y_train, x_test, y_test
```
Then in `dataset.py:47` pass `stratify=(dataset_name == "paysim")` (or always for `BINARY`). Also apply `StandardScaler` on numerical columns of `x_train` and transform `x_test`.

#### E) Add a `task_type` mapping entry in `utils.py:27-35`
```python
dataset_tasks = {
    "a9a": "BINARY",
    "cod-rna": "BINARY",
    "ijcnn1": "BINARY",
    "abalone": "REG",
    "cpusmall": "REG",
    "space_ga": "REG",
    "YearPredictionMSD": "REG",
    "paysim": "BINARY",          # NEW
}
```

#### F) Add Hydra config files
Create `hfedxgboost/conf/dataset/paysim.yaml`:
```yaml
defaults:
  - task: Binary_Classification
dataset_name: "paysim"
train_ratio: 0.75
early_stop_patience_rounds: 10
# PaySim-specific
stratify: true
test_size: 0.10        # of full
val_size: 0.10         # of full (kept centralized on server)
non_iid:
  enabled: true
  alpha: 0.5           # Dirichlet alpha
  partition_by: "type" # or "isFraud" or a numerical bin
smote:
  enabled: true
  sampling_strategy: 0.1   # minority:majority ratio after SMOTE
  k_neighbors: 5
```

Create `hfedxgboost/conf/clients/paysim_5_clients.yaml`:
```yaml
n_estimators_client: 100
num_rounds: 30
client_num: 5
num_iterations: 500
xgb:
  max_depth: 8
CNN:
  lr: .0005
```

#### G) Loss class weighting (recommended, optional)
Because of imbalance, switching from `nn.BCELoss` to `nn.BCEWithLogitsLoss` with `pos_weight` would help — but that requires removing `Sigmoid` from the CNN head (`models.py:78`) and is a deeper change. With SMOTE in place you can keep BCE.

### 4.3 — Dirichlet non-IID partitioning

**The baseline only supports uniform IID** (`random_split` in `dataset.py:127`). New code is required.

Add (or replace) inside `divide_dataset_between_clients` in `hfedxgboost/dataset.py`:

```python
import numpy as np
from torch.utils.data import Subset

def _dirichlet_indices(labels: np.ndarray, num_clients: int, alpha: float, seed: int = 0):
    """Split sample indices via Dirichlet over label classes (Hsu et al., 2019)."""
    rng = np.random.default_rng(seed)
    classes = np.unique(labels)
    client_idx = [[] for _ in range(num_clients)]
    for c in classes:
        idx_c = np.where(labels == c)[0]
        rng.shuffle(idx_c)
        proportions = rng.dirichlet([alpha] * num_clients)
        # avoid empty shards: clip at least 1 per client if class has enough samples
        splits = (np.cumsum(proportions) * len(idx_c)).astype(int)[:-1]
        for ci, part in enumerate(np.split(idx_c, splits)):
            client_idx[ci].extend(part.tolist())
    return [np.array(ix) for ix in client_idx]


def divide_dataset_between_clients(
    trainset, testset, pool_size, batch_size, val_ratio=0.0,
    non_iid_alpha: float | None = None,
    labels: np.ndarray | None = None,
):
    if non_iid_alpha is not None and labels is not None:
        idx_per_client = _dirichlet_indices(labels, pool_size, non_iid_alpha)
        datasets = [Subset(trainset, ix.tolist()) for ix in idx_per_client]
    else:
        # existing uniform IID path
        lengths = [len(trainset) // pool_size] * pool_size
        if sum(lengths) != len(trainset):
            lengths[-1] = len(trainset) - sum(lengths[0:-1])
        datasets = random_split(trainset, lengths, torch.Generator().manual_seed(0))
    # ...existing val split + DataLoader creation...
```

Then plumb the new args through `main.py:69-75`:
```python
trainloaders, valloaders, testloader = divide_dataset_between_clients(
    TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
    TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y_test)),
    batch_size=cfg.batch_size,
    pool_size=cfg.clients.client_num,
    val_ratio=cfg.val_ratio,
    non_iid_alpha=cfg.dataset.non_iid.alpha if cfg.dataset.non_iid.enabled else None,
    labels=y_train,
)
```

### 4.4 — Local SMOTE per client

The right insertion point is `FlClient.get_parameters` in `hfedxgboost/client.py:185-194` (and the local subset that flows through `single_tree_preds_from_each_client`). Because `batch_size="whole"`, the client's data lives in `self.trainloader_original` as one `(data, label)` batch.

Replace the inner loop:

```python
# client.py:185-194 — BEFORE
def get_parameters(self, ins):
    for dataset in self.trainloader_original:
        data, label = dataset[0], dataset[1]
    tree = fit_xgboost(self.config, self.config.dataset.task.task_type, data, label, 100)
    return GetParametersRes(...), (tree, int(self.cid))
```

```python
# client.py — AFTER
from imblearn.over_sampling import SMOTE

def _apply_smote(self, X, y):
    smote_cfg = self.config.dataset.smote
    if not smote_cfg.enabled:
        return X, y
    X_np = X.detach().cpu().numpy() if torch.is_tensor(X) else np.asarray(X)
    y_np = y.detach().cpu().numpy() if torch.is_tensor(y) else np.asarray(y)
    y_np = y_np.astype(int).ravel()
    if np.unique(y_np).size < 2 or (y_np == 1).sum() < smote_cfg.k_neighbors + 1:
        return X, y     # not enough minority samples on this client
    sm = SMOTE(
        sampling_strategy=smote_cfg.sampling_strategy,
        k_neighbors=smote_cfg.k_neighbors,
        random_state=int(self.cid),
    )
    X_res, y_res = sm.fit_resample(X_np, y_np)
    return (
        torch.from_numpy(X_res.astype(np.float32)),
        torch.from_numpy(y_res.astype(np.float32)),
    )

def get_parameters(self, ins):
    for dataset in self.trainloader_original:
        data, label = dataset[0], dataset[1]
    data, label = self._apply_smote(data, label)
    tree = fit_xgboost(self.config, self.config.dataset.task.task_type, data, label, 100)
    # Replace the loader so subsequent `fit` rounds train CNN on rebalanced data too:
    self.trainloader_original = DataLoader(
        TensorDataset(data, label),
        batch_size=len(data), pin_memory=True, shuffle=True,
    )
    return GetParametersRes(
        status=Status(Code.OK, ""),
        parameters=ndarrays_to_parameters(self.net.get_weights()),
    ), (tree, int(self.cid))
```

This is the **only place each client touches its raw features**; placing SMOTE here covers both the XGBoost fit and the downstream CNN tree-features pipeline.

### 4.5 — Preprocessing pipeline placement

- **Column dropping & feature engineering & one-hot encoding** — inside `_load_paysim_csv` in `dataset_preparation.py` (see 4.2-C). These are deterministic from raw columns and safe to do before splitting.
- **StandardScaler** — must be **fit on train only**, applied to test/val. Best place: inside `load_single_dataset` in `hfedxgboost/dataset.py:26-72`, **after** the `train_test_split` call:

```python
# dataset.py
from sklearn.preprocessing import StandardScaler

def load_single_dataset(task_type, dataset_name, train_ratio=0.75):
    datafiles_paths = download_data(dataset_name)
    X, Y = datafiles_fusion(datafiles_paths)
    x_train, y_train, x_test, y_test = train_test_split(
        X, Y, train_ratio=train_ratio,
        stratify=(task_type.upper() == "BINARY"),
    )
    if dataset_name == "paysim":
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train).astype(np.float32)
        x_test  = scaler.transform(x_test).astype(np.float32)
        y_train = y_train.astype(np.float32)
        y_test  = y_test.astype(np.float32)
    elif task_type.upper() == "BINARY":
        y_train, y_test = modify_labels(y_train, y_test)
    ...
```
SMOTE goes **after** scaling (per client) — see 4.4.

### 4.6 — Stratified train/val/test split & centralized eval

The baseline only has train/test (`train_ratio`) plus an optional **per-client** val (`val_ratio`) that is **never used** because `val_ratio=0.0` in `base.yaml`. To get a **centralized val + test on the server**:

1. In `dataset_preparation.py`, add a 3-way split helper used when `dataset_name == "paysim"`:
   ```python
   def train_val_test_split(X, y, test_size=0.10, val_size=0.10, seed=2023, stratify=None):
       from sklearn.model_selection import train_test_split as sk
       strat = y if stratify else None
       x_tv, x_te, y_tv, y_te = sk(X, y, test_size=test_size, random_state=seed, stratify=strat)
       val_rel = val_size / (1.0 - test_size)
       strat2 = y_tv if stratify else None
       x_tr, x_va, y_tr, y_va = sk(x_tv, y_tv, test_size=val_rel, random_state=seed, stratify=strat2)
       return x_tr, y_tr, x_va, y_va, x_te, y_te
   ```

2. Extend `load_single_dataset` to return 6-tuple for PaySim (or a dict).
3. In `main.py:63-75` use the val/test as **server-side** tensors:
   ```python
   x_tr, y_tr, x_va, y_va, x_te, y_te = load_single_dataset_3way(...)
   server_valloader = get_dataloader(TensorDataset(torch.from_numpy(x_va), torch.from_numpy(y_va)), "test", cfg.batch_size)
   testloader       = get_dataloader(TensorDataset(torch.from_numpy(x_te), torch.from_numpy(y_te)), "test", cfg.batch_size)
   trainloaders, valloaders, _ = divide_dataset_between_clients(
       TensorDataset(torch.from_numpy(x_tr), torch.from_numpy(y_tr)),
       TensorDataset(torch.from_numpy(x_te), torch.from_numpy(y_te)),  # unused server-side
       batch_size=cfg.batch_size, pool_size=cfg.clients.client_num, val_ratio=0.0,
       non_iid_alpha=cfg.dataset.non_iid.alpha if cfg.dataset.non_iid.enabled else None,
       labels=y_tr,
   )
   ```
4. Centralized eval already runs on `testloader` via `serverside_eval` (`main.py:100-104` → `server.py:350-406`). To add a centralized **val** loader pass it through `functools.partial(serverside_eval, cfg=cfg, testloader=testloader, valloader=server_valloader)` and extend `serverside_eval` to also call `test()` on `valloader`. The `EarlyStop` already keys off the eval loss — if you want it to key off val instead of test, swap which loader is passed.

---

## TASK 5 — HOW TO RUN

### 5.1 — Run the baseline as-is
Per `README.md:111-129`:
```bash
# Activate env (Poetry):
poetry shell                                      # or: source .venv/bin/activate

# Centralized (single dataset):
python -m hfedxgboost.main --config-name Centralized_Baseline dataset=abalone xgboost_params_centralized=abalone_xgboost_centralized

# Centralized (all datasets, paper hyperparams):
python -m hfedxgboost.main --config-name centralized_basline_all_datasets_paper_config

# Federated:
python -m hfedxgboost.main dataset=a9a clients=a9a_5_clients
python -m hfedxgboost.main --multirun clients=a9a_2_clients,a9a_5_clients,a9a_10_clients dataset=a9a
```

### 5.2 — Run with PaySim (after the changes above)
```bash
# Place CSV first:
mkdir -p dataset/paysim
cp /path/to/PS_20174392719_1491204439457_log.csv dataset/paysim/paysim.csv

# Run:
python -m hfedxgboost.main dataset=paysim clients=paysim_5_clients
```

### 5.3 — Common flags / tweaks
- **Number of clients & rounds**: by selecting a `clients/<file>` config — or override:
  ```bash
  python -m hfedxgboost.main dataset=paysim clients=paysim_5_clients clients.client_num=10 clients.num_rounds=50
  ```
- **Dirichlet alpha**:
  ```bash
  python -m hfedxgboost.main dataset=paysim clients=paysim_5_clients dataset.non_iid.enabled=true dataset.non_iid.alpha=0.1
  ```
- **Enable / disable SMOTE**:
  ```bash
  python -m hfedxgboost.main dataset=paysim clients=paysim_5_clients dataset.smote.enabled=true dataset.smote.sampling_strategy=0.2
  python -m hfedxgboost.main dataset=paysim clients=paysim_5_clients dataset.smote.enabled=false
  ```
- **Per-client tree count / max_depth / CNN lr**:
  ```bash
  python -m hfedxgboost.main dataset=paysim clients=paysim_5_clients \
        clients.n_estimators_client=200 clients.xgb.max_depth=6 clients.CNN.lr=0.0001
  ```
- **W&B**: `use_wandb=True wandb.setup.project=paysim_fl`.

### 5.4 — Gotchas observed in the codebase

1. **Hardcoded `n_estimators=100` per round in `FlClient.get_parameters`** (`client.py:189`) — overrides `cfg.n_estimators_client`. If you want each client to train a different number of trees, edit this line.
2. **`batch_size: "whole"`** in `base.yaml` (`base.yaml:11`) makes a **whole-dataset batch**. With 6.3M PaySim rows this will OOM. Override: `batch_size=8192` and also raise `run_experiment.batch_size`. Note the two batch-size fields are different (`cfg.batch_size` for raw data, `cfg.run_experiment.batch_size` for CNN training).
3. **Dataset path hardcoded**: `./dataset/<name>/` from cwd (`dataset_preparation.py:34`). Hydra changes cwd to its run dir, but `download_data` uses a relative path — when you run from the project root it works, but launching from elsewhere creates a new `dataset/` directory in Hydra's output dir. Use `cd .../hfedxgboost && python -m hfedxgboost.main ...`.
4. **`server.max_workers`** in `base.yaml:32` is the **string `"None"`** then compared to `"None"` in `server.py:128` — keep this as a string if you override it.
5. **`fraction_evaluate=0.0`** in `base.yaml:43` ⇒ federated eval is a no-op and all evaluation is centralized via `serverside_eval`. Increase to enable per-client validation.
6. **Labels assumed to be `{0,1}` or `{-1,+1}`**: `modify_labels` (`dataset_preparation.py:244`) flips `-1 → 0` — PaySim labels are already `{0,1}`, so it's harmless but unnecessary; the existing `load_single_dataset` will call it because task is `BINARY`. With your StandardScaler branch you can skip it for `paysim`.
7. **`load_svmlight_file` won't read CSV** — README §"How to add a new dataset" warns this explicitly. You **must** edit `datafiles_fusion`.
8. **Single-tuple test-loader edge case** (`utils.py:298-321`): on round 0 the server passes `client_tree_ensamples` as a single tuple (not a list), so `temp_trees = [tree] * client_num`. After round 1 it becomes a list of one tuple per client. Make sure if you change `client_num` mid-run you don't trip this.
9. **`cfg.client_num` is referenced in `server.py:398`** as `cfg.client_num` (top-level interpolation from `clients.client_num`) — keep the interpolation in `base.yaml:14` intact when you copy.
10. **CNN architecture depends on `n_estimators_client` and `client_num`**: the conv1d kernel + the linear layer's input dim are computed from those values (`models.py:65-73`). Don't mix runs across different `(client_num, n_estimators_client)` pairs with the same `serverside_eval` cache.
11. **`pyproject.toml` pins `torch==2.8.0` and `torchvision==0.23.0`** — `torchvision` is unused in the source; you can drop it if you switch to plain pip and want to slim dependencies.
