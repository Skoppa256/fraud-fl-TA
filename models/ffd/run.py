"""FFD (Federated learning for Fraud Detection) — Yang et al. (2019).

Adapted for PaySim (13 features, K=5 clients).

Key differences from FedAvg-LR/SVM
----------------------------------
- 1D CNN model (PyTorch, not sklearn).
- :class:`AccuracyWeightedFedAvg` aggregation (weights by data size × local AUPRC).
- SGD optimizer, ``local_epochs=5``, ``batch_size=80``, ``lr=0.01``.
- Oversampling (SMOTE/ADASYN) is applied **per-round inside the client**
  rather than once before training, following the paper's procedure step 3.

CLI examples
------------
    python -m models.ffd.run --scheme iid --num_rounds 50
    python -m models.ffd.run --scheme dirichlet --alpha 0.5 --oversampling smote
"""

from __future__ import annotations

import argparse
import os
import time
from typing import Any, Dict

import flwr as fl
import torch
import yaml

from evaluation.results_writer import build_run_name, write_fl_results
from partitioning.dirichlet import get_partition
from preprocessing.paysim import load_paysim

from .client import build_client_fn
from .model import FFDModel
from .server import make_server_eval_fn
from .strategy import get_strategy


MODEL_NAME: str = "ffd"
N_FEATURES: int = 13
VALID_OVERSAMPLING = ("none", "smote", "adasyn")


def _str2bool(v) -> bool:
    if isinstance(v, bool):
        return v
    s = str(v).lower()
    if s in ("yes", "true", "t", "y", "1"):
        return True
    if s in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError(f"boolean expected, got {v!r}")


def _load_base_cfg() -> Dict[str, Any]:
    path = os.path.join(os.path.dirname(__file__), "conf", "base.yaml")
    with open(path) as f:
        return yaml.safe_load(f)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=f"FL run for {MODEL_NAME}")
    p.add_argument("--scheme", choices=["iid", "dirichlet"], default=None)
    p.add_argument("--alpha", type=float, default=None)
    p.add_argument("--num_rounds", type=int, default=None)
    p.add_argument("--num_clients", type=int, default=None)
    p.add_argument("--local_epochs", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument(
        "--oversampling",
        choices=list(VALID_OVERSAMPLING),
        default=None,
        help="Local oversampling method applied per-client (none | smote | adasyn).",
    )
    p.add_argument("--random_seed", type=int, default=None)
    p.add_argument("--use_wandb", type=_str2bool, default=None)
    p.add_argument("--wandb_project", type=str, default=None)
    return p.parse_args()


def _apply_cli_overrides(cfg: dict, args: argparse.Namespace) -> dict:
    for k, v in vars(args).items():
        if v is None:
            continue
        if k in ("scheme", "alpha"):
            cfg["partition"][k] = v
        else:
            cfg[k] = v
    return cfg


def run(cfg: dict):
    """Run the FFD pipeline end-to-end. Returns ``(history, state)``."""
    t_start = time.time()
    seed = int(cfg["random_seed"])
    scheme = cfg["partition"]["scheme"]
    alpha = cfg["partition"]["alpha"]
    num_clients = int(cfg["num_clients"])
    num_rounds = int(cfg["num_rounds"])
    oversampling = str(cfg.get("oversampling", "smote")).lower()

    print(
        f"[run] === {MODEL_NAME} | scheme={scheme} alpha={alpha} "
        f"K={num_clients} R={num_rounds} oversampling={oversampling} "
        f"seed={seed} ==="
    )

    data = load_paysim(random_state=seed)
    x_train, y_train = data["x_train"], data["y_train"]
    x_val, y_val = data["x_val"], data["y_val"]
    x_test, y_test = data["x_test"], data["y_test"]

    clients = get_partition(
        x_train,
        y_train,
        scheme=scheme,
        alpha=alpha,
        num_clients=num_clients,
        random_state=seed,
    )

    # NB: oversampling is NOT pre-applied here. FFD applies it per-round
    # inside FFDClient.fit() per Yang et al. (2019) procedure step 3.

    run_name = build_run_name(MODEL_NAME, scheme, alpha, oversampling, seed)
    wandb_run = None
    if bool(cfg.get("use_wandb", False)):
        import wandb

        wandb_run = wandb.init(
            project=cfg.get("wandb_project", "fraud-fl-TA"),
            name=run_name,
            config=cfg,
        )

    server_eval_fn, eval_state = make_server_eval_fn(
        cfg=cfg,
        input_dim=N_FEATURES,
        x_val=x_val,
        y_val=y_val,
        x_test=x_test,
        y_test=y_test,
        wandb_run=wandb_run,
    )

    # Initial parameters: a fresh FFDModel's weights.
    init_model = FFDModel(input_dim=N_FEATURES)
    initial_parameters = fl.common.ndarrays_to_parameters(init_model.get_weights())

    strategy = get_strategy(
        cfg, initial_parameters=initial_parameters, server_eval_fn=server_eval_fn
    )
    client_fn = build_client_fn(clients, cfg, seed=seed)

    num_gpus_per_client = float(
        cfg.get("num_gpus_per_client", 0.2 if torch.cuda.is_available() else 0.0)
    )
    num_cpus_per_client = int(cfg.get("num_cpus_per_client", 1))
    client_resources = {"num_cpus": num_cpus_per_client, "num_gpus": num_gpus_per_client}

    print(
        f"[run] FL starting: {num_rounds} rounds, {num_clients} clients "
        f"(client_resources={client_resources})"
    )
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
        client_resources=client_resources,
    )

    print(
        f"[run] best val_auprc: {eval_state['best_val_auprc']:.4f} "
        f"at round {eval_state['best_round']}"
    )
    if eval_state.get("final_test"):
        ft = eval_state["final_test"]
        print(
            f"[run] final test: auprc={ft['test_auprc']:.4f} | "
            f"f1={ft['test_f1']:.4f} | precision={ft['test_precision']:.4f} | "
            f"recall={ft['test_recall']:.4f}"
        )

    duration_seconds = time.time() - t_start
    write_fl_results(
        model=MODEL_NAME,
        scheme=scheme,
        alpha=alpha,
        oversampling=oversampling,
        seed=seed,
        num_rounds=num_rounds,
        num_clients=num_clients,
        best_round=eval_state.get("best_round", -1),
        best_val_auprc=eval_state.get("best_val_auprc", -1.0),
        history=eval_state.get("history") or [],
        final_test=eval_state.get("final_test"),
        duration_seconds=duration_seconds,
    )

    if wandb_run is not None:
        wandb_run.summary["best_val_auprc"] = eval_state["best_val_auprc"]
        wandb_run.summary["best_round"] = eval_state["best_round"]
        wandb_run.summary["duration_seconds"] = duration_seconds
        if eval_state.get("final_test"):
            wandb_run.summary.update(eval_state["final_test"])
        wandb_run.finish()

    return history, eval_state


def main() -> None:
    cfg = _load_base_cfg()
    args = _parse_args()
    cfg = _apply_cli_overrides(cfg, args)
    print(f"[run] resolved config: {cfg}")
    run(cfg)


if __name__ == "__main__":
    main()
