"""End-to-end entry point for the FedAvg-LR pipeline.

CLI examples
------------
    python -m models.fedavg_lr.run --scheme iid --num_rounds 50
    python -m models.fedavg_lr.run --scheme dirichlet --alpha 0.5 --oversampling smote
    python -m models.fedavg_lr.run --scheme iid --num_rounds 50 --oversampling adasyn
    python -m models.fedavg_lr.run --scheme iid --num_rounds 50 --oversampling none
"""

from __future__ import annotations

import argparse
import os
from typing import Any, Dict

import flwr as fl
import yaml

from preprocessing.paysim import load_paysim
from preprocessing.oversampling import apply_oversampling_to_all_clients, VALID_METHODS
from partitioning.dirichlet import get_partition

from .client import build_client_fn
from .server import make_server_eval_fn
from .strategy import get_strategy


MODEL_NAME: str = "fedavg_lr"
N_FEATURES: int = 13


def _str2bool(v) -> bool:
    if isinstance(v, bool):
        return v
    s = v.lower()
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
    p.add_argument(
        "--oversampling",
        choices=list(VALID_METHODS),
        default=None,
        help="Local oversampling method applied per-client (smote | adasyn | none).",
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
    """Run the FedAvg-LR pipeline end-to-end. Returns ``(history, state)``."""
    seed = int(cfg["random_seed"])
    scheme = cfg["partition"]["scheme"]
    alpha = cfg["partition"]["alpha"]
    num_clients = int(cfg["num_clients"])
    num_rounds = int(cfg["num_rounds"])

    print(
        f"[run] === {MODEL_NAME} | scheme={scheme} alpha={alpha} "
        f"K={num_clients} R={num_rounds} seed={seed} ==="
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

    clients = apply_oversampling_to_all_clients(
        clients,
        method=str(cfg.get("oversampling", "smote")),
        k_neighbors=int(cfg.get("smote_k_neighbors", 5)),
        sampling_strategy=cfg.get("smote_sampling_strategy", "auto"),
        base_seed=seed,
    )

    wandb_run = None
    if bool(cfg.get("use_wandb", False)):
        import wandb

        run_name = f"{MODEL_NAME}_{scheme}_alpha{alpha}_seed{seed}"
        wandb_run = wandb.init(
            project=cfg.get("wandb_project", "hfedxgboost-paysim"),
            name=run_name,
            config=cfg,
        )

    server_eval_fn, eval_state = make_server_eval_fn(
        cfg=cfg,
        x_val=x_val,
        y_val=y_val,
        x_test=x_test,
        y_test=y_test,
        wandb_run=wandb_run,
    )

    strategy = get_strategy(cfg, n_features=N_FEATURES, server_eval_fn=server_eval_fn)
    client_fn = build_client_fn(clients, cfg, seed=seed)

    print(f"[run] FL starting: {num_rounds} rounds, {num_clients} clients")
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": 0},
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

    if wandb_run is not None:
        wandb_run.summary["best_val_auprc"] = eval_state["best_val_auprc"]
        wandb_run.summary["best_round"] = eval_state["best_round"]
        if eval_state.get("final_test"):
            wandb_run.summary.update(eval_state["final_test"])
        wandb_run.finish()

    return history, eval_state


def main():
    cfg = _load_base_cfg()
    args = _parse_args()
    cfg = _apply_cli_overrides(cfg, args)
    print(f"[run] resolved config: {cfg}")
    run(cfg)


if __name__ == "__main__":
    main()
