"""End-to-end entry point for the GBM best-model-selection pipeline.

CLI examples
------------
    python -m models.gbm_bestmodel.run --scheme iid --num_rounds 50
    python -m models.gbm_bestmodel.run --scheme dirichlet --alpha 0.5 --oversampling smote
    python -m models.gbm_bestmodel.run --scheme iid --num_rounds 50 --oversampling adasyn
    python -m models.gbm_bestmodel.run --scheme iid --num_rounds 50 --oversampling none

For a fast plumbing-only smoke test, override ``--max_iter`` (the yaml
default is 100; HistGBM at max_iter=100 fits in ~10-30 s per client on
PaySim, so the full config is usually fine for smoke tests too)::

    python -m models.gbm_bestmodel.run --scheme iid --num_rounds 3 \
        --num_clients 2 --oversampling none --max_iter 10
"""

from __future__ import annotations

import argparse
import os
import time
from typing import Any, Dict

import flwr as fl
import yaml

from evaluation.results_writer import build_run_name, write_fl_results
from preprocessing.loader import DATASETS, load_dataset
from preprocessing.oversampling import apply_oversampling_to_all_clients, VALID_METHODS
from partitioning.dirichlet import get_partition

from .client import build_client_fn
from .strategy import BestModelSelection


MODEL_NAME: str = "gbm"


def _str2bool(v) -> bool:
    if isinstance(v, bool):
        return v
    s = v.lower()
    if s in ("yes", "true", "t", "y", "1"):
        return True
    if s in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError(f"boolean expected, got {v!r}")


def _parse_sampling_strategy(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return s


def _load_base_cfg() -> Dict[str, Any]:
    path = os.path.join(os.path.dirname(__file__), "conf", "base.yaml")
    with open(path) as f:
        return yaml.safe_load(f)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=f"FL run for {MODEL_NAME}")
    p.add_argument("--dataset", choices=list(DATASETS), default=None)
    p.add_argument("--scheme", choices=["iid", "dirichlet"], default=None)
    p.add_argument("--alpha", type=float, default=None)
    p.add_argument("--num_rounds", type=int, default=None)
    p.add_argument("--num_clients", type=int, default=None)
    p.add_argument(
        "--oversampling",
        choices=list(VALID_METHODS),
        default=None,
        help="Local oversampling method applied per-client (smote | adasyn | none).",
    )
    p.add_argument("--random_seed", type=int, default=None)
    p.add_argument("--use_wandb", type=_str2bool, default=None)
    p.add_argument("--wandb_project", type=str, default=None)
    p.add_argument("--max_iter", type=int, default=None,
                   help="override gbm_params.max_iter (smoke-test knob)")
    p.add_argument("--max_depth", type=int, default=None,
                   help="override gbm_params.max_depth")
    p.add_argument(
        "--sampling_strategy",
        type=_parse_sampling_strategy,
        default=None,
        help=(
            "Per-client imblearn sampling_strategy. 'auto' = 1:1 fraud:non-fraud. "
            "A float sets the post-resample minority/majority ratio "
            "(e.g. 0.01 for 1:100 fraud:non-fraud)."
        ),
    )
    p.add_argument("--learning_rate", type=float, default=None,
                   help="override gbm_params.learning_rate")
    return p.parse_args()


def _apply_cli_overrides(cfg: dict, args: argparse.Namespace) -> dict:
    for k, v in vars(args).items():
        if v is None:
            continue
        if k in ("scheme", "alpha"):
            cfg["partition"][k] = v
        elif k in ("max_iter", "max_depth", "learning_rate"):
            cfg.setdefault("gbm_params", {})[k] = v
        else:
            cfg[k] = v
    return cfg


def run(cfg: dict):
    """Run the GBM best-model FL pipeline. Returns ``(history, state)``."""
    t_start = time.time()
    seed = int(cfg["random_seed"])
    dataset = str(cfg.get("dataset", "paysim")).lower()
    scheme = cfg["partition"]["scheme"]
    alpha = cfg["partition"]["alpha"]
    oversampling = str(cfg.get("oversampling", "smote")).lower()
    num_clients = int(cfg["num_clients"])
    num_rounds = int(cfg["num_rounds"])

    print(
        f"[run] === {MODEL_NAME} | dataset={dataset} scheme={scheme} alpha={alpha} "
        f"K={num_clients} R={num_rounds} oversampling={oversampling} "
        f"seed={seed} ==="
    )

    data = load_dataset(dataset, random_state=seed)
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
        method=oversampling,
        k_neighbors=int(cfg.get("smote_k_neighbors", 5)),
        sampling_strategy=cfg.get("sampling_strategy", "auto"),
        base_seed=seed,
    )

    run_name = build_run_name(MODEL_NAME, scheme, alpha, oversampling, seed)
    wandb_run = None
    if bool(cfg.get("use_wandb", False)):
        import wandb

        wandb_run = wandb.init(
            project=cfg.get("wandb_project", "fraud-fl-TA"),
            name=run_name,
            config=cfg,
        )

    strategy = BestModelSelection(
        cfg=cfg,
        x_val=x_val,
        y_val=y_val,
        x_test=x_test,
        y_test=y_test,
        wandb_run=wandb_run,
    )
    client_fn = build_client_fn(clients, cfg, seed=seed)

    print(f"[run] FL starting: {num_rounds} rounds, {num_clients} clients")
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": 0},
    )

    state = strategy.state
    print(
        f"[run] best val_auprc: {state['best_val_auprc']:.4f} "
        f"at round {state['best_round']} "
        f"(selected client {state['best_client_id']})"
    )
    if state.get("final_test"):
        ft = state["final_test"]
        print(
            f"[run] final test: auprc={ft['test_auprc']:.4f} | "
            f"f1={ft['test_f1']:.4f} | precision={ft['test_precision']:.4f} | "
            f"recall={ft['test_recall']:.4f}"
        )

    if state.get("client_selections"):
        trace = " ".join(
            f"r{s['round']}:c{s['selected_client_id']}"
            for s in state["client_selections"]
        )
        print(f"[run] selection trace: {trace}")

    duration_seconds = time.time() - t_start
    write_fl_results(
        model=MODEL_NAME,
        dataset=dataset,
        scheme=scheme,
        alpha=alpha,
        oversampling=oversampling,
        seed=seed,
        num_rounds=num_rounds,
        num_clients=num_clients,
        best_round=state.get("best_round", -1),
        best_val_auprc=state.get("best_val_auprc", -1.0),
        history=state.get("history") or [],
        final_test=state.get("final_test"),
        duration_seconds=duration_seconds,
    )

    if wandb_run is not None:
        wandb_run.summary["best_val_auprc"] = state["best_val_auprc"]
        wandb_run.summary["best_round"] = state["best_round"]
        wandb_run.summary["best_client_id"] = state["best_client_id"]
        wandb_run.summary["duration_seconds"] = duration_seconds
        if state.get("final_test"):
            wandb_run.summary.update(state["final_test"])
        wandb_run.finish()

    return history, state


def main():
    cfg = _load_base_cfg()
    args = _parse_args()
    cfg = _apply_cli_overrides(cfg, args)
    print(f"[run] resolved config: {cfg}")
    run(cfg)


if __name__ == "__main__":
    main()
