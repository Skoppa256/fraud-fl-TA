"""Create and connect the building blocks for your experiments; start the simulation.

It includes processioning the dataset, instantiate strategy, specify how the global
model is going to be evaluated, etc. At the end, this script saves the results.
"""

import functools
import time
from typing import Any, Dict, List, Optional, Union

import flwr as fl
import hydra
import torch
import wandb
from flwr.common import Scalar
from flwr.server.app import ServerConfig
from flwr.server.client_manager import SimpleClientManager
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import TensorDataset

from evaluation.results_writer import build_run_name, write_fl_results
from hfedxgboost.client import FlClient
from hfedxgboost.dataset import divide_dataset_between_clients, load_single_dataset
from hfedxgboost.server import FlServer, serverside_eval
from hfedxgboost.utils import (
    CentralizedResultsWriter,
    EarlyStop,
    ResultsWriter,
    create_res_csv,
    local_clients_performance,
    run_centralized,
)


CANONICAL_MODEL: str = "fedxgbllr"


def _build_history_state(
    history,
    num_rounds: int,
) -> Dict[str, Any]:
    """Convert Flower ``History`` into the schema the shared CSV writer wants.

    Pulls per-round val_* metrics out of ``history.metrics_centralized``,
    finds the best round by ``val_auprc``, and pulls the final round's
    ``test_*`` metrics for the summary row.
    """
    metrics_cen = getattr(history, "metrics_centralized", {}) or {}
    rounds_set = set()
    for series in metrics_cen.values():
        for r, _ in series:
            rounds_set.add(int(r))
    rounds_sorted = sorted(rounds_set)

    def _series_to_map(key: str) -> Dict[int, float]:
        return {int(r): float(v) for r, v in metrics_cen.get(key, [])}

    val_auprc_map = _series_to_map("val_auprc")
    val_f1_map = _series_to_map("val_f1")
    val_precision_map = _series_to_map("val_precision")
    val_recall_map = _series_to_map("val_recall")
    test_auprc_map = _series_to_map("test_auprc")
    test_f1_map = _series_to_map("test_f1")
    test_precision_map = _series_to_map("test_precision")
    test_recall_map = _series_to_map("test_recall")

    hist_rows: List[Dict[str, Any]] = []
    for r in rounds_sorted:
        hist_rows.append(
            {
                "round": r,
                "val_auprc": val_auprc_map.get(r, ""),
                "val_f1": val_f1_map.get(r, ""),
                "val_precision": val_precision_map.get(r, ""),
                "val_recall": val_recall_map.get(r, ""),
            }
        )

    best_round = -1
    best_val_auprc = -1.0
    for r, v in val_auprc_map.items():
        if v > best_val_auprc:
            best_val_auprc = v
            best_round = r
    if best_round == -1 and rounds_sorted:
        # No val_auprc series (e.g. dataset != paysim). Fall back to test_auprc.
        for r, v in test_auprc_map.items():
            if v > best_val_auprc:
                best_val_auprc = v
                best_round = r

    final_round = max(rounds_sorted) if rounds_sorted else num_rounds
    final_test: Optional[Dict[str, float]] = None
    if final_round in test_auprc_map:
        final_test = {
            "test_auprc": test_auprc_map[final_round],
            "test_f1": test_f1_map.get(final_round, 0.0),
            "test_precision": test_precision_map.get(final_round, 0.0),
            "test_recall": test_recall_map.get(final_round, 0.0),
        }

    return {
        "best_round": best_round,
        "best_val_auprc": best_val_auprc,
        "history": hist_rows,
        "final_test": final_test,
    }


@hydra.main(config_path="conf", config_name="base", version_base=None)
def main(cfg: DictConfig) -> None:
    """Run the baseline.

    Parameters
    ----------
    cfg : DictConfig
        An omegaconf object that stores the hydra config.
    """
    # 1. Print parsed config
    print(OmegaConf.to_yaml(cfg))
    writer: Union[ResultsWriter, CentralizedResultsWriter]
    if cfg.centralized:
        if cfg.dataset.dataset_name == "all":
            run_centralized(cfg, dataset_name=cfg.dataset.dataset_name)
        else:
            writer = CentralizedResultsWriter(cfg)
            create_res_csv("results_centralized.csv", writer.fields)
            writer.write_res(
                "results_centralized.csv",
                run_centralized(cfg, dataset_name=cfg.dataset.dataset_name)[0],
                run_centralized(cfg, dataset_name=cfg.dataset.dataset_name)[1],
            )
    else:
        t_start = time.time()
        non_iid_cfg = cfg.dataset.get("non_iid", {})
        non_iid_alpha = (
            non_iid_cfg.get("alpha", None)
            if non_iid_cfg.get("enabled", False)
            else None
        )
        oversampling_method = OmegaConf.select(
            cfg, "dataset.oversampling.method", default="none"
        )
        scheme = "dirichlet" if non_iid_cfg.get("enabled", False) else "iid"
        random_seed = int(OmegaConf.select(cfg, "random_seed", default=42))

        run_name = build_run_name(
            CANONICAL_MODEL,
            scheme,
            non_iid_alpha,
            oversampling_method,
            random_seed,
        )

        if cfg.use_wandb:
            wandb.init(
                **cfg.wandb.setup,
                group=f"{cfg.dataset.dataset_name}",
                name=run_name,
            )
            wandb.config.update(
                {
                    "client_num": cfg.client_num,
                    "num_rounds": cfg.run_experiment.num_rounds,
                    "n_estimators_client": cfg.n_estimators_client,
                    "dataset": cfg.dataset.dataset_name,
                    "non_iid_alpha": non_iid_alpha if non_iid_alpha else "IID",
                    "oversampling": oversampling_method,
                    "random_seed": random_seed,
                    "xgb_max_depth": cfg.XGBoost.max_depth,
                    "cnn_lr": cfg.clients.CNN.lr,
                },
                allow_val_change=True,
            )

        print("Dataset Name", cfg.dataset.dataset_name)
        early_stopper = EarlyStop(cfg)
        x_train, y_train, x_test, y_test = load_single_dataset(
            cfg.dataset.task.task_type,
            cfg.dataset.dataset_name,
            train_ratio=cfg.dataset.train_ratio,
        )

        trainloaders, valloaders, testloader = divide_dataset_between_clients(
            TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
            TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y_test)),
            batch_size=cfg.batch_size,
            pool_size=cfg.clients.client_num,
            val_ratio=cfg.val_ratio,
            non_iid_alpha=non_iid_alpha,
            random_state=random_seed,
        )
        print(
            f"Data partitioned across {cfg.clients.client_num} clients"
            f" and {cfg.val_ratio} of local dataset reserved for validation."
        )
        if cfg.show_each_client_performance_on_its_local_data:
            local_clients_performance(
                cfg, trainloaders, x_test, y_test, cfg.dataset.task.task_type
            )

        # Configure the strategy
        def fit_config(server_round: int) -> Dict[str, Scalar]:
            print(f"Configuring round {server_round}")
            return {
                "num_iterations": cfg.run_experiment.fit_config.num_iterations,
                "batch_size": cfg.run_experiment.batch_size,
            }

        # FedXgbNnAvg
        strategy = instantiate(
            cfg.strategy,
            on_fit_config_fn=fit_config,
            on_evaluate_config_fn=(
                lambda r: {"batch_size": cfg.run_experiment.batch_size}
            ),
            evaluate_fn=functools.partial(
                serverside_eval,
                cfg=cfg,
                testloader=testloader,
            ),
        )

        print(
            f"FL experiment configured for {cfg.run_experiment.num_rounds} rounds with",
            f"{cfg.clients.client_num} client in the pool.",
        )

        def client_fn(cid: str) -> fl.client.Client:
            """Create a federated learning client."""
            return FlClient(cfg, trainloaders[int(cid)], valloaders[int(cid)], cid)

        # Ray reserves a GPU fraction per virtual client (client_resources.
        # num_gpus). On a CPU-only host (e.g. local macOS dev) there is no GPU
        # to reserve, so the ActorPool comes up empty and the simulation aborts.
        # Fall back to CPU-only placement when CUDA is unavailable; the GPU
        # config is preserved on machines that have one (e.g. Kaggle).
        if not torch.cuda.is_available() and cfg.client_resources.num_gpus:
            print(
                "[main] no CUDA device found — forcing client_resources.num_gpus=0 "
                f"(was {cfg.client_resources.num_gpus})"
            )
            cfg.client_resources.num_gpus = 0

        # Start the simulation
        history = fl.simulation.start_simulation(
            client_fn=client_fn,
            server=FlServer(
                cfg=cfg,
                client_manager=SimpleClientManager(),
                early_stopper=early_stopper,
                strategy=strategy,
            ),
            num_clients=cfg.clients.client_num,
            client_resources=cfg.client_resources,
            config=ServerConfig(num_rounds=cfg.run_experiment.num_rounds),
            strategy=strategy,
            ray_init_args={"num_gpus": torch.cuda.device_count()},
        )

        print(history)
        writer = ResultsWriter(cfg)
        print(
            "Best Result",
            writer.extract_best_res(history)[0],
            "best_res_round",
            writer.extract_best_res(history)[1],
        )
        create_res_csv("results.csv", writer.fields)
        writer.write_res("results.csv")

        # Shared structured CSV — same schema as the other models.
        duration_seconds = time.time() - t_start
        state = _build_history_state(
            history, num_rounds=int(cfg.run_experiment.num_rounds)
        )
        write_fl_results(
            model=CANONICAL_MODEL,
            dataset=str(cfg.dataset.dataset_name),
            scheme=scheme,
            alpha=non_iid_alpha,
            oversampling=str(oversampling_method),
            seed=random_seed,
            num_rounds=int(cfg.run_experiment.num_rounds),
            num_clients=int(cfg.clients.client_num),
            best_round=state["best_round"],
            best_val_auprc=state["best_val_auprc"],
            history=state["history"],
            final_test=state["final_test"],
            duration_seconds=duration_seconds,
        )

        if cfg.use_wandb:
            wandb.summary["best_val_auprc"] = state["best_val_auprc"]
            wandb.summary["best_round"] = state["best_round"]
            wandb.summary["duration_seconds"] = duration_seconds
            if state["final_test"]:
                wandb.summary.update(state["final_test"])
            wandb.finish()


if __name__ == "__main__":
    main()
