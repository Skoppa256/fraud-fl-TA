"""Create and connect the building blocks for your experiments; start the simulation.

It includes processioning the dataset, instantiate strategy, specify how the global
model is going to be evaluated, etc. At the end, this script saves the results.
"""

import functools
import sys
import time
from typing import Any, Dict, List, NoReturn, Optional, Union

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

import numpy as np

from evaluation.results_writer import build_run_name, write_fl_results
from evaluation.metrics import baseline_auprc
from evaluation import model_persistence
from experiments import resources
from experiments import data_cache
from hfedxgboost.client import FlClient
from hfedxgboost import dataset_preparation as _dp
from hfedxgboost.dataset import (
    divide_dataset_between_clients,
    get_dataloader,
    load_single_dataset,
)
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
    test_brier_map = _series_to_map("test_brier")
    test_cal_intercept_map = _series_to_map("test_cal_intercept")
    test_cal_slope_map = _series_to_map("test_cal_slope")
    threshold_map = _series_to_map("threshold")

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
        # Calibration + threshold (present only if the server surfaced them).
        if final_round in test_brier_map:
            final_test["test_brier"] = test_brier_map[final_round]
        if final_round in test_cal_intercept_map:
            final_test["test_cal_intercept"] = test_cal_intercept_map[final_round]
        if final_round in test_cal_slope_map:
            final_test["test_cal_slope"] = test_cal_slope_map[final_round]
        if final_round in threshold_map:
            final_test["threshold"] = threshold_map[final_round]

    return {
        "best_round": best_round,
        "best_val_auprc": best_val_auprc,
        "history": hist_rows,
        "final_test": final_test,
    }


def _abort_no_successful_rounds(cfg: DictConfig, reason: str) -> NoReturn:
    """Fail a run that produced no successfully-aggregated rounds: log the
    reason, write NO results row / artifact, close wandb as failed, exit non-zero.

    Guards against a zero-round run entering the results table as legitimate data
    (Bug A) and against a silent Ray OOM masquerading as success (Bug B)."""
    print(f"[main] RUN FAILED — {reason}. Writing NO results row and exiting non-zero.")
    try:
        if bool(OmegaConf.select(cfg, "use_wandb", default=False)):
            wandb.finish(exit_code=1)
    except Exception:  # noqa: BLE001 — never mask the failure with a wandb error
        pass
    sys.exit(1)


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
        # Route data + partition through the shared content-addressed cache so
        # FedXGBllr consumes byte-identical data/partition as every other model
        # and emits matching data_hash/partition_hash. This also FIXES the legacy
        # IID divergence: dataset.py partitioned IID via torch
        # random_split(manual_seed(0)) — a different, seed-independent split from
        # the shared numpy partitioner every other model uses. Non-IID already
        # used the shared partitioner and is unchanged. The two-stage training
        # (tree ensemble aggregation → CNN) below is untouched.
        _dataset_name = str(cfg.dataset.dataset_name)
        _data, data_hash = data_cache.get_preprocessed(_dataset_name, random_seed)
        x_test = np.asarray(_data["x_test"], dtype=np.float32)
        y_test = np.asarray(_data["y_test"], dtype=np.float32)
        # serverside_eval reads the held-out val split via get_val_cache; populate
        # it explicitly since we bypass the download_data side effect.
        _dp.set_val_cache(
            _dataset_name,
            (np.asarray(_data["x_val"], dtype=np.float32),
             np.asarray(_data["y_val"], dtype=np.float32)),
        )
        _scheme_cache = "iid" if non_iid_alpha is None else "dirichlet"
        clients, partition_hash = data_cache.get_partition_clients(
            _dataset_name, random_seed, _scheme_cache, non_iid_alpha,
            int(cfg.clients.client_num),
        )
        n_below_floor = sum(1 for c in clients if int(c["n_fraud"]) < 6)
        # y as float32 for BCE (matches the previous load path). val_ratio is 0.0
        # in this study (central val used instead), so no inner client val split.
        assert float(cfg.val_ratio) == 0.0, (
            "cache-routed FedXGBllr assumes val_ratio=0.0 (central val); "
            f"got {cfg.val_ratio}"
        )
        client_datasets = [
            TensorDataset(
                torch.from_numpy(np.asarray(c["x"], dtype=np.float32)),
                torch.from_numpy(np.asarray(c["y"], dtype=np.float32)),
            )
            for c in clients
        ]
        trainloaders = [get_dataloader(ds, "train", cfg.batch_size) for ds in client_datasets]
        valloaders = [None] * len(client_datasets)
        testloader = get_dataloader(
            TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y_test)),
            "test", cfg.batch_size,
        )
        print(
            f"Data partitioned across {cfg.clients.client_num} clients"
            f" via shared cache (scheme={_scheme_cache}); central val used."
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
        _final_capture: Dict[str, Any] = {}
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
                capture=_final_capture,
            ),
        )

        print(
            f"FL experiment configured for {cfg.run_experiment.num_rounds} rounds with",
            f"{cfg.clients.client_num} client in the pool.",
        )

        def client_fn(cid: str) -> fl.client.Client:
            """Create a federated learning client."""
            return FlClient(cfg, trainloaders[int(cid)], valloaders[int(cid)], cid)

        # Ray CPU/GPU allocation from the central config
        # (experiments/sweep_resources.yaml) — the single source of truth, not the
        # Hydra config. gpu_available=False (no CUDA) zeroes the GPU request so the
        # ActorPool is not empty on a CPU-only host; on a GPU box it is a no-op.
        _gpu = torch.cuda.is_available()
        _res = resources.for_model("fedxgbllr", gpu_available=_gpu)
        resources.pin_threads()
        client_resources = {"num_cpus": _res["num_cpus"], "num_gpus": _res["num_gpus"]}
        print(f"[main] client_resources={client_resources} (from sweep_resources.yaml)")

        # Start the simulation
        # A total client failure (e.g. every actor OOM-killed by Ray) can either
        # propagate out of start_simulation OR be swallowed, leaving an empty
        # History. BOTH must become a non-zero-exit / write-nothing failure —
        # never a NaN results row that looks legitimate. The try/except handles
        # the propagated case; the rounds_completed==0 guard below handles the
        # swallowed case.
        try:
            history = fl.simulation.start_simulation(
                client_fn=client_fn,
                server=FlServer(
                    cfg=cfg,
                    client_manager=SimpleClientManager(),
                    early_stopper=early_stopper,
                    strategy=strategy,
                ),
                num_clients=cfg.clients.client_num,
                client_resources=client_resources,
                config=ServerConfig(num_rounds=cfg.run_experiment.num_rounds),
                strategy=strategy,
                ray_init_args={
                    "num_gpus": torch.cuda.device_count(),
                    "object_store_memory": resources.object_store_memory(),
                },
            )
        except Exception as exc:  # noqa: BLE001 — any sim abort is a run failure
            _abort_no_successful_rounds(
                cfg, reason=f"simulation aborted: {type(exc).__name__}: {exc}"
            )

        print(history)

        # Guard FIRST, before ANY results are written (upstream results.csv or the
        # shared CSV): a run with zero successfully-aggregated rounds produced no
        # model and no real metrics — writing a NaN row would poison the results
        # table (and look legitimate if persistence happened to succeed). Fail
        # loudly, write nothing, exit non-zero. Common cause here: Ray actor OOM.
        duration_seconds = time.time() - t_start
        state = _build_history_state(
            history, num_rounds=int(cfg.run_experiment.num_rounds)
        )
        if len(state.get("history") or []) == 0:
            _abort_no_successful_rounds(
                cfg,
                reason=(
                    f"0/{int(cfg.run_experiment.num_rounds)} rounds aggregated "
                    "successfully (no central-eval history) — every client likely "
                    "OOM-killed; check log for ray.exceptions.OutOfMemoryError"
                ),
            )

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
            data_hash=data_hash,
            partition_hash=partition_hash,
            rounds_completed=len(state.get("history") or []),
            n_clients_below_smote_floor=n_below_floor,
            baseline_auprc=baseline_auprc(y_test),
        )
        # Persist the two-stage final global model: per-client boosters + CNN.
        if _final_capture.get("cnn") is not None and _final_capture.get("trees") is not None:
            _trees_raw = _final_capture["trees"]
            _trees = [t[0] if isinstance(t, (tuple, list)) else t for t in _trees_raw]
            _ft = state.get("final_test") or {}
            _run_name = build_run_name(
                CANONICAL_MODEL, scheme, non_iid_alpha, str(oversampling_method), random_seed)
            model_persistence.persist_run(
                "fedxgbllr", dataset=_dataset_name, run_name=_run_name,
                scaler=_data.get("scaler"), feature_names=_data.get("feature_names", []),
                data_hash=data_hash, partition_hash=partition_hash,
                threshold=_ft.get("threshold"),
                fedxgbllr_trees=_trees, fedxgbllr_cnn=_final_capture["cnn"],
                arch_config={"n_channel": 64,
                             "n_estimators_client": int(cfg.n_estimators_client),
                             "client_num": int(cfg.clients.client_num)},
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
