"""FedAvg strategy wrapper for the FedAvg-SVM pipeline."""

from __future__ import annotations

from typing import Callable, Dict, List, Tuple

import flwr as fl
import numpy as np


def weighted_average(metrics: List[Tuple[int, Dict[str, float]]]) -> Dict[str, float]:
    """Aggregate client-reported metrics weighted by sample count."""
    if not metrics:
        return {}
    total = sum(int(n) for n, _ in metrics)
    if total == 0:
        return {}
    keys = {
        k for _, m in metrics for k, v in m.items() if isinstance(v, (int, float))
    }
    out: Dict[str, float] = {}
    for k in keys:
        out[k] = sum(int(n) * float(m.get(k, 0.0)) for n, m in metrics) / total
    return out


def get_strategy(
    cfg: dict, n_features: int, server_eval_fn: Callable
) -> fl.server.strategy.FedAvg:
    """Build a vanilla FedAvg strategy with centralized server-side eval."""
    init_coef = np.zeros((1, n_features), dtype=np.float32)
    init_intercept = np.zeros(1, dtype=np.float32)
    initial_parameters = fl.common.ndarrays_to_parameters([init_coef, init_intercept])

    num_clients = int(cfg["num_clients"])
    local_epochs = int(cfg["local_epochs"])

    return fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=0.0,
        min_fit_clients=num_clients,
        min_evaluate_clients=num_clients,
        min_available_clients=num_clients,
        evaluate_fn=server_eval_fn,
        initial_parameters=initial_parameters,
        on_fit_config_fn=lambda rnd: {"local_epochs": local_epochs, "round": rnd},
        fit_metrics_aggregation_fn=weighted_average,
        evaluate_metrics_aggregation_fn=weighted_average,
    )
