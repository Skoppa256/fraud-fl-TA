"""AccuracyWeightedFedAvg for BertFraud.

Same aggregation formula as FFD (Yang et al., 2019):

    w_{t+1} = Σ_c (n_c/n * α_c * w_c) / Σ_c (n_c/n * α_c)

where α_c = local AUPRC reported by client c. Falls back to standard
FedAvg (weight by n_c/n only) when all α_c == 0 (cold-start round 0).
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import flwr as fl
import numpy as np


def weighted_average(
    metrics: List[Tuple[int, Dict[str, float]]],
) -> Dict[str, float]:
    if not metrics:
        return {}
    total = sum(int(n) for n, _ in metrics)
    if total == 0:
        return {}
    keys = {
        k for _, m in metrics for k, v in m.items() if isinstance(v, (int, float))
    }
    return {
        k: sum(int(n) * float(m.get(k, 0.0)) for n, m in metrics) / total
        for k in keys
    }


class AccuracyWeightedFedAvg(fl.server.strategy.FedAvg):
    """FedAvg weighted by data size × local AUPRC (same as FFD strategy)."""

    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return None, {}

        weights_results: List[Tuple[List[np.ndarray], int, float]] = []
        for _, fit_res in results:
            params = fl.common.parameters_to_ndarrays(fit_res.parameters)
            n_c = int(fit_res.num_examples)
            alpha_c = float(fit_res.metrics.get("local_auprc", 0.0))
            weights_results.append((params, n_c, alpha_c))

        n_total = sum(n_c for _, n_c, _ in weights_results)
        if n_total == 0:
            return None, {}

        combined = [(n_c / n_total) * alpha_c for _, n_c, alpha_c in weights_results]
        weight_sum = sum(combined)

        fallback = False
        if weight_sum == 0:
            fallback = True
            combined = [n_c / n_total for _, n_c, _ in weights_results]
            weight_sum = sum(combined)

        normalized = [w / weight_sum for w in combined]

        n_layers = len(weights_results[0][0])
        aggregated: List[np.ndarray] = []
        for i in range(n_layers):
            layer = sum(
                norm * params[i]
                for norm, (params, _, _) in zip(normalized, weights_results)
            )
            aggregated.append(layer)

        print(
            f"[bert_fraud] round {server_round} aggregation weights"
            f"{' (fallback to FedAvg — all alpha_c=0)' if fallback else ''}:"
        )
        for i, (norm, (_, n_c, alpha_c)) in enumerate(
            zip(normalized, weights_results)
        ):
            print(
                f"  client {i}: n={n_c:,} | local_auprc={alpha_c:.4f} | "
                f"weight={norm:.4f}"
            )

        metrics_aggregated: Dict[str, float] = {}
        if self.fit_metrics_aggregation_fn is not None:
            metrics_aggregated = self.fit_metrics_aggregation_fn(
                [(fit_res.num_examples, fit_res.metrics) for _, fit_res in results]
            )

        return fl.common.ndarrays_to_parameters(aggregated), metrics_aggregated


def get_strategy(
    cfg: dict,
    initial_parameters: fl.common.Parameters,
    server_eval_fn: Callable,
) -> AccuracyWeightedFedAvg:
    num_clients = int(cfg["num_clients"])
    local_epochs = int(cfg["local_epochs"])

    return AccuracyWeightedFedAvg(
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
