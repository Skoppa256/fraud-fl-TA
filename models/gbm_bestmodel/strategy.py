"""Custom Flower strategy: best-AUPRC client model wins each round.

Implements the rule from Aljunaid et al. 2025::

    W*_t  =  argmax_i  A( W_i, V )

where :math:`W_i` is client ``i``'s local GBM, :math:`V` is the held-out
server validation set, and :math:`A` is the AUPRC scorer. The winning
model is broadcast back to all clients next round (clients ignore it and
retrain from scratch — see ``client.GBMClient``), and is used by the
server for the round's centralized eval and any final-round test eval.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import flwr as fl
import numpy as np
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    Parameters,
    Scalar,
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from sklearn.metrics import average_precision_score

from .client import array_to_model
from .server import compute_metrics, new_early_stop_state, update_early_stop


class BestModelSelection(fl.server.strategy.Strategy):
    """Best-model client selection (no parameter averaging).

    Parameters
    ----------
    cfg:
        Resolved run-config dict.
    x_val, y_val:
        Server-held validation set used both for selection and for the
        per-round centralized metric log.
    x_test, y_test:
        Server-held test set, scored only on the final round.
    wandb_run:
        Optional active ``wandb.Run`` for streaming metrics.
    """

    def __init__(
        self,
        cfg: dict,
        x_val: np.ndarray,
        y_val: np.ndarray,
        x_test: np.ndarray,
        y_test: np.ndarray,
        wandb_run=None,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.x_val = x_val
        self.y_val = y_val
        self.x_test = x_test
        self.y_test = y_test
        self.wandb_run = wandb_run
        self.num_clients: int = int(cfg["num_clients"])
        self.num_rounds: int = int(cfg["num_rounds"])
        self.patience: int = int(cfg.get("early_stop_patience", 10))
        self.state: Dict[str, Any] = new_early_stop_state()

    # ------------------------------------------------------------------ #
    # Initialization & client orchestration
    # ------------------------------------------------------------------ #
    def initialize_parameters(
        self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        # Empty payload — clients ignore the initial parameters and train
        # a local GBM from scratch on round 1.
        return fl.common.ndarrays_to_parameters([np.array([], dtype=np.uint8)])

    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
    ) -> List[Tuple[ClientProxy, FitIns]]:
        clients = client_manager.sample(
            num_clients=self.num_clients,
            min_num_clients=self.num_clients,
        )
        fit_ins = FitIns(parameters, {"round": server_round})
        return [(c, fit_ins) for c in clients]

    def configure_evaluate(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        return []

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures,
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        return None, {}

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        # All centralized eval happens inside ``aggregate_fit`` so we
        # do not double-score the winner here.
        return None

    # ------------------------------------------------------------------ #
    # Best-model selection
    # ------------------------------------------------------------------ #
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures,
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        if not results:
            return None, {}

        per_client = self._evaluate_each_client(results)
        eligible = [c for c in per_client if not c["skipped"]]
        if not eligible:
            print(
                f"[server] round {server_round}: all clients skipped "
                "— keeping previous global model"
            )
            return None, {}

        winner = max(eligible, key=lambda c: c["val_auprc"])
        m_val = compute_metrics(self.y_val, winner["scores"], threshold=0.5)
        val_loss = 1.0 - m_val["auprc"]

        self._log_round(server_round, per_client, winner, m_val, val_loss)

        update_early_stop(
            self.state,
            server_round=server_round,
            val_auprc=m_val["auprc"],
            winner_client_id=winner["client_id"],
            patience=self.patience,
        )
        self.state["client_selections"].append(
            {
                "round": server_round,
                "selected_client_id": winner["client_id"],
                "val_auprc": m_val["auprc"],
            }
        )
        self.state["history"].append(
            {
                "round": server_round,
                "selected_client_id": winner["client_id"],
                "val_loss": val_loss,
                **{f"val_{k}": v for k, v in m_val.items()},
                **{
                    f"client_{c['client_id']}_auprc": c["val_auprc"]
                    for c in per_client
                },
            }
        )

        if server_round == self.num_rounds:
            self._final_test_eval(server_round, winner)

        aggregated_metrics: Dict[str, Scalar] = {
            "selected_client_id": float(winner["client_id"]),
            "val_auprc": m_val["auprc"],
            "val_f1": m_val["f1"],
            "val_precision": m_val["precision"],
            "val_recall": m_val["recall"],
        }
        return winner["params"], aggregated_metrics

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _evaluate_each_client(
        self, results: List[Tuple[ClientProxy, FitRes]]
    ) -> List[Dict[str, Any]]:
        per_client: List[Dict[str, Any]] = []
        for _, fit_res in results:
            client_id = int(fit_res.metrics.get("client_id", -1))
            skipped = float(fit_res.metrics.get("skipped", 0.0)) > 0.5
            if skipped:
                per_client.append(
                    {
                        "client_id": client_id,
                        "model": None,
                        "params": None,
                        "val_auprc": -1.0,
                        "scores": None,
                        "skipped": True,
                    }
                )
                continue
            arrays = fl.common.parameters_to_ndarrays(fit_res.parameters)
            model = array_to_model(arrays[0])
            scores = model.predict_proba(self.x_val)[:, 1]
            auprc = float(average_precision_score(self.y_val, scores))
            per_client.append(
                {
                    "client_id": client_id,
                    "model": model,
                    "params": fit_res.parameters,
                    "val_auprc": auprc,
                    "scores": scores,
                    "skipped": False,
                }
            )
        per_client.sort(key=lambda c: c["client_id"])
        return per_client

    def _log_round(
        self,
        server_round: int,
        per_client: List[Dict[str, Any]],
        winner: Dict[str, Any],
        m_val: Dict[str, float],
        val_loss: float,
    ) -> None:
        per_client_str = " | ".join(
            f"client_{c['client_id']}_auprc={c['val_auprc']:.4f}"
            for c in per_client
        )
        print(
            f"[server] round {server_round} | {per_client_str} | "
            f"selected_client={winner['client_id']} | "
            f"val_auprc={m_val['auprc']:.4f} | val_f1={m_val['f1']:.4f} | "
            f"val_precision={m_val['precision']:.4f} | "
            f"val_recall={m_val['recall']:.4f}"
        )
        if self.wandb_run is not None:
            log_dict: Dict[str, Any] = {
                "round": server_round,
                "best_client_id": int(winner["client_id"]),
                "val_loss": val_loss,
                "val_auprc": m_val["auprc"],
                "val_f1": m_val["f1"],
                "val_precision": m_val["precision"],
                "val_recall": m_val["recall"],
            }
            for c in per_client:
                log_dict[f"client_{c['client_id']}_auprc"] = c["val_auprc"]
            self.wandb_run.log(log_dict)

    def _final_test_eval(self, server_round: int, winner: Dict[str, Any]) -> None:
        test_scores = winner["model"].predict_proba(self.x_test)[:, 1]
        m_test = compute_metrics(self.y_test, test_scores, threshold=0.5)
        test_loss = 1.0 - m_test["auprc"]
        print(
            f"[server] FINAL round {server_round} | "
            f"test_auprc={m_test['auprc']:.4f} | test_f1={m_test['f1']:.4f} | "
            f"test_precision={m_test['precision']:.4f} | "
            f"test_recall={m_test['recall']:.4f}"
        )
        self.state["final_test"] = {
            "test_auprc": m_test["auprc"],
            "test_f1": m_test["f1"],
            "test_precision": m_test["precision"],
            "test_recall": m_test["recall"],
        }
        if self.wandb_run is not None:
            self.wandb_run.log(
                {
                    "round": server_round,
                    "test_loss": test_loss,
                    **self.state["final_test"],
                }
            )
