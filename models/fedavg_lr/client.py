"""Flower NumPyClient for Logistic Regression + FedAvg.

Each client holds a private local subset of the PaySim training set and
trains a :class:`sklearn.linear_model.LogisticRegression` on it. The
server aggregates ``coef_`` / ``intercept_`` via standard FedAvg.

Parameter serialization
-----------------------
A client's parameters are transmitted as the two-element list
``[coef_ (1, n_features) float32, intercept_ (1,) float32]``. This is
the same shape every round, which Flower requires.

``warm_start=True`` is the FedAvg-correct setting for sklearn LR: each
local ``fit`` call uses the server's aggregated coefficients as the
initial guess instead of restarting from zero.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List

import flwr as fl
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score


def _build_lr(cfg: dict, seed: int) -> LogisticRegression:
    params = dict(cfg.get("lr_params", {}))
    return LogisticRegression(
        C=float(params.get("C", 1.0)),
        max_iter=int(params.get("max_iter", 1000)),
        warm_start=bool(params.get("warm_start", True)),
        random_state=seed,
    )


def _initialise_unfit(model: LogisticRegression, n_features: int) -> None:
    """Pre-populate sklearn attributes so the model can predict pre-fit."""
    model.coef_ = np.zeros((1, n_features), dtype=np.float64)
    model.intercept_ = np.zeros(1, dtype=np.float64)
    model.classes_ = np.array([0, 1])
    model.n_features_in_ = n_features


class FraudFLClient(fl.client.NumPyClient):
    """FedAvg LR client.

    Parameters
    ----------
    client_data:
        Dict produced by ``apply_smote_to_all_clients`` / ``get_partition``.
    cfg:
        The merged run-config dict (yaml + CLI).
    seed:
        Base random seed. The client uses ``seed + client_id`` internally.
    """

    def __init__(self, client_data: dict, cfg: dict, seed: int) -> None:
        self.client_id: int = int(client_data["client_id"])
        self.x: np.ndarray = client_data["x"]
        self.y: np.ndarray = client_data["y"]
        self.seed: int = int(seed) + self.client_id
        self.cfg = cfg
        self.model: LogisticRegression = _build_lr(cfg, self.seed)
        # Feature count read from the local partition so the client model
        # adapts to any dataset (13 for PaySim, 30 for creditcard, ...).
        _initialise_unfit(self.model, int(self.x.shape[1]))

    def get_parameters(self, config: Dict[str, Any]) -> List[np.ndarray]:
        return [
            self.model.coef_.astype(np.float32, copy=False),
            self.model.intercept_.astype(np.float32, copy=False),
        ]

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        coef, intercept = parameters
        self.model.coef_ = coef.astype(np.float64, copy=False)
        self.model.intercept_ = intercept.astype(np.float64, copy=False)
        self.model.classes_ = np.array([0, 1])
        self.model.n_features_in_ = int(coef.shape[1])

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        local_epochs = int(config.get("local_epochs", 1))
        round_no = int(config.get("round", 0))

        if len(np.unique(self.y)) < 2:
            print(
                f"[client {self.client_id}] round {round_no}: only one class "
                "in local data — skipping local fit"
            )
            return (
                self.get_parameters({}),
                int(len(self.y)),
                {"client_id": float(self.client_id), "skipped": 1.0},
            )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            for _ in range(local_epochs):
                self.model.fit(self.x, self.y)

        local_scores = self.model.predict_proba(self.x)[:, 1]
        local_auprc = float(average_precision_score(self.y, local_scores))
        return (
            self.get_parameters({}),
            int(len(self.y)),
            {"local_auprc": local_auprc, "client_id": float(self.client_id)},
        )

    def evaluate(self, parameters, config):
        # Flower requires the method; with fraction_evaluate=0.0 the server
        # never invokes it. We still implement a real metric for completeness.
        self.set_parameters(parameters)
        scores = self.model.predict_proba(self.x)[:, 1]
        auprc = float(average_precision_score(self.y, scores))
        return 1.0 - auprc, int(len(self.y)), {"local_auprc": auprc}


def build_client_fn(clients: List[dict], cfg: dict, seed: int):
    """Return a Flower-compatible ``client_fn`` closing over the client list.

    Flower 1.5 accepts ``NumPyClient`` directly (no ``.to_client()`` wrapper —
    that helper was introduced in later versions).
    """

    def client_fn(cid: str) -> fl.client.NumPyClient:
        return FraudFLClient(
            client_data=clients[int(cid)], cfg=cfg, seed=seed
        )

    return client_fn
