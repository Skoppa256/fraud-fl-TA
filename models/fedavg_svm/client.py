"""Flower NumPyClient for Linear SVM + FedAvg.

Mirrors the LR client one-to-one except that the underlying estimator
is :class:`sklearn.svm.LinearSVC`. Two consequences worth flagging:

1. ``LinearSVC`` does **not** accept ``warm_start`` in sklearn 1.5, so
   the corresponding yaml key is silently dropped when instantiating
   the estimator. Each local ``fit`` therefore retrains from scratch on
   that round's local subset; the server still aggregates the resulting
   parameters via FedAvg.

2. ``LinearSVC`` has no ``predict_proba``; ranking scores come from
   :meth:`LinearSVC.decision_function`. The binary-prediction threshold
   becomes ``0`` (sign of the margin), not ``0.5``.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List

import flwr as fl
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import average_precision_score
from sklearn.svm import LinearSVC


N_FEATURES: int = 13


def _build_svm(cfg: dict, seed: int) -> LinearSVC:
    params = dict(cfg.get("svm_params", {}))
    # LinearSVC has no warm_start in sklearn 1.5 — drop it if present so
    # the yaml stays declarative without crashing the constructor.
    params.pop("warm_start", None)
    return LinearSVC(
        C=float(params.get("C", 1.0)),
        max_iter=int(params.get("max_iter", 1000)),
        dual="auto",
        random_state=seed,
    )


def _initialise_unfit(model: LinearSVC, n_features: int) -> None:
    """Pre-populate sklearn attributes so the model can score pre-fit."""
    model.coef_ = np.zeros((1, n_features), dtype=np.float64)
    model.intercept_ = np.zeros(1, dtype=np.float64)
    model.classes_ = np.array([0, 1])
    model.n_features_in_ = n_features


class FraudFLClient(fl.client.NumPyClient):
    """FedAvg Linear-SVM client. See module docstring for the LinearSVC caveat."""

    def __init__(self, client_data: dict, cfg: dict, seed: int) -> None:
        self.client_id: int = int(client_data["client_id"])
        self.x: np.ndarray = client_data["x"]
        self.y: np.ndarray = client_data["y"]
        self.seed: int = int(seed) + self.client_id
        self.cfg = cfg
        self.model: LinearSVC = _build_svm(cfg, self.seed)
        _initialise_unfit(self.model, N_FEATURES)

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
        self.model.n_features_in_ = N_FEATURES

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

        local_scores = self.model.decision_function(self.x)
        local_auprc = float(average_precision_score(self.y, local_scores))
        return (
            self.get_parameters({}),
            int(len(self.y)),
            {"local_auprc": local_auprc, "client_id": float(self.client_id)},
        )

    def evaluate(self, parameters, config):
        # Unused with fraction_evaluate=0.0; implemented for completeness.
        self.set_parameters(parameters)
        scores = self.model.decision_function(self.x)
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
