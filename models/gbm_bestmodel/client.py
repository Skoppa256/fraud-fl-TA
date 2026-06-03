"""Flower NumPyClient for GBM with best-model selection.

Each client trains a fresh :class:`sklearn.ensemble.GradientBoostingClassifier`
on its local subset every round and ships the whole model to the server.
Tree ensembles cannot be element-wise averaged, so the server picks one
winner per round (see ``strategy.BestModelSelection``).

Serialization
-------------
``pickle`` round-trips the model through a uint8 byte array wrapped as a
Flower ``Parameters`` payload. **Pickle is unsafe for untrusted networks
(arbitrary code execution risk)** — this is fine here because we run the
whole simulation on one machine, but should NOT be used as-is in a
production cross-silo deployment.
"""

from __future__ import annotations

import pickle
import warnings
from typing import Any, Dict, List

import flwr as fl
import numpy as np
# Using HistGradientBoostingClassifier (histogram-based GBM) instead of
# GradientBoostingClassifier for computational tractability on PaySim scale
# (~4.4M rows). Both implement gradient boosting — the aggregation paradigm
# (best-model selection) is identical. HistGBM is ~10x faster due to
# histogram-based split finding, equivalent to LightGBM's core algorithm.
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import average_precision_score


def model_to_parameters_array(model: object) -> np.ndarray:
    """Pickle ``model`` and wrap the bytes as a uint8 ndarray."""
    return np.frombuffer(pickle.dumps(model), dtype=np.uint8)


def array_to_model(arr: np.ndarray):
    """Inverse of :func:`model_to_parameters_array`."""
    return pickle.loads(arr.tobytes())


def model_to_parameters(model: object) -> fl.common.Parameters:
    return fl.common.ndarrays_to_parameters([model_to_parameters_array(model)])


def parameters_to_model(parameters: fl.common.Parameters):
    arrays = fl.common.parameters_to_ndarrays(parameters)
    return array_to_model(arrays[0])


def _empty_payload() -> List[np.ndarray]:
    """Sentinel payload representing 'no model yet'."""
    return [np.array([], dtype=np.uint8)]


class GBMClient(fl.client.NumPyClient):
    """FedGBM best-model-selection client.

    Note
    ----
    The client retrains from scratch every round, ignoring the global
    model it receives in ``parameters``. This is intentional: the spec
    requires fixed ``n_estimators`` and warm-starting a GBM with extra
    trees would grow the ensemble unboundedly across rounds.
    """

    def __init__(
        self,
        x_local: np.ndarray,
        y_local: np.ndarray,
        client_id: int,
        cfg: dict,
        seed: int,
    ) -> None:
        self.x_local = x_local
        self.y_local = y_local
        self.client_id: int = int(client_id)
        self.cfg = cfg
        self.seed: int = int(seed) + self.client_id
        self.model: HistGradientBoostingClassifier | None = None

    def _build_local_model(self) -> HistGradientBoostingClassifier:
        p = self.cfg.get("gbm_params", {})
        # HistGradientBoostingClassifier uses ``max_iter`` (not ``n_estimators``)
        # and has no ``subsample`` parameter. ``early_stopping=False`` keeps
        # the boosting budget fixed at ``max_iter`` so the local model has the
        # same capacity every round.
        return HistGradientBoostingClassifier(
            max_iter=int(p.get("max_iter", 100)),
            learning_rate=float(p.get("learning_rate", 0.1)),
            max_depth=int(p.get("max_depth", 6)),
            early_stopping=False,
            random_state=self.seed,
        )

    def get_parameters(self, config: Dict[str, Any]) -> List[np.ndarray]:
        if self.model is None:
            return _empty_payload()
        return [model_to_parameters_array(self.model)]

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        # No-op: client retrains from scratch each round.
        return None

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        round_no = int(config.get("round", 0))

        if len(np.unique(self.y_local)) < 2:
            print(
                f"[client {self.client_id}] round {round_no}: only one class "
                "in local data — skipping local fit"
            )
            return (
                _empty_payload(),
                int(len(self.y_local)),
                {
                    "client_id": float(self.client_id),
                    "skipped": 1.0,
                    "local_auprc": 0.0,
                },
            )

        self.model = self._build_local_model()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            self.model.fit(self.x_local, self.y_local)

        local_scores = self.model.predict_proba(self.x_local)[:, 1]
        local_auprc = float(average_precision_score(self.y_local, local_scores))
        return (
            [model_to_parameters_array(self.model)],
            int(len(self.y_local)),
            {
                "client_id": float(self.client_id),
                "skipped": 0.0,
                "local_auprc": local_auprc,
            },
        )

    def evaluate(self, parameters, config):
        # Server-side eval only (the strategy never asks clients to evaluate).
        return 0.0, int(len(self.y_local)), {}


def build_client_fn(clients: List[dict], cfg: dict, seed: int):
    """Return a Flower-compatible ``client_fn`` closing over the client list."""

    def client_fn(cid: str) -> fl.client.NumPyClient:
        idx = int(cid)
        return GBMClient(
            x_local=clients[idx]["x"],
            y_local=clients[idx]["y"],
            client_id=idx,
            cfg=cfg,
            seed=seed,
        )

    return client_fn
