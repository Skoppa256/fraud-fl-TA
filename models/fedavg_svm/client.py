"""Flower NumPyClient for Linear SVM + FedAvg.

The underlying estimator is :class:`sklearn.linear_model.SGDClassifier`
with ``loss="hinge"`` — i.e. a linear support-vector classifier trained
by stochastic gradient descent. This is a deliberate choice over
:class:`sklearn.svm.LinearSVC`: SGD is what makes the model
**FedAvg-correct**.

Why not ``LinearSVC``?
----------------------
``LinearSVC`` is a *batch* liblinear solver with no incremental fit and no
``warm_start``. Every ``fit`` call solves the local problem to its global
optimum from scratch, ignoring the server-aggregated weights set on
``coef_``. With fixed local data and seeds that means each client returns
the *same* coefficients every round, FedAvg averages identical inputs, and
the global model never moves — the metric curves are flat after round 1.

``SGDClassifier`` fixes this — but only if it is wired up correctly. Two
non-obvious requirements:

1. **Seed the aggregated weights via ``warm_start`` + ``fit``, not
   ``partial_fit``.** SGD trains on a *private* buffer (``_standard_coef``)
   that is only initialised from ``coef_`` when ``coef_init`` is passed.
   ``partial_fit`` always passes ``coef_init=None``, so weights written onto
   ``coef_`` by ``set_parameters`` are ignored and every round restarts from
   zero — exactly the no-progress failure we are trying to avoid. ``fit``
   with ``warm_start=True`` instead does ``coef_init = self.coef_``
   internally, so the aggregated weights genuinely seed the next round.

2. **Keep each local fit partial.** With ``tol=None`` and a *small*
   ``max_iter`` (the SGD epoch budget per round), each ``fit`` takes a fixed
   handful of gradient passes from the aggregated weights instead of
   re-converging to the local optimum. A full re-solve every round would
   flatten the curves just like LinearSVC did. Averaging these partial
   updates is what lets the global model improve round over round.

Scoring
-------
``loss="hinge"`` has no ``predict_proba``; ranking scores come from
:meth:`~sklearn.linear_model.SGDClassifier.decision_function`. The
binary-prediction threshold is ``0`` (sign of the margin), not ``0.5``.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List

import flwr as fl
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import average_precision_score


N_FEATURES: int = 13
CLASSES: np.ndarray = np.array([0, 1])


def _build_svm(cfg: dict, seed: int) -> SGDClassifier:
    params = dict(cfg.get("svm_params", {}))
    return SGDClassifier(
        loss="hinge",
        alpha=float(params.get("alpha", 1e-4)),
        max_iter=int(params.get("max_iter", 5)),  # SGD epochs per local round
        tol=None,  # run a fixed #epochs; never early-stop to the local optimum
        learning_rate=str(params.get("learning_rate", "optimal")),
        eta0=float(params.get("eta0", 0.0)),
        random_state=seed,
        warm_start=True,  # fit() seeds coef_init from the aggregated coef_
    )


def _initialise_unfit(model: SGDClassifier, n_features: int) -> None:
    """Pre-populate sklearn attributes so the model can score pre-fit."""
    model.coef_ = np.zeros((1, n_features), dtype=np.float64)
    model.intercept_ = np.zeros(1, dtype=np.float64)
    model.classes_ = CLASSES
    model.n_features_in_ = n_features


class FraudFLClient(fl.client.NumPyClient):
    """FedAvg linear-SVM (SGD-hinge) client. See module docstring for why SGD."""

    def __init__(self, client_data: dict, cfg: dict, seed: int) -> None:
        self.client_id: int = int(client_data["client_id"])
        self.x: np.ndarray = client_data["x"]
        self.y: np.ndarray = client_data["y"]
        self.seed: int = int(seed) + self.client_id
        self.cfg = cfg
        self.model: SGDClassifier = _build_svm(cfg, self.seed)
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
        self.model.classes_ = CLASSES
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

        # fit() with warm_start=True continues from the weights just loaded by
        # set_parameters() (i.e. the server's aggregated model): it passes
        # coef_init=self.coef_ into SGD's private buffer, which partial_fit
        # would NOT do. With tol=None + small max_iter each call is a few
        # gradient passes, not a full re-solve — that partialness is what lets
        # FedAvg make progress across rounds.
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
