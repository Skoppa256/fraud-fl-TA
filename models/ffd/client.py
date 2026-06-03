"""Flower NumPyClient for FFD (Federated learning for Fraud Detection).

Implements the Yang et al. (2019) local procedure:
1. Receive global weights from the server.
2. Optionally apply SMOTE/ADASYN on the local partition (per-round, as
   specified by the paper's procedure step 3).
3. Train the local 1D CNN with SGD (lr=0.01, batch_size=80, 5 epochs).
4. Compute local AUPRC on the local data — used as the ``alpha_c`` weight
   in :class:`AccuracyWeightedFedAvg` aggregation.
5. Return ``(weights, n_samples_after_oversampling, {"local_auprc": alpha_c})``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import flwr as fl
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader, TensorDataset

from .model import FFDModel


class FFDClient(fl.client.NumPyClient):
    """FFD client running a 1D CNN with SGD + optional local oversampling."""

    def __init__(
        self,
        x_local: np.ndarray,
        y_local: np.ndarray,
        client_id: int,
        cfg: Dict[str, Any],
        seed: int = 42,
    ) -> None:
        self.x_local: np.ndarray = x_local
        self.y_local: np.ndarray = y_local
        self.client_id: int = int(client_id)
        self.cfg: Dict[str, Any] = cfg
        self.seed: int = int(seed) + self.client_id
        self.model: FFDModel = FFDModel(input_dim=int(x_local.shape[1]))

    def get_parameters(self, config: Dict[str, Any]) -> List[np.ndarray]:
        return self.model.get_weights()

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        self.model.set_weights(parameters)

    def fit(
        self, parameters: List[np.ndarray], config: Dict[str, Any]
    ) -> Tuple[List[np.ndarray], int, Dict[str, float]]:
        self.set_parameters(parameters)

        x_train, y_train = self._apply_oversampling(self.x_local, self.y_local)

        self._local_train(x_train, y_train, config)

        local_auprc = self._compute_local_auprc(self.x_local, self.y_local)

        return (
            self.get_parameters({}),
            int(len(x_train)),
            {"local_auprc": float(local_auprc), "client_id": float(self.client_id)},
        )

    def evaluate(
        self, parameters: List[np.ndarray], config: Dict[str, Any]
    ) -> Tuple[float, int, Dict[str, float]]:
        # Server handles evaluation centrally; this is a no-op.
        return 0.0, 0, {}

    def _local_train(
        self, x: np.ndarray, y: np.ndarray, config: Dict[str, Any]
    ) -> None:
        local_epochs = int(config.get("local_epochs", self.cfg["local_epochs"]))
        batch_size = int(self.cfg["batch_size"])
        lr = float(self.cfg["lr"])

        torch.manual_seed(self.seed)
        dataset = TensorDataset(
            torch.from_numpy(x.astype(np.float32)),
            torch.from_numpy(y.astype(np.int64)),  # CrossEntropyLoss needs int64
        )
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        self.model.train()
        for _ in range(local_epochs):
            for xb, yb in loader:
                xb, yb = xb.to(self.model.device), yb.to(self.model.device)
                optimizer.zero_grad()
                loss = criterion(self.model(xb), yb)
                loss.backward()
                optimizer.step()

    def _compute_local_auprc(self, x: np.ndarray, y: np.ndarray) -> float:
        if len(np.unique(y)) < 2:
            return 0.0
        probs = self.model.predict_proba(x)[:, 1]
        return float(average_precision_score(y, probs))

    def _apply_oversampling(
        self, x: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        method = str(self.cfg.get("oversampling", "smote")).lower()
        if method == "none":
            return x.astype(np.float32, copy=False), y.astype(np.int64, copy=False)

        client_dict = {
            "x": x,
            "y": y.astype(np.int32, copy=False),
            "client_id": self.client_id,
            "n_samples": int(len(y)),
            "n_fraud": int((y == 1).sum()),
            "fraud_ratio": float((y == 1).mean()) if len(y) > 0 else 0.0,
        }

        if method == "smote":
            from preprocessing.smote import apply_smote

            result = apply_smote(client_dict, enabled=True, base_seed=self.seed)
        elif method == "adasyn":
            from preprocessing.adasyn import apply_adasyn

            result = apply_adasyn(client_dict, enabled=True, base_seed=self.seed)
        else:
            return x.astype(np.float32, copy=False), y.astype(np.int64, copy=False)

        return (
            result["x"].astype(np.float32, copy=False),
            result["y"].astype(np.int64, copy=False),
        )


def build_client_fn(clients: List[dict], cfg: dict, seed: int):
    """Return a Flower ``client_fn`` closing over the partitioned client list."""

    def client_fn(cid: str) -> fl.client.NumPyClient:
        cd = clients[int(cid)]
        return FFDClient(
            x_local=cd["x"],
            y_local=cd["y"],
            client_id=int(cd["client_id"]),
            cfg=cfg,
            seed=seed,
        )

    return client_fn
