"""Flower NumPyClient for BertFraud (FT-Transformer tabular model).

Local training procedure:
1. Receive global weights from the server.
2. Apply SMOTE/ADASYN on the local partition per-round (same as FFD).
3. Train with AdamW for ``local_epochs`` epochs.
4. Compute local AUPRC on the *original* (pre-oversampling) local data
   — used as the α_c weight in AccuracyWeightedFedAvg.
5. Return (weights, n_samples_after_oversampling, {local_auprc}).
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import flwr as fl
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader, TensorDataset

from .model import BertFraudModel


def _build_model(cfg: Dict[str, Any]) -> BertFraudModel:
    bp = cfg.get("bert_params", {})
    return BertFraudModel(
        input_dim=int(cfg.get("input_dim", 13)),
        d_model=int(bp.get("d_model", 64)),
        nhead=int(bp.get("nhead", 4)),
        num_layers=int(bp.get("num_layers", 2)),
        dim_feedforward=int(bp.get("dim_feedforward", 256)),
        dropout=float(bp.get("dropout", 0.1)),
        device=cfg.get("device", "cpu"),
    )


class BertFraudClient(fl.client.NumPyClient):
    """FT-Transformer client with per-round oversampling and AdamW optimisation."""

    def __init__(
        self,
        x_local: np.ndarray,
        y_local: np.ndarray,
        client_id: int,
        cfg: Dict[str, Any],
        seed: int = 42,
    ) -> None:
        self.x_local = x_local
        self.y_local = y_local
        self.client_id = int(client_id)
        self.cfg = cfg
        self.seed = int(seed) + self.client_id
        self.model = _build_model(cfg)

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
        weight_decay = float(self.cfg.get("weight_decay", 1e-4))

        torch.manual_seed(self.seed)
        dataset = TensorDataset(
            torch.from_numpy(x.astype(np.float32)),
            torch.from_numpy(y.astype(np.int64)),
        )
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )
        criterion = nn.CrossEntropyLoss()

        self.model.train()
        for _ in range(local_epochs):
            for xb, yb in loader:
                xb = xb.to(self.model.device)
                yb = yb.to(self.model.device)
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
        sampling_strategy = self.cfg.get("sampling_strategy", "auto")

        if method == "smote":
            from preprocessing.smote import apply_smote

            result = apply_smote(
                client_dict,
                enabled=True,
                sampling_strategy=sampling_strategy,
                base_seed=self.seed,
            )
        elif method == "adasyn":
            from preprocessing.adasyn import apply_adasyn

            result = apply_adasyn(
                client_dict,
                enabled=True,
                sampling_strategy=sampling_strategy,
                base_seed=self.seed,
            )
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
        return BertFraudClient(
            x_local=cd["x"],
            y_local=cd["y"],
            client_id=int(cd["client_id"]),
            cfg=cfg,
            seed=seed,
        )

    return client_fn
