"""Feature Tokenizer + Transformer (FT-Transformer) for tabular fraud detection.

Architecture
------------
Adapted from Gorishniy et al. (2021) "Revisiting Deep Learning Models for
Tabular Data" (NeurIPS 2021), applied to the PaySim fraud dataset.

    Input  : (batch, input_dim=13) — standardised float32 features
    Tokenize: each feature x_i (scalar) → z_i = W_i * x_i + b_i ∈ R^d_model
              giving (batch, n_feat, d_model)
    [CLS]  : learnable token prepended → (batch, 1+n_feat, d_model)
    Encoder: N × TransformerEncoderLayer (Pre-LN, GELU, multi-head attention)
    Head   : CLS token output → LayerNorm → Linear → GELU → Dropout → Linear(2)

This is BERT-style in that:
  - A [CLS] token pools sequence-level information for classification.
  - Self-attention allows each feature to attend to every other feature,
    learning interaction effects that 1D-CNN and linear models miss.
  - Pre-LN (norm_first=True) stabilises training without warmup schedules.
"""

from __future__ import annotations

from typing import List

import numpy as np
import torch
import torch.nn as nn


class BertFraudModel(nn.Module):
    def __init__(
        self,
        input_dim: int = 13,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        device: str | None = None,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model

        # Per-feature linear tokenization: x_i (scalar) → W_i * x_i + b_i
        # W_i ∈ R^{d_model}, b_i ∈ R^{d_model}, one set of (W, b) per feature.
        self.feature_weights = nn.Parameter(torch.empty(input_dim, d_model))
        self.feature_biases = nn.Parameter(torch.zeros(input_dim, d_model))
        nn.init.trunc_normal_(self.feature_weights, std=0.02)

        # Learnable [CLS] token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Transformer encoder (Pre-LN for stable FL training without LR warmup)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Classification head on [CLS] output
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, dim_feedforward // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward // 4, 2),
        )

        if device is not None:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, input_dim) float32
        Returns:
            logits: (batch, 2) — pass directly to CrossEntropyLoss
        """
        # Tokenize: (B, n_feat) → (B, n_feat, d_model)
        tokens = x.unsqueeze(-1) * self.feature_weights + self.feature_biases

        # Prepend [CLS]: (B, 1+n_feat, d_model)
        cls = self.cls_token.expand(x.size(0), -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)

        # Self-attention over feature tokens
        out = self.transformer(tokens)  # (B, 1+n_feat, d_model)

        # [CLS] position → classification head
        return self.head(out[:, 0, :])  # (B, 2)

    def get_weights(self) -> List[np.ndarray]:
        return [p.detach().cpu().numpy() for p in self.parameters()]

    def set_weights(self, weights: List[np.ndarray]) -> None:
        for p, w in zip(self.parameters(), weights):
            p.data = torch.from_numpy(np.array(w)).to(p.device).type_as(p.data)

    def predict_proba(self, x_np: np.ndarray, batch_size: int = 2048) -> np.ndarray:
        """Inference on numpy array. Returns (N, 2) softmax probabilities.

        Processes in chunks to avoid OOM on large val/test sets (~945K rows).
        """
        self.eval()
        results = []
        with torch.no_grad():
            for start in range(0, len(x_np), batch_size):
                chunk = x_np[start : start + batch_size].astype(np.float32)
                x = torch.from_numpy(chunk).to(self.device)
                probs = torch.softmax(self.forward(x), dim=1).cpu().numpy()
                results.append(probs)
        return np.concatenate(results, axis=0)
