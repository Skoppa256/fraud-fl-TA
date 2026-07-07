"""FFD Model Architecture (Yang et al., 2019) adapted for PaySim.

Original paper: 30 PCA features (European Credit Card dataset).
Adaptation: ``input_dim=13`` to match PaySim after preprocessing.

Architecture
------------
    Input  : (batch, 1, input_dim)         # tabular features as 1D sequence
    Conv1  : Conv1d(1 -> 32, k=3) -> ReLU -> MaxPool1d(k=2)
    Conv2  : Conv1d(32 -> 64, k=3) -> ReLU -> MaxPool1d(k=2)
    Flatten -> FC(512) -> ReLU -> Dropout(0.5)
    Output : FC(2)                          # 2-class logits (0=normal, 1=fraud)

With input_dim=13 the spatial dim collapses as: 13 -> 11 -> 5 -> 3 -> 1,
so the FC input is 64 * 1 = 64. The flatten dim is computed dynamically
inside :meth:`_get_flatten_dim` so the same architecture stays valid if
the feature count changes later.
"""

from __future__ import annotations

from typing import List

import numpy as np
import torch
import torch.nn as nn


class FFDModel(nn.Module):
    def __init__(self, input_dim: int = 13, device: str | None = None) -> None:
        super().__init__()
        self.input_dim = int(input_dim)

        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )

        self._flatten_dim = self._get_flatten_dim(self.input_dim)

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self._flatten_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 2),
        )

        if device is not None:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def _get_flatten_dim(self, input_dim: int) -> int:
        """Run a dummy forward pass through conv stack to size the FC input."""
        with torch.no_grad():
            dummy = torch.zeros(1, 1, input_dim)
            x = self.conv2(self.conv1(dummy))
            return int(x.numel())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args:
            x: (batch, input_dim) float32
        Returns:
            logits: (batch, 2) — pass directly to ``CrossEntropyLoss``.
        """
        x = x.unsqueeze(1)  # (batch, 1, input_dim)
        x = self.conv1(x)
        x = self.conv2(x)
        return self.fc(x)

    def get_weights(self) -> List[np.ndarray]:
        """Return all parameters as a list of numpy arrays."""
        return [p.detach().cpu().numpy() for p in self.parameters()]

    def set_weights(self, weights: List[np.ndarray]) -> None:
        """Load parameters from a list of numpy arrays (must match shapes)."""
        for p, w in zip(self.parameters(), weights):
            p.data = torch.from_numpy(np.array(w)).to(p.device).type_as(p.data)

    def predict_proba(self, x_np: np.ndarray, batch_size: int = 8192) -> np.ndarray:
        """Inference on numpy array. Returns ``(N, 2)`` softmax probabilities.

        Inference is chunked into mini-batches. A full-batch forward pass over a
        large local partition (e.g. ~400k rows) produces multi-GiB conv/FC
        activations; with several Ray client actors sharing one GPU that
        exhausts memory (CUDA OOM). Batching bounds the peak activation size.
        """
        self.eval()
        x_all = np.ascontiguousarray(x_np, dtype=np.float32)
        out = np.empty((x_all.shape[0], 2), dtype=np.float32)
        with torch.no_grad():
            for start in range(0, x_all.shape[0], batch_size):
                xb = torch.from_numpy(x_all[start : start + batch_size]).to(self.device)
                logits = self.forward(xb)
                out[start : start + batch_size] = torch.softmax(logits, dim=1).cpu().numpy()
        return out
