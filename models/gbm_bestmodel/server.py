"""Server-side helpers for the GBM best-model selection pipeline.

The actual round-by-round orchestration lives in
:class:`models.gbm_bestmodel.strategy.BestModelSelection`. This module
collects the metric-computation and early-stop bookkeeping helpers so
the strategy stays focused on the FL protocol.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)


def compute_metrics(
    y_true: np.ndarray, scores: np.ndarray, threshold: float = 0.5
) -> Dict[str, float]:
    """AUPRC + thresholded F1 / precision / recall for one set of scores."""
    preds = (scores > threshold).astype(np.int32)
    return {
        "auprc": float(average_precision_score(y_true, scores)),
        "f1": float(f1_score(y_true, preds, zero_division=0)),
        "precision": float(precision_score(y_true, preds, zero_division=0)),
        "recall": float(recall_score(y_true, preds, zero_division=0)),
    }


def new_early_stop_state() -> Dict[str, Any]:
    """Initialise the shared tracker the strategy mutates each round."""
    return {
        "best_val_auprc": -1.0,
        "best_round": -1,
        "best_client_id": -1,
        "patience_counter": 0,
        "early_stop_triggered": False,
        "history": [],
        "client_selections": [],
        "final_test": None,
    }


def update_early_stop(
    state: Dict[str, Any],
    server_round: int,
    val_auprc: float,
    winner_client_id: int,
    patience: int,
) -> None:
    """Mutate ``state`` with the latest round's outcome.

    Flower 1.5 cannot hard-halt a running simulation from inside a
    strategy, so the trigger only logs and flips a flag; the orchestrator
    surfaces ``best_*`` fields after the run finishes.
    """
    if val_auprc > state["best_val_auprc"]:
        state["best_val_auprc"] = val_auprc
        state["best_round"] = server_round
        state["best_client_id"] = winner_client_id
        state["patience_counter"] = 0
    else:
        state["patience_counter"] += 1
    if (
        state["patience_counter"] >= patience
        and not state["early_stop_triggered"]
    ):
        print(
            f"[server] early-stop signal at round {server_round}: "
            f"val_auprc unchanged for {patience} rounds "
            f"(best={state['best_val_auprc']:.4f} at round "
            f"{state['best_round']})"
        )
        state["early_stop_triggered"] = True
