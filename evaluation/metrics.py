"""Shared, threshold-fair metric helpers for every FL and centralized run.

AUPRC is the primary metric and is threshold-free. F1 / precision / recall
are threshold-dependent, so to compare models whose scores live on different
scales — LR / GBM / FFD probabilities in ``[0, 1]``, SVM decision margins, and
the FedXGBllr CNN's compressed sigmoid outputs — on equal footing, the decision
threshold is *tuned on the validation set to maximize F1* and then applied
unchanged to the test set.

A fixed 0.5 (or 0.0 for SVM margins) is unfair: a model whose positive-class
scores never cross it scores ``F1 = precision = recall = 0`` despite ranking
well — which is exactly what happened to FedXGBllr on PaySim. Tuning per model
at its own F1-optimal operating point removes that artifact while keeping the
threshold-free AUPRC as the headline number.

The canonical end-of-run entry point is :func:`tuned_metrics`; per-round eval
loops can call :func:`best_f1_threshold` + :func:`metrics_at_threshold`.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)

__all__ = [
    "auprc",
    "best_f1_threshold",
    "metrics_at_threshold",
    "tuned_metrics",
]


def _as_arrays(y_true, scores) -> Tuple[np.ndarray, np.ndarray]:
    return np.asarray(y_true), np.asarray(scores, dtype=np.float64)


def auprc(y_true, scores) -> float:
    """Average precision (AUPRC). ``0.0`` if labels are degenerate/empty."""
    y_true, scores = _as_arrays(y_true, scores)
    if y_true.size == 0 or np.unique(y_true).size < 2:
        return 0.0
    return float(average_precision_score(y_true, scores))


def best_f1_threshold(y_true, scores) -> float:
    """Score cut-off that maximizes F1 over the PR curve of ``(y_true, scores)``.

    Sweeps the actual score values, so it is scale-agnostic — valid for
    probabilities *and* SVM decision margins. Predictions are made with
    ``score >= threshold``. Returns the score median if the labels are
    degenerate (single class / empty), which never happens on the real
    validation split but keeps the helper total.
    """
    y_true, scores = _as_arrays(y_true, scores)
    if y_true.size == 0 or np.unique(y_true).size < 2:
        return float(np.median(scores)) if scores.size else 0.5
    precision, recall, thresholds = precision_recall_curve(y_true, scores)

    denom = precision[:-1] + recall[:-1]
    f1 = np.divide(
        2.0 * precision[:-1] * recall[:-1],
        denom,
        out=np.zeros_like(denom),
        where=denom > 0,
    )
    if f1.size == 0:
        return float(np.median(scores)) if scores.size else 0.5
    return float(thresholds[int(np.argmax(f1))])


def metrics_at_threshold(y_true, scores, threshold: float) -> Dict[str, float]:
    """AUPRC (threshold-free) + F1/precision/recall at ``score >= threshold``."""
    y_true, scores = _as_arrays(y_true, scores)
    preds = (scores >= threshold).astype(np.int32)
    return {
        "auprc": auprc(y_true, scores),
        "f1": float(f1_score(y_true, preds, zero_division=0)),
        "precision": float(precision_score(y_true, preds, zero_division=0)),
        "recall": float(recall_score(y_true, preds, zero_division=0)),
    }


def tuned_metrics(
    y_val, val_scores, y_test, test_scores
) -> Tuple[float, Dict[str, float], Dict[str, float]]:
    """Tune the F1-optimal threshold on val, then score val *and* test at it.

    Returns ``(threshold, val_metrics, test_metrics)``.
    """
    threshold = best_f1_threshold(y_val, val_scores)
    return (
        threshold,
        metrics_at_threshold(y_val, val_scores, threshold),
        metrics_at_threshold(y_test, test_scores, threshold),
    )
