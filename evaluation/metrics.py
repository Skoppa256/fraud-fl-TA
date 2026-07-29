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
    "calibration_metrics",
    "calibration_for",
    "baseline_auprc",
    "NA",
]

# Sentinel for a metric that is not applicable (e.g. calibration for a model
# that emits decision-function margins, not probabilities). Written verbatim to
# CSVs so downstream code never mistakes it for a real 0.0.
NA = "NA"


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


def calibration_metrics(y_true, probs, is_probability: bool = True) -> Dict[str, object]:
    """Calibration diagnostics on the central test set.

    Motivated by van den Goorbergh et al. (2022): imbalance corrections such as
    SMOTE do not improve discrimination but systematically overestimate predicted
    probabilities. AUPRC/F1 cannot detect that; these can.

    Returns ``{"brier", "cal_intercept", "cal_slope"}``:

    * **Brier score** — mean squared error between predicted probability and the
      0/1 outcome (lower is better).
    * **Calibration slope** — from the *joint* logistic recalibration
      ``y ~ a + b·logit(p)``: the coefficient ``b``. Slope ``1`` is ideal;
      ``< 1`` means over-confident (too extreme) probabilities, ``> 1`` means
      under-confident (too compressed — e.g. FedXGBllr, whose CNN collapses its
      outputs toward 0 while still ranking well, so a large positive slope is the
      correct, cleanly-converged diagnostic that the probabilities are far too
      compressed, not a numerical artifact).
    * **Calibration-in-the-large (reported as ``cal_intercept``)** — the
      intercept ``a`` of a *slope-fixed-at-1* offset model
      ``y ~ 1 + offset(logit(p))``, i.e. the shift solving
      ``mean(sigmoid(logit(p) + a)) = mean(y)``. Following van Calster et al.'s
      calibration hierarchy, this "are predictions right on average" number is
      reported SEPARATELY from the slope because the joint-fit intercept is
      coupled with the slope and is not independently interpretable (for a badly
      under-dispersed model the joint intercept explodes to meaningless
      magnitudes, e.g. 138, while calibration-in-the-large stays interpretable).
      ``0`` = calibrated in the large; ``> 0`` = systematic *under*-prediction
      (mean predicted below the event rate); ``< 0`` = *over*-prediction.

    ``is_probability=False`` (e.g. an SVM ``decision_function`` margin, which is
    NOT a probability) returns :data:`NA` for every field — a hinge-loss margin
    has no probability to recover, so passing it through a sigmoid would be a
    fabricated calibration, not a measured one.
    """
    if not is_probability:
        return {"brier": NA, "cal_intercept": NA, "cal_slope": NA}

    y_true = np.asarray(y_true).astype(np.int32)
    p = np.asarray(probs, dtype=np.float64)
    if y_true.size == 0 or np.unique(y_true).size < 2:
        return {"brier": NA, "cal_intercept": NA, "cal_slope": NA}

    brier = float(np.mean((p - y_true) ** 2))

    # Logistic recalibration: outcome ~ logit(p). Clip to keep the logit finite
    # AND well-conditioned. eps=1e-6 bounds the logit to ~±13.8; a smaller eps
    # (e.g. 1e-15 → logit ±34.5) numerically *inflates* the recalibration
    # degeneracy for models whose probabilities saturate to exactly 0/1 (e.g.
    # HistGradientBoosting produces hard 1.0 predictions). Applied UNIFORMLY to
    # every model — it is a no-op for non-saturating models (LR, SVM-NA, XGBoost)
    # and is not a per-model hack. A genuinely over-confident model still shows a
    # low calibration slope, which is a real finding, not an artifact.
    eps = 1e-6
    p_clipped = np.clip(p, eps, 1.0 - eps)
    logit = np.log(p_clipped / (1.0 - p_clipped))
    if np.allclose(logit, logit[0]):
        # Degenerate predictor (all identical scores) — slope undefined.
        return {"brier": brier, "cal_intercept": NA, "cal_slope": NA}

    try:
        from sklearn.linear_model import LogisticRegression

        # Calibration SLOPE: coefficient b of the joint fit y ~ a + b·logit(p).
        # Unpenalised so the fit is a true recalibration, not shrunk toward 0.
        lr = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000)
        lr.fit(logit.reshape(-1, 1), y_true)
        slope = float(lr.coef_[0][0])
        # Calibration-in-the-large: intercept of the slope-fixed-at-1 offset model.
        intercept = _calibration_in_the_large(logit, y_true)
    except Exception as exc:  # numerical failure — report Brier, NA the rest
        print(f"  [calibration] recalibration failed ({exc}); intercept/slope=NA")
        return {"brier": brier, "cal_intercept": NA, "cal_slope": NA}

    return {"brier": brier, "cal_intercept": intercept, "cal_slope": slope}


def _calibration_in_the_large(logit: np.ndarray, y_true: np.ndarray) -> float:
    """Intercept ``a`` of the slope-fixed-at-1 offset model ``y ~ 1 + offset(logit)``.

    Solves ``mean(sigmoid(logit + a)) = mean(y)`` — the MLE first-order condition
    for the intercept-only logistic model with ``logit(p)`` as a fixed offset.
    ``mean(sigmoid(logit + a))`` is strictly increasing in ``a`` and brackets
    ``mean(y) ∈ (0, 1)``, so a plain bisection is exact and dependency-free (no
    scipy/statsmodels needed). Returns ``0`` when already calibrated in the large.
    """
    ybar = float(np.mean(y_true))
    lo, hi = -50.0, 50.0

    def mean_pred(a: float) -> float:
        return float(np.mean(1.0 / (1.0 + np.exp(-(logit + a)))))

    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if mean_pred(mid) < ybar:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-10:
            break
    return float(0.5 * (lo + hi))


def calibration_for(y_test, test_scores, is_probability: bool) -> Dict[str, object]:
    """``test_``-prefixed calibration dict for merging into a final-test record.

    Thin wrapper over :func:`calibration_metrics` that renames the keys to match
    the results schema (``test_brier`` / ``test_cal_intercept`` /
    ``test_cal_slope``). ``is_probability=False`` yields all-:data:`NA`.
    """
    c = calibration_metrics(y_test, test_scores, is_probability=is_probability)
    return {
        "test_brier": c["brier"],
        "test_cal_intercept": c["cal_intercept"],
        "test_cal_slope": c["cal_slope"],
    }


def baseline_auprc(y_test) -> float:
    """Random-classifier AUPRC = positive-class prevalence of the test set."""
    y = np.asarray(y_test)
    return float(y.mean()) if y.size else 0.0


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
