"""Validation-set iteration selection for HistGradientBoostingClassifier.

The sweep fits GBM at ``max_iter=100`` with ``early_stopping=False``. On the
extreme-imbalance no-SMOTE arm the full budget overfits into saturated,
degenerate probabilities (creditcard test AUPRC 0.18, paysim 0.07), while
XGBoost on *identical* data reaches 0.836 / 0.984 — the gap is a boosting-budget
artifact, not an imbalance finding. On BAF, where GBM (0.161) already matches
XGBoost (0.157) with no saturation, selection is expected to change nothing.

This module keeps the boosting prefix that maximises AUPRC on the **central**
validation set — the same signal the federated best-model strategy uses across
clients, here applied across iterations, per-arm adaptive by construction (the
no-SMOTE arm picks a short prefix, the SMOTE arm keeps most trees).

Chosen over ``early_stopping='auto'`` deliberately: 'auto' carves a hidden 10%
internal validation holdout — a second, model-specific split inside one of six
models that breaks the comparability the shared cache/hash layer guarantees —
and it degrades the SMOTE arm to rescue the no-SMOTE arm.
"""

from __future__ import annotations

import copy
from typing import Tuple

import numpy as np
from sklearn.metrics import average_precision_score


def select_best_iteration(model, x_val, y_val) -> Tuple[int, np.ndarray, float]:
    """Return ``(k_star, val_scores_at_k_star, val_auprc_at_k_star)``.

    ``k_star`` (1-based) is the boosting-prefix length maximising validation
    AUPRC over ``model.staged_predict_proba(x_val)``. Ties resolve to the SHORTER
    prefix (strict ``>``), preferring the simpler model. If the validation labels
    are degenerate (single class), the full budget is kept.
    """
    y_val = np.asarray(y_val)
    if np.unique(y_val).size < 2:
        n = int(getattr(model, "n_iter_", len(model._predictors)))
        return n, model.predict_proba(x_val)[:, 1], 0.0

    best_k, best_auprc, best_scores = 1, -np.inf, None
    for i, proba in enumerate(model.staged_predict_proba(x_val)):
        s = proba[:, 1]
        a = float(average_precision_score(y_val, s))
        if a > best_auprc:  # strict > keeps the shorter prefix on ties
            best_auprc, best_k, best_scores = a, i + 1, s
    return best_k, best_scores, float(best_auprc)


def truncate_to_iterations(model, k: int):
    """Copy of a fitted ``HistGradientBoostingClassifier`` using only its first
    ``k`` boosting iterations.

    ``predict_proba`` on the copy equals ``staged_predict_proba(...)[k-1]``
    exactly: boosting is additive, so the first ``k`` trees are independent of
    the configured budget (verified bitwise against a fresh ``max_iter=k`` fit).
    HistGBM exposes no public prefix-predict, so the ``_predictors`` list is
    sliced directly; ``n_iter_`` is a read-only property over ``_predictors`` and
    therefore reports the selected count automatically after the slice.
    """
    m = copy.deepcopy(model)
    m._predictors = m._predictors[: int(k)]
    return m
