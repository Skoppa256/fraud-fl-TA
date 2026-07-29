"""GBM validation-set iteration selection: truncation is exact, selection adaptive.

Runnable via ``pytest`` or ``python tests/<file>.py``.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402

from models.gbm_bestmodel.iteration_selection import (  # noqa: E402
    select_best_iteration,
    truncate_to_iterations,
)


def _fit(x, y, max_iter=60):
    return HistGradientBoostingClassifier(
        max_iter=max_iter, learning_rate=0.1, max_depth=6,
        random_state=42, early_stopping=False,
    ).fit(x, y)


def _imbalanced(n=6000, pos=40, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 8)).astype(np.float32)
    y = np.zeros(n, dtype=int)
    idx = rng.choice(n, pos, replace=False)
    y[idx] = 1
    x[idx] += 1.4  # separable-ish signal
    return x, y


def test_truncate_equals_staged_and_refit():
    """Truncating _predictors == staged prediction == a fresh max_iter=k fit."""
    x, y = _imbalanced()
    m = _fit(x, y)
    staged = list(m.staged_predict_proba(x))
    for k in (1, 5, 23, m.n_iter_):
        s_staged = staged[k - 1][:, 1]
        s_trunc = truncate_to_iterations(m, k).predict_proba(x)[:, 1]
        s_refit = _fit(x, y, max_iter=k).predict_proba(x)[:, 1]
        assert np.allclose(s_staged, s_trunc, atol=1e-12), f"trunc≠staged at k={k}"
        assert np.allclose(s_staged, s_refit, atol=1e-12), f"refit≠staged at k={k}"
    # n_iter_ reflects the slice (read-only property over _predictors).
    assert truncate_to_iterations(m, 7).n_iter_ == 7


def test_select_returns_scores_matching_truncated_model():
    """The returned k*, scores are self-consistent with the truncated model."""
    x, y = _imbalanced()
    m = _fit(x, y)
    k, scores, auprc = select_best_iteration(m, x, y)
    assert 1 <= k <= m.n_iter_
    assert np.allclose(scores, truncate_to_iterations(m, k).predict_proba(x)[:, 1])
    assert 0.0 <= auprc <= 1.0 + 1e-9  # AP can overshoot 1.0 by fp epsilon


def test_degenerate_val_keeps_full_budget():
    """Single-class validation labels → keep the full budget, no crash."""
    x, y = _imbalanced()
    m = _fit(x, y)
    k, _scores, auprc = select_best_iteration(m, x, np.zeros_like(y))
    assert k == m.n_iter_ and auprc == 0.0


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nAll {len(fns)} GBM iteration-selection tests passed.")
