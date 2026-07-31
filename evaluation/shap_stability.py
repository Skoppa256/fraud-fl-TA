"""Cross-client SHAP stability metrics.

Given one feature-importance vector per client (mean |SHAP| over the shared
explanation set), quantify how much clients agree on which features matter:

* :func:`spearman_stability` — mean pairwise Spearman rank correlation over the
  full importance ranking. Null is 0 regardless of dimensionality.
* :func:`jaccard_at_k` — mean pairwise Jaccard overlap of the top-k feature sets.
  Kept for continuity with §2.2.8, but NOT chance-corrected: its expected value
  under random selection is ``k²/d / (2k − k²/d)``, which shrinks as the feature
  count ``d`` grows, so a fixed top-k Jaccard is not comparable across datasets
  of different dimensionality (@nogueira2018stability).
* :func:`kuncheva_index` — Kuncheva's consistency index, chance-corrected:
  ``IC(A,B) = (r − k²/d) / (k − k²/d)`` for two size-k selections sharing ``r``
  features from ``d`` total. IC ≈ 0 for random selection at ANY ``d`` and ``k``,
  IC = 1 for identical, so it is the measure to use for cross-DATASET claims.
"""

from __future__ import annotations

import itertools
from typing import List, Sequence

import numpy as np


def _pairs(n: int):
    return list(itertools.combinations(range(n), 2))


def spearman_stability(importances: Sequence[np.ndarray]) -> float:
    """Mean pairwise Spearman rank correlation across client importance vectors."""
    from scipy.stats import spearmanr

    imp = [np.asarray(v, dtype=float) for v in importances]
    pr = _pairs(len(imp))
    if not pr:
        return float("nan")
    vals = []
    for i, j in pr:
        rho = spearmanr(imp[i], imp[j]).correlation
        vals.append(0.0 if rho is None or np.isnan(rho) else float(rho))
    return float(np.mean(vals))


def _topk_set(v: np.ndarray, k: int):
    return set(np.argsort(np.asarray(v, dtype=float))[::-1][:k].tolist())


def jaccard_at_k(importances: Sequence[np.ndarray], k: int = 5) -> float:
    """Mean pairwise Jaccard overlap of the top-k feature sets (NOT chance-corrected)."""
    sets = [_topk_set(v, k) for v in importances]
    pr = _pairs(len(sets))
    if not pr:
        return float("nan")
    return float(np.mean([len(sets[i] & sets[j]) / len(sets[i] | sets[j]) for i, j in pr]))


def kuncheva_pair(a: set, b: set, d: int) -> float:
    """Kuncheva consistency index for two equal-size selections from ``d`` features."""
    k = len(a)
    if k != len(b):
        raise ValueError("Kuncheva index requires equal-size selections")
    if k == 0 or k == d:
        return 1.0  # degenerate: no room for chance variation
    exp = k * k / d
    return (len(a & b) - exp) / (k - exp)


def kuncheva_index(importances: Sequence[np.ndarray], d: int, k: int = 5) -> float:
    """Mean pairwise Kuncheva consistency index over top-k client selections.

    Chance-corrected: ≈0 under random selection regardless of ``d`` or ``k``,
    which is what makes it comparable across the 13/30/55-feature datasets.
    """
    sets = [_topk_set(v, k) for v in importances]
    pr = _pairs(len(sets))
    if not pr:
        return float("nan")
    return float(np.mean([kuncheva_pair(sets[i], sets[j], d) for i, j in pr]))


def mean_importance(importances: Sequence[np.ndarray]) -> np.ndarray:
    """Consensus importance = elementwise mean over client importance vectors."""
    return np.mean(np.vstack([np.asarray(v, dtype=float) for v in importances]), axis=0)
