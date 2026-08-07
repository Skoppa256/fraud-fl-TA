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
    """Mean pairwise Spearman rank correlation across client importance vectors.

    Returns ``nan`` (NEVER 0.0) when any pair's correlation is undefined. ``spearmanr``
    returns nan for a constant/degenerate importance vector — e.g. when attributions
    collapse to near-zero ties (FedXGBllr on PaySim, whose predicted probabilities are
    compressed to ~1e-9). An undefined correlation and a zero correlation mean OPPOSITE
    things: 0.0 says "clients disagree completely"; nan says "the ranking is degenerate
    and rank correlation is not defined". Coercing nan to 0.0 silently reports a
    degenerate cell as maximal disagreement — so we propagate it instead. Use
    :func:`near_zero_fraction` / :func:`degenerate_clients` to explain the nan.
    """
    import warnings

    from scipy.stats import ConstantInputWarning, spearmanr

    imp = [np.asarray(v, dtype=float) for v in importances]
    pr = _pairs(len(imp))
    if not pr:
        return float("nan")
    vals = []
    with warnings.catch_warnings():  # a constant input IS the nan we handle below
        warnings.simplefilter("ignore", ConstantInputWarning)
        for i, j in pr:
            rho = spearmanr(imp[i], imp[j]).correlation
            vals.append(float("nan") if rho is None else float(rho))
    return float(np.mean(vals))  # any nan pair -> nan cell (propagated, not coerced)


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


def near_zero_fraction(importances: Sequence[np.ndarray], eps: float = 1e-9) -> List[float]:
    """Per-client fraction of features whose mean|SHAP| is <= ``eps`` (absolute).

    A fraction near 1.0 means the model attributes almost nothing to any feature —
    e.g. FedXGBllr on PaySim, whose predicted probabilities compress to ~1e-9 so the
    attributions collapse. Report this so §4.5 can state whether the attributions are
    meaningful at all rather than reading a degenerate cell as a stability value.
    """
    return [float(np.mean(np.abs(np.asarray(v, float)) <= eps)) for v in importances]


def is_degenerate(v: np.ndarray, atol: float = 1e-12, rtol: float = 1e-9) -> bool:
    """True if an importance vector carries no usable ranking signal.

    Two degenerate cases, both of which make stability metrics meaningless:
      * all-(near-)zero — ``max|v| <= atol`` (KernelSHAP underflowed to nothing);
      * constant magnitude — ``ptp(|v|) <= rtol * max|v|`` (no feature outranks another).
    On such a vector ``spearmanr`` returns nan, and ``argsort`` of the ties yields an
    index-order top-k identical across clients — faking Jaccard = Kuncheva = 1.0. Any
    metric emitted for a degenerate cell is an artifact, so callers must report the
    cell as undefined instead.
    """
    v = np.abs(np.asarray(v, float))
    m = float(v.max()) if v.size else 0.0
    if m <= atol:
        return True
    return float(np.ptp(v)) <= rtol * m


def degenerate_clients(importances: Sequence[np.ndarray],
                       atol: float = 1e-12, rtol: float = 1e-9) -> List[int]:
    """Indices of clients whose importance vector is degenerate (see :func:`is_degenerate`)."""
    return [i for i, v in enumerate(importances) if is_degenerate(v, atol, rtol)]
