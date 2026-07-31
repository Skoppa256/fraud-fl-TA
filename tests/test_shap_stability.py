"""SHAP stability metrics — Kuncheva is chance-corrected, Jaccard@5 is not.

The decisive test (per the analysis spec): for RANDOM client rankings at 13, 30,
and 55 features, Kuncheva's index returns ≈0 at every dimensionality while
Jaccard@5 returns a nonzero value that DIFFERS across dimensionalities — which is
exactly why Kuncheva is required for cross-dataset stability claims.
Runnable via ``pytest`` or ``python tests/<file>.py``.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from evaluation.shap_stability import (  # noqa: E402
    jaccard_at_k, kuncheva_index, spearman_stability, kuncheva_pair, mean_importance,
)


def _random_importances(n_clients, d, seed):
    rng = np.random.default_rng(seed)
    return [rng.random(d) for _ in range(n_clients)]


def test_kuncheva_near_zero_for_random_at_all_dims():
    """Kuncheva ≈ 0 for random rankings at 13, 30, 55 features (averaged over trials)."""
    for d in (13, 30, 55):
        vals = [kuncheva_index(_random_importances(5, d, s), d=d, k=5) for s in range(200)]
        assert abs(np.mean(vals)) < 0.03, f"Kuncheva not chance-corrected at d={d}: {np.mean(vals)}"


def test_jaccard_is_not_chance_corrected():
    """Jaccard@5 under random selection is nonzero AND differs across dimensionality."""
    means = {}
    for d in (13, 30, 55):
        means[d] = np.mean([jaccard_at_k(_random_importances(5, d, s), k=5) for s in range(200)])
    # nonzero at low d, and systematically shrinks with d (the incomparability)
    assert means[13] > 0.10, means
    assert means[13] > means[30] > means[55], means           # not comparable across settings
    assert means[55] < 0.10                                     # while Kuncheva stays ~0 everywhere


def test_identical_rankings_give_one():
    v = [np.array([5.0, 4, 3, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0])] * 4
    assert abs(kuncheva_index(v, d=13, k=5) - 1.0) < 1e-9
    assert abs(jaccard_at_k(v, k=5) - 1.0) < 1e-9
    assert abs(spearman_stability(v) - 1.0) < 1e-9


def test_kuncheva_pair_bounds_and_chance():
    # exact chance value: r == k²/d -> IC == 0
    d, k = 20, 5
    # two disjoint size-5 sets from 20 features share r=0 < k²/d=1.25 -> negative
    assert kuncheva_pair(set(range(5)), set(range(5, 10)), d) < 0
    # identical -> 1
    assert abs(kuncheva_pair(set(range(5)), set(range(5)), d) - 1.0) < 1e-9


def test_mean_importance_shape():
    imp = _random_importances(5, 30, 0)
    m = mean_importance(imp)
    assert m.shape == (30,)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nAll {len(fns)} SHAP-stability tests passed.")
