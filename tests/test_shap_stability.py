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
    near_zero_fraction, is_degenerate, degenerate_clients,
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


def test_spearman_propagates_nan_never_zero():
    """A constant/all-zero importance vector -> Spearman nan, NOT a coerced 0.0.

    Regression for the PaySim FedXGBllr cells: nan (undefined) and 0.0 (complete
    disagreement) mean opposite things and must never be conflated.
    """
    allzero = [np.zeros(13) for _ in range(5)]
    assert np.isnan(spearman_stability(allzero))          # not 0.0
    const = [np.full(13, 3.0) for _ in range(5)]
    assert np.isnan(spearman_stability(const))
    # one degenerate client is enough to make the mean undefined
    mixed = [np.arange(13.0), np.arange(13.0), np.zeros(13),
             np.arange(13.0), np.arange(13.0)]
    assert np.isnan(spearman_stability(mixed))


def test_degenerate_detection():
    """is_degenerate / degenerate_clients flag all-zero and constant vectors."""
    assert is_degenerate(np.zeros(13))
    assert is_degenerate(np.full(13, 7.0))
    assert is_degenerate(np.full(13, 1e-15))              # underflowed to ~nothing
    assert not is_degenerate(np.arange(13.0))
    imps = [np.arange(13.0), np.zeros(13), np.full(13, 2.0), np.arange(13.0), np.arange(13.0)]
    assert degenerate_clients(imps) == [1, 2]
    assert not degenerate_clients([np.arange(13.0)] * 5)


def test_near_zero_fraction():
    v = np.array([0.0, 0.0, 0.0, 1.0, 2.0])                # 3/5 zero
    assert abs(near_zero_fraction([v])[0] - 0.6) < 1e-9
    assert near_zero_fraction([np.zeros(10)])[0] == 1.0    # all-zero cell
    assert near_zero_fraction([np.arange(1.0, 11.0)])[0] == 0.0


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nAll {len(fns)} SHAP-stability tests passed.")
