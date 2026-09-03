"""Phase-2 inference machinery — exact matching-null test, BH, profiles.

The decisive properties (per the RQ3 addendum): the exchangeability test's null
is EVERY perfect matching of the 2K vectors (945 at K=5, enumerated — an exact
test whose minimum p is 1/945); it holds its size under H0 and rejects when
clients genuinely differ; degenerate vectors yield undefined (never a fabricated
agreement); BH is the standard step-up; the weighted rank correlation stops the
near-zero tail from driving the statistic. Pure numpy/scipy — runnable off the
box via ``pytest`` (no shap/torch needed).
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from evaluation.shap_inference import (  # noqa: E402
    benjamini_hochberg, cell_inference, collapsed_seeds, double_factorial_odd,
    exchangeability_test, floor_and_between, kuncheva_profile, perfect_matchings,
    spearman, weighted_between_within, weighted_spearman,
)


def _h0_cell(rng, K=5, M=20, noise=0.05):
    """K clients x 2 seeds sharing ONE true importance vector + estimator noise."""
    true = rng.random(M) + 0.1
    return np.abs(true[None, None, :] + noise * rng.standard_normal((K, 2, M)))


def _h1_cell(rng, K=5, M=20, noise=0.05, shift=1.5):
    """Clients 0-1 follow a genuinely different importance vector than 2-4."""
    a, b = rng.random(M) + 0.1, rng.random(M) + 0.1
    b[: M // 2] += shift
    g = np.empty((K, 2, M))
    for c in range(K):
        true = a if c < 2 else b
        g[c] = np.abs(true[None, :] + noise * rng.standard_normal((2, M)))
    return g


# --------------------------------------------------------------------------- #
# matchings
# --------------------------------------------------------------------------- #
def test_matching_count_is_double_factorial():
    assert double_factorial_odd(4) == 3
    assert double_factorial_odd(10) == 945
    assert sum(1 for _ in perfect_matchings(list(range(4)))) == 3
    assert sum(1 for _ in perfect_matchings(list(range(10)))) == 945


def test_matchings_are_perfect():
    for m in perfect_matchings(list(range(6))):
        flat = sorted(i for pair in m for i in pair)
        assert flat == list(range(6))  # every item exactly once


# --------------------------------------------------------------------------- #
# exchangeability test
# --------------------------------------------------------------------------- #
def test_exact_null_holds_size_under_h0():
    """Rejection rate at alpha=0.05 stays near 0.05 when H0 is true (exact test)."""
    rng = np.random.default_rng(0)
    rej = 0
    reps = 150
    for _ in range(reps):
        g = _h0_cell(rng)
        flat = [g[c, s] for c in range(5) for s in range(2)]
        _, p, n = exchangeability_test(flat)
        assert n == 945
        assert p >= 1.0 / 945 - 1e-12  # observed matching is in the null set
        rej += p <= 0.05
    assert rej / reps < 0.12, f"size not controlled: {rej}/{reps}"


def test_rejects_when_clients_differ():
    rng = np.random.default_rng(1)
    g = _h1_cell(rng)
    flat = [g[c, s] for c in range(5) for s in range(2)]
    obs, p, n = exchangeability_test(flat)
    assert obs > 0
    assert p <= 0.05, f"clear two-population structure not detected: p={p}"


def test_collapsed_client_axis_yields_undefined_not_perfect_agreement():
    """Identical clients within a seed => undefined, never stability 1.0.

    The structural-1.0 regression: a shared background AND a shared explanation
    set hand every client identical inputs, so Spearman/Jaccard/Kuncheva all read
    1.0 and the exchangeability test reports p = 1.0 while measuring nothing.
    """
    rng = np.random.default_rng(20)
    one = np.abs(rng.random(30) + 0.1)
    g = np.repeat(one[None, None, :], 5, axis=0)      # 5 identical clients
    g = np.concatenate([g, g], axis=1)                # 2 seeds, both collapsed
    assert collapsed_seeds(g) == [0, 1]

    res = cell_inference(g)
    assert res["status"] == "undefined", res
    assert "collapsed client axis" in res["reason"]
    for k in ("p_value", "between_mean", "floor_mean", "delta", "disattenuated"):
        assert k not in res, f"{k} reported for a collapsed cell"

    # one seed collapsed, the other not => still undefined (the arm is broken)
    half = g.copy()
    half[:, 1] = np.abs(one[None, :] + 0.05 * rng.standard_normal((5, 30)))
    assert collapsed_seeds(half) == [0]
    assert cell_inference(half)["status"] == "undefined"

    # genuinely distinct clients are NOT flagged
    ok = np.abs(one[None, None, :] + 0.05 * rng.standard_normal((5, 2, 30)))
    assert collapsed_seeds(ok) == []
    assert cell_inference(ok)["status"] == "ok"

    # a single client cannot collapse a client axis that does not exist
    assert collapsed_seeds(ok[:1]) == []


def test_degenerate_vector_yields_undefined_not_agreement():
    rng = np.random.default_rng(2)
    g = _h0_cell(rng)
    g[3, 1] = 0.0  # one all-zero (client, seed) vector
    flat = [g[c, s] for c in range(5) for s in range(2)]
    obs, p, n = exchangeability_test(flat)
    assert obs != obs and p != p and n == 0  # nan, nan, 0 — never a fake value
    res = cell_inference(g)
    assert res["status"] == "undefined"
    assert (3, 1) in res["degenerate_client_seed"]
    assert "p_value" not in res  # NO stability numbers on an undefined cell


# --------------------------------------------------------------------------- #
# cell_inference
# --------------------------------------------------------------------------- #
def test_cell_inference_counts_and_h0_behaviour():
    rng = np.random.default_rng(3)
    g = _h0_cell(rng)
    F, B = floor_and_between(g)
    assert len(F) == 5 and len(B) == 20  # K and K(K-1) with 2 seeds
    res = cell_inference(g)
    assert res["status"] == "ok" and res["n_matchings"] == 945
    assert abs(res["delta"]) < 0.05  # floor and between agree under H0
    assert 0 < res["disattenuated"] <= 1.5


def test_cell_inference_detects_signal():
    rng = np.random.default_rng(4)
    res = cell_inference(_h1_cell(rng))
    assert res["status"] == "ok"
    assert res["delta"] > 0.05          # clients disagree beyond seeds
    assert res["p_value"] <= 0.05


def test_single_client_reports_floor_only():
    rng = np.random.default_rng(5)
    res = cell_inference(_h0_cell(rng, K=1))
    assert res["status"] == "ok" and res["n_clients"] == 1
    assert res["floor_mean"] == res["floor_min"]
    assert res["between_mean"] != res["between_mean"]  # nan: no between quantity
    assert res["p_value"] != res["p_value"]            # nan: no test


# --------------------------------------------------------------------------- #
# BH
# --------------------------------------------------------------------------- #
def test_benjamini_hochberg_known_example():
    # classical example: p=(.01,.02,.03,.04,.05) at m=5 -> adj=(.05,.05,.05,.05,.05)
    adj = benjamini_hochberg([0.01, 0.02, 0.03, 0.04, 0.05])
    assert np.allclose(adj, [0.05] * 5)
    # order-preserving + capped at 1
    adj2 = benjamini_hochberg([0.001, 0.9, 0.04])
    assert adj2[0] < adj2[2] < adj2[1] and max(adj2) <= 1.0
    # nan passes through, finite entries adjusted with m = #finite
    adj3 = benjamini_hochberg([0.01, float("nan"), 0.04])
    assert adj3[1] != adj3[1] and adj3[0] == 0.02


# --------------------------------------------------------------------------- #
# weighted rank correlation
# --------------------------------------------------------------------------- #
def test_weighted_spearman_uniform_weights_matches_plain():
    rng = np.random.default_rng(6)
    a, b = rng.random(30), rng.random(30)
    w = np.ones(30)
    assert abs(weighted_spearman(a, b, w) - spearman(a, b)) < 1e-9


def test_weighted_spearman_ignores_near_zero_tail():
    """Two vectors agree on the (heavy) head, disagree only on the ~zero tail:
    plain Spearman is dragged down, the weighted version is not."""
    rng = np.random.default_rng(7)
    head = np.array([10.0, 8, 6, 4, 2])
    a = np.concatenate([head, 1e-6 * rng.random(25)])
    b = np.concatenate([head, 1e-6 * rng.random(25)])
    w = (np.abs(a) + np.abs(b)) / 2
    assert spearman(a, b) < 0.6
    assert weighted_spearman(a, b, w) > 0.99


def test_weighted_between_within_keys():
    rng = np.random.default_rng(8)
    out = weighted_between_within(_h0_cell(rng))
    assert set(out) == {"weighted_between_mean", "weighted_within_mean"}
    assert out["weighted_between_mean"] > 0.8  # H0: high agreement everywhere


# --------------------------------------------------------------------------- #
# profile over k
# --------------------------------------------------------------------------- #
def test_kuncheva_profile_identical_rankings():
    v = np.tile(np.arange(20, 0, -1.0), (5, 2, 1))  # all clients/seeds identical
    prof = kuncheva_profile(v, d=20)
    assert prof["k"] == list(range(1, 21))
    assert np.allclose(prof["between_mean"], 1.0)
    assert np.allclose(prof["within_mean"], 1.0)


def test_kuncheva_profile_band_contains_mean():
    rng = np.random.default_rng(9)
    prof = kuncheva_profile(_h0_cell(rng, M=20, noise=0.3), d=20)
    for lo, mid, hi in zip(prof["within_min"], prof["within_mean"], prof["within_max"]):
        assert lo <= mid + 1e-12 and mid <= hi + 1e-12


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
