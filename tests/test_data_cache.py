"""Tests for the preprocessing/partition cache (commit 2).

Uses the smallest dataset (creditcard, ~285k rows) so the suite stays fast, and a
throwaway cache dir so it never touches results/cache. Runnable via ``pytest`` or
``python tests/<file>.py``.
"""

import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from experiments import data_cache  # noqa: E402

_DATASET = "creditcard"
_SEED = 42


def test_content_addressed_key_tracks_source():
    """The preprocessing key changes if the preprocessor source hash changes."""
    cfg = data_cache.preprocessing_config(_DATASET, _SEED)
    k1 = data_cache._cfg_key(cfg)
    # Same config → same key (stable).
    assert data_cache._cfg_key(data_cache.preprocessing_config(_DATASET, _SEED)) == k1
    # A different source hash (as if the preprocessor were edited) → different key.
    edited = dict(cfg, preproc_sha="deadbeef" * 8)
    assert data_cache._cfg_key(edited) != k1
    # Different seed → different key.
    assert data_cache._cfg_key(data_cache.preprocessing_config(_DATASET, 123)) != k1


def test_preprocess_roundtrip_preserves_dtype_and_values():
    with tempfile.TemporaryDirectory() as tmp:
        data, h1 = data_cache.get_preprocessed(_DATASET, _SEED, cache_root=tmp)
        # Second call hits the cache and returns the identical hash.
        data2, h2 = data_cache.get_preprocessed(_DATASET, _SEED, cache_root=tmp)
        assert h1 == h2, "cache hit produced a different data hash"
        for k in ("x_train", "y_train", "x_val", "y_val", "x_test", "y_test"):
            assert data[k].dtype == data2[k].dtype, f"dtype drift on {k}"
            assert np.array_equal(data[k], data2[k]), f"values changed on {k}"
        # x arrays are float32 (models must not silently receive float64).
        assert data["x_train"].dtype == np.float32
        assert data["y_train"].dtype == np.int32


def test_partition_integrity_disjoint_and_complete():
    with tempfile.TemporaryDirectory() as tmp:
        data, _ = data_cache.get_preprocessed(_DATASET, _SEED, cache_root=tmp)
        n = len(data["y_train"])
        for cond, alpha in [("iid", None), ("noniid", 0.5)]:
            idx, phash = data_cache.get_partition_indices(
                _DATASET, _SEED, alpha, cond, cache_root=tmp
            )
            concat = np.concatenate(idx)
            assert len(concat) == n, f"{cond}: rows dropped/duplicated"
            assert np.array_equal(np.unique(concat), np.arange(n)), (
                f"{cond}: union != full training index set"
            )
            assert isinstance(phash, str) and len(phash) == 64


def test_partition_hash_stable_across_reload():
    with tempfile.TemporaryDirectory() as tmp:
        _idx, h1 = data_cache.get_partition_indices(_DATASET, _SEED, 0.5, "noniid", cache_root=tmp)
        _idx2, h2 = data_cache.get_partition_indices(_DATASET, _SEED, 0.5, "noniid", cache_root=tmp)
        assert h1 == h2, "partition hash changed on cache reload"


def test_centralized_has_no_partition():
    with tempfile.TemporaryDirectory() as tmp:
        try:
            data_cache.get_partition_indices(_DATASET, _SEED, None, "centralized", cache_root=tmp)
        except ValueError:
            return
        raise AssertionError("centralized should have no client partition")


def test_cache_partition_matches_direct_get_partition():
    """The cached indices reproduce a direct get_partition's per-client minority counts."""
    from partitioning.dirichlet import get_partition

    with tempfile.TemporaryDirectory() as tmp:
        data, _ = data_cache.get_preprocessed(_DATASET, _SEED, cache_root=tmp)
        y = np.asarray(data["y_train"])
        direct = get_partition(
            np.asarray(data["x_train"]), y, scheme="dirichlet", alpha=0.5,
            num_clients=5, random_state=_SEED,
        )
        direct_counts = {c["client_id"]: int((c["y"] == 1).sum()) for c in direct}
        cached_counts = data_cache.partition_minority_counts(
            _DATASET, _SEED, 0.5, "noniid", cache_root=tmp
        )
        assert cached_counts == direct_counts, (
            f"cache vs direct mismatch: {cached_counts} != {direct_counts}"
        )


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nAll {len(fns)} data-cache tests passed.")
