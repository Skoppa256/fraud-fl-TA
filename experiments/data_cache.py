"""Content-addressed caching of preprocessed data and client partitions.

Guarantees every model within a ``(dataset, seed)`` sees byte-identical arrays,
and every model within a ``(dataset, seed, alpha, condition)`` sees the identical
partition — proven by hashes written into each run's record and a post-sweep
equality assert, rather than assumed.

Design notes
------------
* **Content-addressed keys.** The preprocessing cache is keyed on a hash of the
  *resolved preprocessing config* — which includes the SHA-256 of the
  preprocessor source and of ``preprocessing/loader.py`` — not merely
  ``(dataset, seed)``. Editing a preprocessor changes the key, forcing a
  recompute; a stale cache can never silently serve old arrays to new code (which
  would pass the equality assert because every run would agree on the same wrong
  data). Partitions are keyed the same way, including the partitioner source hash.

* **dtype preservation.** Arrays are persisted with ``np.save`` (dtype-exact) and
  reloaded-and-asserted on write, so a silent float64→float32 round-trip cannot
  change results. See :func:`get_preprocessed`.

* **Non-invasive partition indices.** ``partitioning`` is not modified. The
  partitioner's RNG depends only on ``y_train`` + seed (not on ``x`` values), and
  ``_build_client_record`` merely slices ``x_train[indices]``. So calling
  ``get_partition`` with an index-proxy ``x = arange(n)`` returns each client's
  ``x`` == its index array, provably identical to the real partition. That is how
  indices are recovered here.

* **Integrity assert.** On partition load the client index arrays are asserted
  pairwise disjoint and their union asserted to equal the full training index set
  exactly — a partitioner that drops or duplicates rows is otherwise invisible.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_ROOT = PROJECT_ROOT / "results" / "cache"

# Canonical arrays whose hash proves data-level comparability across models.
_CANONICAL = ("x_train", "y_train", "x_val", "y_val", "x_test", "y_test")

_PREPROC_SOURCE = {
    "paysim": PROJECT_ROOT / "preprocessing" / "paysim.py",
    "creditcard": PROJECT_ROOT / "preprocessing" / "creditcard.py",
    "baf": PROJECT_ROOT / "preprocessing" / "baf.py",
}
_LOADER_SOURCE = PROJECT_ROOT / "preprocessing" / "loader.py"
_PARTITIONER_SOURCE = PROJECT_ROOT / "partitioning" / "dirichlet.py"


# --------------------------------------------------------------------------- #
# Hashing helpers
# --------------------------------------------------------------------------- #
def _sha256_file(path: os.PathLike | str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _sha256_array(a: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(str(a.dtype).encode())
    h.update(str(a.shape).encode())
    h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def _sha256_arrays(arrays: List[np.ndarray]) -> str:
    h = hashlib.sha256()
    for a in arrays:
        h.update(_sha256_array(a).encode())
    return h.hexdigest()


def _cfg_key(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:16]


# --------------------------------------------------------------------------- #
# Resolved configs (content-addressed keys)
# --------------------------------------------------------------------------- #
def preprocessing_config(dataset: str, seed: int) -> dict:
    if dataset not in _PREPROC_SOURCE:
        raise KeyError(f"no preprocessor registered for dataset '{dataset}'")
    return {
        "kind": "preprocessing",
        "dataset": dataset,
        "seed": int(seed),
        "preproc_sha": _sha256_file(_PREPROC_SOURCE[dataset]),
        "loader_sha": _sha256_file(_LOADER_SOURCE),
    }


def partition_config(
    dataset: str, seed: int, alpha: Optional[float], condition: str, num_clients: int = 5
) -> dict:
    return {
        "kind": "partition",
        "dataset": dataset,
        "seed": int(seed),
        "alpha": None if alpha is None else float(alpha),
        "condition": condition,
        "num_clients": int(num_clients),
        "partitioner_sha": _sha256_file(_PARTITIONER_SOURCE),
    }


# --------------------------------------------------------------------------- #
# Preprocessing cache
# --------------------------------------------------------------------------- #
def _preproc_dir(cfg: dict, cache_root: Path) -> Path:
    return Path(cache_root) / "preprocessing" / f"{cfg['dataset']}_seed{cfg['seed']}_{_cfg_key(cfg)}"


def get_preprocessed(
    dataset: str, seed: int, cache_root: os.PathLike | str = CACHE_ROOT
) -> Tuple[Dict[str, object], str]:
    """Return ``(data_dict, data_hash)`` — from cache if fresh, else compute+persist.

    ``data_hash`` is the SHA-256 over the canonical arrays and is what every run
    records for the comparability assert.
    """
    cfg = preprocessing_config(dataset, seed)
    d = _preproc_dir(cfg, Path(cache_root))
    manifest = d / "manifest.json"
    if manifest.is_file():
        return _load_preprocessed(d)

    from preprocessing.loader import load_dataset

    data = load_dataset(dataset, random_state=seed)
    _save_preprocessed(d, data, cfg)
    # Reload so callers always get the persisted arrays (round-trip guarantee),
    # and assert dtype/values survived the round-trip.
    loaded, data_hash = _load_preprocessed(d)
    for k in _CANONICAL:
        assert np.asarray(data[k]).dtype == loaded[k].dtype, (
            f"dtype changed on cache round-trip for {k}: "
            f"{np.asarray(data[k]).dtype} -> {loaded[k].dtype}"
        )
    return loaded, data_hash


def _save_preprocessed(d: Path, data: Dict[str, object], cfg: dict) -> None:
    tmp = d.with_name(d.name + ".tmp")
    if tmp.exists():
        import shutil

        shutil.rmtree(tmp)
    tmp.mkdir(parents=True, exist_ok=True)

    import joblib

    arrays_meta: Dict[str, dict] = {}
    for key, val in data.items():
        if isinstance(val, np.ndarray):
            np.save(tmp / f"{key}.npy", val)
            arrays_meta[key] = {
                "dtype": str(val.dtype),
                "shape": list(val.shape),
                "sha256": _sha256_array(val),
            }
        elif key == "feature_names":
            (tmp / "feature_names.json").write_text(json.dumps(list(val)))
        elif key == "scaler":
            joblib.dump(val, tmp / "scaler.joblib")

    data_hash = _sha256_arrays([np.asarray(data[k]) for k in _CANONICAL])
    manifest = {
        "config": cfg,
        "config_key": _cfg_key(cfg),
        "data_hash": data_hash,
        "arrays": arrays_meta,
    }
    (tmp / "manifest.json").write_text(json.dumps(manifest, indent=2))
    # Atomic-ish publish.
    os.replace(tmp, d)


def _load_preprocessed(d: Path) -> Tuple[Dict[str, object], str]:
    manifest = json.loads((d / "manifest.json").read_text())
    out: Dict[str, object] = {}
    for key, meta in manifest["arrays"].items():
        arr = np.load(d / f"{key}.npy", allow_pickle=False)
        assert str(arr.dtype) == meta["dtype"], (
            f"cache dtype mismatch for {key}: {arr.dtype} != {meta['dtype']}"
        )
        out[key] = arr
    fn = d / "feature_names.json"
    if fn.is_file():
        out["feature_names"] = json.loads(fn.read_text())
    sc = d / "scaler.joblib"
    if sc.is_file():
        import joblib

        out["scaler"] = joblib.load(sc)
    return out, manifest["data_hash"]


# --------------------------------------------------------------------------- #
# Partition cache
# --------------------------------------------------------------------------- #
def _condition_to_scheme(condition: str, alpha: Optional[float]) -> Tuple[str, Optional[float]]:
    if condition == "iid":
        return "iid", None
    if condition == "noniid":
        if alpha is None:
            raise ValueError("noniid condition requires alpha")
        return "dirichlet", float(alpha)
    raise ValueError(
        f"condition '{condition}' has no client partition "
        f"(centralized is a single global fit — no partition cache)"
    )


def _extract_indices(y_train: np.ndarray, scheme: str, alpha: Optional[float],
                     num_clients: int, seed: int) -> List[np.ndarray]:
    """Recover per-client index arrays without touching partitioning logic.

    Uses an index-proxy x (arange) so each returned client['x'] is its index
    array — identical to the real partition because the RNG depends only on
    y_train + seed (see module docstring).
    """
    from partitioning.dirichlet import get_partition

    n = int(len(y_train))
    proxy = np.arange(n, dtype=np.int64).reshape(-1, 1)
    recs = get_partition(proxy, y_train, scheme=scheme, alpha=alpha,
                         num_clients=num_clients, random_state=seed)
    return [np.asarray(r["x"]).ravel().astype(np.int64) for r in recs]


def _assert_partition_integrity(indices: List[np.ndarray], n_train: int) -> None:
    """Client index arrays must be pairwise disjoint and union to the full set."""
    concat = np.concatenate(indices) if indices else np.array([], dtype=np.int64)
    # No duplicates across clients (pairwise disjoint) and no drops.
    assert len(concat) == n_train, (
        f"partition size {len(concat)} != training set size {n_train} "
        f"(rows dropped or duplicated)"
    )
    union = np.unique(concat)
    assert union.size == n_train and union[0] == 0 and union[-1] == n_train - 1, (
        "partition indices are not exactly the full training index set "
        "(0..n-1); a row was dropped or duplicated."
    )


def _partition_dir(cfg: dict, cache_root: Path) -> Path:
    a = "-" if cfg["alpha"] is None else f"a{cfg['alpha']:g}"
    name = f"{cfg['dataset']}_seed{cfg['seed']}_{cfg['condition']}_{a}_{_cfg_key(cfg)}"
    return Path(cache_root) / "partitions" / name


def get_partition_indices(
    dataset: str,
    seed: int,
    alpha: Optional[float],
    condition: str,
    num_clients: int = 5,
    cache_root: os.PathLike | str = CACHE_ROOT,
) -> Tuple[List[np.ndarray], str]:
    """Return ``(client_index_arrays, partition_hash)`` for iid/noniid conditions.

    Loads from cache if fresh; otherwise recovers indices, asserts integrity,
    persists, and returns. ``partition_hash`` goes into every run's record.
    Raises for ``condition='centralized'`` (no partition).
    """
    scheme, resolved_alpha = _condition_to_scheme(condition, alpha)
    cfg = partition_config(dataset, seed, resolved_alpha, condition, num_clients)
    d = _partition_dir(cfg, Path(cache_root))
    manifest = d / "manifest.json"

    if manifest.is_file():
        m = json.loads(manifest.read_text())
        indices = [np.load(d / f"client_{k}.npy") for k in range(m["num_clients"])]
        _assert_partition_integrity(indices, m["n_train"])
        return indices, m["partition_hash"]

    _data, _ = get_preprocessed(dataset, seed, cache_root=cache_root)
    y_train = np.asarray(_data["y_train"])
    indices = _extract_indices(y_train, scheme, resolved_alpha, num_clients, seed)
    _assert_partition_integrity(indices, int(len(y_train)))

    partition_hash = _sha256_arrays([np.sort(ix) for ix in indices])
    _save_partition(d, indices, y_train, cfg, partition_hash)
    return indices, partition_hash


def get_partition_sizes(
    dataset: str,
    seed: int,
    alpha: Optional[float],
    condition: str,
    num_clients: int = 5,
    cache_root: os.PathLike | str = CACHE_ROOT,
) -> List[int]:
    """Per-client training-sample counts for an iid/noniid partition.

    Read straight from the partition manifest (``per_client[].n_samples``) — no
    index arrays loaded — so callers can size a memory estimate from
    ``max(...)`` cheaply. Under Dirichlet the largest client can hold several×
    the mean (e.g. BAF α=0.5: 391k of 700k across 5 clients), and that largest
    partition drives the peak actor footprint. Falls back to computing the
    partition if the manifest is absent.
    """
    scheme, resolved_alpha = _condition_to_scheme(condition, alpha)
    cfg = partition_config(dataset, seed, resolved_alpha, condition, num_clients)
    d = _partition_dir(cfg, Path(cache_root))
    manifest = d / "manifest.json"
    if manifest.is_file():
        m = json.loads(manifest.read_text())
        per_client = m.get("per_client")
        if per_client:
            return [int(c["n_samples"]) for c in per_client]
    indices, _ = get_partition_indices(
        dataset, seed, alpha, condition, num_clients, cache_root
    )
    return [int(len(ix)) for ix in indices]


def _save_partition(d: Path, indices: List[np.ndarray], y_train: np.ndarray,
                    cfg: dict, partition_hash: str) -> None:
    tmp = d.with_name(d.name + ".tmp")
    if tmp.exists():
        import shutil

        shutil.rmtree(tmp)
    tmp.mkdir(parents=True, exist_ok=True)
    per_client = []
    for k, ix in enumerate(indices):
        np.save(tmp / f"client_{k}.npy", ix)
        yk = y_train[ix]
        per_client.append(
            {"client_id": k, "n_samples": int(len(ix)), "n_minority": int((yk == 1).sum())}
        )
    manifest = {
        "config": cfg,
        "config_key": _cfg_key(cfg),
        "partition_hash": partition_hash,
        "n_train": int(len(y_train)),
        "num_clients": len(indices),
        "per_client": per_client,
    }
    (tmp / "manifest.json").write_text(json.dumps(manifest, indent=2))
    os.replace(tmp, d)


def get_partition_clients(
    dataset: str,
    seed: int,
    scheme: str,
    alpha: Optional[float],
    num_clients: int = 5,
    cache_root: os.PathLike | str = CACHE_ROOT,
) -> Tuple[List[dict], str]:
    """Return ``(client_records, partition_hash)`` built from the CACHED data.

    Drop-in replacement for ``partitioning.dirichlet.get_partition`` in the entry
    points: same record structure (``x, y, client_id, n_samples, n_fraud,
    fraud_ratio``), but sliced from the cached preprocessed arrays at the cached
    indices — so every model provably consumes identical data + partition. Accepts
    the ``scheme`` (iid/dirichlet) the entry points already have, mapping it to the
    cache's condition.
    """
    from partitioning.dirichlet import _build_client_record

    condition = "iid" if scheme == "iid" else "noniid"
    indices, phash = get_partition_indices(
        dataset, seed, alpha, condition, num_clients, cache_root
    )
    data, _ = get_preprocessed(dataset, seed, cache_root=cache_root)
    x = np.asarray(data["x_train"])
    y = np.asarray(data["y_train"])
    clients = [_build_client_record(k, ix, x, y) for k, ix in enumerate(indices)]
    return clients, phash


def partition_minority_counts(
    dataset: str, seed: int, alpha: Optional[float], condition: str,
    num_clients: int = 5, cache_root: os.PathLike | str = CACHE_ROOT,
) -> Dict[int, int]:
    """Per-client minority counts from the cached partition (client_id -> count)."""
    indices, _ = get_partition_indices(dataset, seed, alpha, condition,
                                       num_clients, cache_root)
    data, _ = get_preprocessed(dataset, seed, cache_root=cache_root)
    y = np.asarray(data["y_train"])
    return {k: int((y[ix] == 1).sum()) for k, ix in enumerate(indices)}
