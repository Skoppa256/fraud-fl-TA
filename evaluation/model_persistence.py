"""Persist final global models + provenance for the SHAP phase.

SHAP runs after the sweep against these exact frozen models, so each run must
persist its final global model paired with the *exact* data it consumed. Three
model kinds:

* **sklearn** (LR, SVM, GBM, centralized XGB) — ``joblib`` dump.
* **torch** (FFD, BERT) — ``state_dict`` + a JSON architecture config so the
  module can be reconstructed without the training code.
* **fedxgbllr** — the two-stage artifact: each client's XGBoost booster
  (``save_model`` → JSON) plus the aggregator CNN's ``state_dict`` + arch config.

Alongside the model, :func:`save_provenance` writes the fitted scaler, the ordered
feature names, and the ``data_hash`` / ``partition_hash`` the run consumed, tied
together in ``manifest.json``.

Round-trip contract (:func:`assert_roundtrip`): reload the artifact and assert
predictions on a fixed sample match the in-memory model — **bitwise** for
sklearn/tree models, and **within ``RTOL``/``ATOL``** for the deep models (whose
GPU/CPU kernels and non-associative float reductions make bitwise equality
unrealistic).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run_artifact_dir(dataset: str, run_name: str) -> Path:
    """Canonical per-run artifact directory: results/models/<dataset>/<run_name>/.

    Deterministic from ``(dataset, run_name)`` so the entry point (which builds the
    run_name) and the runner (which reconstructs it) agree without passing paths.
    """
    d = Path(_REPO_ROOT) / "results" / "models" / dataset / run_name
    d.mkdir(parents=True, exist_ok=True)
    return d


def manifest_ok(run_dir: os.PathLike | str) -> bool:
    """True iff the artifact dir has a manifest.json with the required keys."""
    p = Path(run_dir) / "manifest.json"
    if not p.is_file():
        return False
    try:
        m = json.loads(p.read_text())
    except Exception:
        return False
    return {"model_kind", "data_hash", "feature_names_n"} <= set(m)


# Deep-model reload tolerance (state_dict round-trip on the same device is exact
# in practice, but we allow a small margin for float non-determinism).
RTOL = 1e-5
ATOL = 1e-6


# --------------------------------------------------------------------------- #
# sklearn
# --------------------------------------------------------------------------- #
def save_sklearn(model, run_dir: os.PathLike | str) -> str:
    import joblib

    d = Path(run_dir)
    d.mkdir(parents=True, exist_ok=True)
    path = d / "model.joblib"
    joblib.dump(model, path)
    return str(path)


def load_sklearn(run_dir: os.PathLike | str):
    import joblib

    return joblib.load(Path(run_dir) / "model.joblib")


# --------------------------------------------------------------------------- #
# torch (FFD / BERT)
# --------------------------------------------------------------------------- #
def save_torch(model, arch_config: Dict[str, Any], run_dir: os.PathLike | str) -> str:
    import torch

    d = Path(run_dir)
    d.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), d / "state_dict.pt")
    (d / "arch_config.json").write_text(json.dumps(arch_config, indent=2))
    return str(d / "state_dict.pt")


def load_torch(model_cls, run_dir: os.PathLike | str, device: str = "cpu"):
    """Reconstruct ``model_cls(**arch_config)`` and load its weights."""
    import torch

    d = Path(run_dir)
    arch = json.loads((d / "arch_config.json").read_text())
    model = model_cls(**arch)
    sd = torch.load(d / "state_dict.pt", map_location=device)
    model.load_state_dict(sd)
    model.eval()
    return model


# --------------------------------------------------------------------------- #
# fedxgbllr (two-stage: per-client boosters + aggregator CNN)
# --------------------------------------------------------------------------- #
def save_fedxgbllr(trees: Sequence, cnn, cnn_arch: Dict[str, Any],
                   run_dir: os.PathLike | str) -> str:
    """Persist each client's XGBoost booster + the aggregator CNN state_dict."""
    import torch

    d = Path(run_dir)
    (d / "boosters").mkdir(parents=True, exist_ok=True)
    for i, est in enumerate(trees):
        # XGBClassifier/XGBRegressor.save_model → portable JSON booster.
        est.save_model(str(d / "boosters" / f"client_{i}.json"))
    torch.save(cnn.state_dict(), d / "cnn_state_dict.pt")
    (d / "cnn_arch_config.json").write_text(json.dumps(cnn_arch, indent=2))
    (d / "n_boosters.json").write_text(json.dumps({"n": len(trees)}))
    return str(d)


def load_fedxgbllr(xgb_cls, cnn_cls, cnn_ctor_args, run_dir: os.PathLike | str,
                   device: str = "cpu"):
    """Reload ``(trees, cnn)``. ``cnn_ctor_args`` is passed to ``cnn_cls(*args)``."""
    import torch

    d = Path(run_dir)
    n = json.loads((d / "n_boosters.json").read_text())["n"]
    trees = []
    for i in range(n):
        est = xgb_cls()
        est.load_model(str(d / "boosters" / f"client_{i}.json"))
        trees.append(est)
    cnn = cnn_cls(*cnn_ctor_args)
    cnn.load_state_dict(torch.load(d / "cnn_state_dict.pt", map_location=device))
    cnn.eval()
    return trees, cnn


# --------------------------------------------------------------------------- #
# provenance + manifest
# --------------------------------------------------------------------------- #
def save_provenance(
    run_dir: os.PathLike | str,
    *,
    scaler,
    feature_names: Sequence[str],
    data_hash: str,
    partition_hash: str,
    config_hash: str,
    model_kind: str,
    extra: Dict[str, Any] | None = None,
) -> str:
    """Write scaler, ordered feature names, and a manifest tying artifact ↔ data."""
    import joblib

    d = Path(run_dir)
    d.mkdir(parents=True, exist_ok=True)
    if scaler is not None:
        joblib.dump(scaler, d / "scaler.joblib")
    (d / "feature_names.json").write_text(json.dumps(list(feature_names)))
    manifest = {
        "model_kind": model_kind,
        "data_hash": data_hash,
        "partition_hash": partition_hash,
        "config_hash": config_hash,
        "feature_names_n": len(feature_names),
        "has_scaler": scaler is not None,
        **(extra or {}),
    }
    (d / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return str(d / "manifest.json")


# --------------------------------------------------------------------------- #
# round-trip assertions
# --------------------------------------------------------------------------- #
def persist_run(
    kind: str,
    *,
    dataset: str,
    run_name: str,
    scaler,
    feature_names: Sequence[str],
    data_hash: str,
    partition_hash: str,
    threshold,
    sklearn_model=None,
    torch_model=None,
    arch_config: Dict[str, Any] | None = None,
    fedxgbllr_trees: Sequence | None = None,
    fedxgbllr_cnn=None,
    reference_pred=None,
) -> str:
    """One call to persist a run's final global model + provenance + manifest.

    Keeps entry-point edits to a single call. ``threshold`` (the resolved
    operating point the metrics were computed at) is stored in the manifest so a
    downstream classification analysis uses the same cut-off as SHAP's model.
    Returns the artifact directory.
    """
    d = run_artifact_dir(dataset, run_name)
    if kind == "sklearn":
        save_sklearn(sklearn_model, d)
    elif kind == "torch":
        save_torch(torch_model, arch_config or {}, d)
    elif kind == "fedxgbllr":
        save_fedxgbllr(fedxgbllr_trees, fedxgbllr_cnn, arch_config or {}, d)
    else:
        raise ValueError(f"unknown model kind {kind!r}")
    save_provenance(
        d, scaler=scaler, feature_names=feature_names, data_hash=data_hash,
        partition_hash=partition_hash, config_hash=run_name, model_kind=kind,
        extra={"threshold": ("" if threshold is None else threshold)},
    )
    if reference_pred is not None:
        # In-memory prediction on a fixed sample, saved at train time so a
        # reload check can assert the reloaded model reproduces it (catches a
        # wrong torch arch config, which otherwise only surfaces weeks later).
        np.save(Path(d) / "reference_pred.npy", np.asarray(reference_pred))
    return str(d)


def assert_bitwise(a: np.ndarray, b: np.ndarray, label: str) -> None:
    if not np.array_equal(np.asarray(a), np.asarray(b)):
        raise AssertionError(f"{label}: reloaded predictions differ bitwise from in-memory")


def assert_within_tol(a: np.ndarray, b: np.ndarray, label: str,
                      rtol: float = RTOL, atol: float = ATOL) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if not np.allclose(a, b, rtol=rtol, atol=atol):
        max_abs = float(np.max(np.abs(a - b)))
        raise AssertionError(
            f"{label}: reloaded predictions differ beyond tol "
            f"(rtol={rtol}, atol={atol}); max|Δ|={max_abs:.3e}"
        )
    return float(np.max(np.abs(a - b))) if a.size else 0.0
