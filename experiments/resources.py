"""Single source of truth for CPU/GPU resource allocation across the sweep.

Every model entry point resolves its Ray/compute allocation from
``experiments/sweep_resources.yaml`` through this module. There are **no built-in
fallback values**: a missing or incomplete config raises, so a bare per-model
invocation crashes loudly rather than silently guessing an allocation. That is
deliberate — a silent per-config default is exactly how the wrong value escaped
undetected before.

Hardware resolution: :func:`for_model` and :func:`xgboost_params` take an explicit
``gpu_available`` flag and downgrade GPU requests to CPU when it is ``False``, so
the CPU-fallback path is centralised in one place rather than re-implemented in
each entry point.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional

import yaml

DEFAULT_PATH = Path(__file__).resolve().parent / "sweep_resources.yaml"

MODELS = ("lr", "svm", "gbm", "ffd", "bert_fraud", "fedxgbllr")
_REQUIRED_MODEL_KEYS = {"device", "num_cpus", "num_gpus"}
_REQUIRED_TOP = {
    "gpu_fraction_default",
    "object_store_memory",
    "threads_per_actor",
    "min_vram_gib",
    "xgboost",
    "models",
}
_REQUIRED_XGB_KEYS = {"device", "tree_method"}


def load_resources(path: Optional[os.PathLike | str] = None) -> dict:
    """Load and fully validate the central resource config. Raises on any gap.

    Raises
    ------
    FileNotFoundError
        If the config file is absent — resources are required, not defaulted.
    ValueError
        If any required top-level key, model, or model/xgboost sub-key is missing.
    """
    p = Path(path) if path is not None else DEFAULT_PATH
    if not p.is_file():
        raise FileNotFoundError(
            f"resource config not found: {p}. Resources are required, not "
            f"defaulted — create sweep_resources.yaml (no built-in fallback exists)."
        )
    with open(p) as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"resource config {p} did not parse to a mapping")

    missing = _REQUIRED_TOP - set(cfg)
    if missing:
        raise ValueError(f"resource config {p} missing top-level keys: {sorted(missing)}")

    models_cfg = cfg["models"]
    if not isinstance(models_cfg, dict):
        raise ValueError(f"resource config {p}: 'models' must be a mapping")
    for m in MODELS:
        if m not in models_cfg:
            raise ValueError(f"resource config {p} missing required model '{m}'")
        gap = _REQUIRED_MODEL_KEYS - set(models_cfg[m])
        if gap:
            raise ValueError(
                f"resource config {p} model '{m}' missing keys: {sorted(gap)}"
            )

    xgb = cfg["xgboost"]
    if not isinstance(xgb, dict) or (_REQUIRED_XGB_KEYS - set(xgb)):
        raise ValueError(
            f"resource config {p} 'xgboost' missing keys: "
            f"{sorted(_REQUIRED_XGB_KEYS - set(xgb if isinstance(xgb, dict) else {}))}"
        )
    return cfg


def for_model(
    model: str,
    gpu_available: bool,
    path: Optional[os.PathLike | str] = None,
) -> Dict[str, object]:
    """Resolve ``{num_cpus, num_gpus, device}`` for a model, honouring hardware.

    ``gpu_available=False`` forces ``num_gpus=0.0`` and ``device='cpu'`` — the
    single, central CPU-fallback path.
    """
    cfg = load_resources(path)
    if model not in cfg["models"]:
        raise KeyError(f"no resources configured for model '{model}'")
    m = cfg["models"][model]
    num_cpus = int(m["num_cpus"])
    num_gpus = float(m["num_gpus"])
    device = str(m["device"])
    # --gpu-fraction crosses the subprocess boundary via SWEEP_GPU_FRACTION so the
    # CHILD (which independently calls for_model) actually gets the requested
    # per-client GPU fraction — not the config default. Without this the flag only
    # affected the parent's plan/manifest, and Ray ran everything sequentially at
    # 1.0 regardless. Applies only to GPU models (config num_gpus > 0).
    frac = os.environ.get("SWEEP_GPU_FRACTION")
    if frac is not None and num_gpus > 0.0:
        num_gpus = float(frac)
    if not gpu_available:
        num_gpus = 0.0
        device = "cpu"
    return {"num_cpus": num_cpus, "num_gpus": num_gpus, "device": device}


def xgboost_params(
    gpu_available: bool,
    path: Optional[os.PathLike | str] = None,
) -> Dict[str, str]:
    """Resolve XGBoost ``{device, tree_method}`` — explicit, never auto-detected."""
    cfg = load_resources(path)
    xgb = cfg["xgboost"]
    device = str(xgb["device"])
    tree_method = str(xgb["tree_method"])
    if not gpu_available:
        device = "cpu"
    return {"device": device, "tree_method": tree_method}


def object_store_memory(path: Optional[os.PathLike | str] = None) -> int:
    return int(load_resources(path)["object_store_memory"])


def gpu_fraction_default(path: Optional[os.PathLike | str] = None) -> float:
    return float(load_resources(path)["gpu_fraction_default"])


def min_vram_gib(path: Optional[os.PathLike | str] = None) -> float:
    return float(load_resources(path)["min_vram_gib"])


def threads_per_actor(path: Optional[os.PathLike | str] = None) -> int:
    return int(load_resources(path)["threads_per_actor"])


def pin_threads(n: Optional[int] = None, path: Optional[os.PathLike | str] = None) -> int:
    """Pin BLAS/OMP/torch thread counts so Ray actors don't thrash the box.

    Sets the env vars AND ``torch.set_num_threads`` at runtime. NB: the env vars
    take full effect only when set before numpy/torch import — the sweep runner
    also sets them in each subprocess's environment (authoritative); this call is
    the in-process belt-and-suspenders. Returns the resolved thread count.
    """
    if n is None:
        n = threads_per_actor(path)
    n = max(1, int(n))
    for var in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[var] = str(n)
    try:
        import torch

        torch.set_num_threads(n)
    except Exception:
        pass
    return n
