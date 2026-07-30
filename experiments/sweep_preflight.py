"""Preflight resource checks — fail loudly BEFORE run 1, never mid-sweep.

A previous sweep crashed the GPU because models requested resources
inconsistently. The core guards here are pure functions that take the detected
hardware as arguments (:func:`assert_gpu_capacity`, :func:`check_vram`,
:func:`check_disk`), so they are unit-testable with a mocked GPU count and VRAM
on a box that has no GPU. :func:`detect_gpus` is the thin, impure wrapper that
reads the real hardware; :func:`run_preflight` wires detection to the pure checks.
"""

from __future__ import annotations

import os
import shutil
from typing import Dict, List, Optional, Sequence, Tuple

from experiments import resources as R

GPU_MODELS = ("ffd", "bert_fraud", "fedxgbllr")
_BYTES_PER_GIB = 1024 ** 3


# --------------------------------------------------------------------------- #
# Memory preflight — estimate per-actor RAM and warn when concurrency × footprint
# exceeds physical RAM. Motivated by PaySim FedXGBllr OOM: 5 concurrent client
# actors each materialising an ~890k×250 tree-margin tensor + XGBoost buffers
# exceeded a 15 GB box, while ULB/BAF (4.5%/16% of PaySim rows) fit.
# --------------------------------------------------------------------------- #
def available_ram_bytes() -> int:
    """Total physical RAM in bytes (psutil if present, else POSIX sysconf).

    Returns ``0`` when it cannot be determined, so callers skip the check rather
    than warn on a bad estimate."""
    try:
        import psutil

        return int(psutil.virtual_memory().total)
    except Exception:  # noqa: BLE001
        try:
            return int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
        except (ValueError, OSError, AttributeError):
            return 0


def estimate_fl_actor_bytes(
    model: str,
    partition_rows: int,
    n_features: int,
    client_num: int = 5,
    n_estimators_client: int = 50,
) -> int:
    """Heuristic peak resident bytes for ONE client actor, from its training
    partition. Deliberately generous — a preflight that under-warns is useless.

    FedXGBllr dominates because ``utils.single_tree_preds_from_each_client``
    materialises a per-sample tree-margin tensor of width
    ``client_num * n_estimators_client`` for the whole partition, on top of the
    XGBoost DMatrix/quantile buffers.
    """
    row = 4  # float32
    base = partition_rows * (n_features + 1) * row  # x-partition + y
    if model == "fedxgbllr":
        # Margin tensor (partition_rows × client_num·n_estimators_client) is the
        # dominant term, held live through CNN autograd plus a DataLoader copy and
        # alongside XGBoost DMatrix/quantile buffers → ×6. ``partition_rows`` is
        # the LARGEST client's partition (peak actor), so this catches Dirichlet
        # skew. Calibrated so the estimate tracks the actual peak at 5 concurrent
        # actors on the 15 GB box: BAF iid (140k/client) ≈7.7 GiB fits, BAF α=0.5
        # (391k largest) ≈16 GiB OOMs, PaySim (890k) OOMs; PaySim sequential
        # (1 actor) ≈8.5 GiB fits. BAF α=1/α=5 (≤262k) stay under (no false alarm).
        margin = partition_rows * (client_num * n_estimators_client) * row
        return int(base * 3 + margin * 6)
    if model == "gbm":
        return int(base * 6)  # HistGBM histogram/binning buffers
    if model in ("ffd", "bert_fraud"):
        return int(base * 4)  # minibatch activations
    return int(base * 3)  # lr / svm


def memory_preflight(
    *,
    model: str,
    max_partition_rows: int,
    n_features: int,
    n_concurrent: int,
    client_num: int = 5,
    n_estimators_client: int = 50,
    available_bytes: Optional[int] = None,
    fixed_overhead_bytes: Optional[int] = None,
    safety: float = 0.85,
) -> Tuple[int, int, bool, str]:
    """Estimate peak RAM for ``n_concurrent`` client actors vs available RAM.

    ``peak = fixed_overhead + n_concurrent × per_actor(max_partition_rows)``.
    Sizing from the LARGEST client partition (not the mean) is essential under
    Dirichlet skew: BAF iid completed at concurrency 5 while BAF α=0.5 OOM'd at
    the same setting because its largest client held 391k of 700k rows. A
    balanced (iid) partition has ``max == mean``, so this is a strict
    generalisation. ``fixed_overhead`` covers the Ray object store (a per-run
    reservation) plus a driver reserve. Returns
    ``(peak_bytes, available_bytes, over_budget, message)``; ``over_budget`` is
    ``False`` when RAM cannot be determined (``available_bytes == 0``)."""
    if available_bytes is None:
        available_bytes = available_ram_bytes()
    if fixed_overhead_bytes is None:
        try:
            fixed_overhead_bytes = R.object_store_memory() + _BYTES_PER_GIB  # +1 GiB driver
        except Exception:  # noqa: BLE001
            fixed_overhead_bytes = 3 * _BYTES_PER_GIB
    partition_rows = max(1, int(max_partition_rows))
    per_actor = estimate_fl_actor_bytes(
        model, partition_rows, n_features, client_num, n_estimators_client
    )
    peak = int(fixed_overhead_bytes) + per_actor * max(1, n_concurrent)
    over = available_bytes > 0 and peak > safety * available_bytes
    g = _BYTES_PER_GIB
    msg = (
        f"est. peak ~{peak / g:.1f} GiB "
        f"({int(fixed_overhead_bytes) / g:.1f} GiB fixed + "
        f"{max(1, n_concurrent)} actor(s) × ~{per_actor / g:.1f} GiB) "
        f"vs {available_bytes / g:.1f} GiB RAM"
        + ("" if available_bytes else " (RAM unknown — check skipped)")
    )
    return peak, available_bytes, over, msg


def assert_gpu_capacity(
    model_resources: Dict[str, dict],
    gpu_count: int,
    concurrency: int = 1,
) -> List[str]:
    """Assert ``num_gpus_per_client × concurrency ≤ gpu_count`` for every GPU model.

    Pure and testable: the caller injects ``gpu_count`` (so this box, which has no
    GPU, can still exercise the assertion). Returns human-readable capacity lines.

    Rules
    -----
    * ``num_gpus == 0`` → CPU-only, no GPU assertion.
    * ``num_gpus > 0`` but ``gpu_count == 0`` → misuse (the request should have
      been CPU-resolved first): raises ``AssertionError``.
    * otherwise asserts the request fits and that at least one actor can schedule
      (``num_gpus_per_client ≤ 1.0`` per GPU — the "ActorPool is empty" failure).
    """
    lines: List[str] = []
    for model, res in model_resources.items():
        ng = float(res["num_gpus"])
        if ng <= 0.0:
            lines.append(f"{model}: CPU-only (num_gpus=0)")
            continue
        if gpu_count <= 0:
            raise AssertionError(
                f"{model} requests num_gpus={ng} but {gpu_count} GPU(s) available; "
                f"CPU fallback should have zeroed this before preflight."
            )
        demand = ng * concurrency
        if demand > gpu_count:
            raise AssertionError(
                f"GPU oversubscription: {model} num_gpus_per_client={ng} × "
                f"concurrency={concurrency} = {demand} > {gpu_count} GPU(s). "
                f"Lower --gpu-fraction or concurrency."
            )
        max_actors = int(gpu_count / ng)
        if max_actors < 1:
            raise AssertionError(
                f"{model} num_gpus_per_client={ng} > 1.0 per GPU; no actor can "
                f"schedule (this is the 'ActorPool is empty' failure)."
            )
        lines.append(
            f"{model}: num_gpus/client={ng}, max_concurrent_actors={max_actors}"
        )
    return lines


def check_vram(gpu_vram_free_bytes: Sequence[int], min_vram_gib: float) -> List[str]:
    """Assert every GPU has at least ``min_vram_gib`` free. Testable via injection."""
    lines: List[str] = []
    need = min_vram_gib * _BYTES_PER_GIB
    for i, free in enumerate(gpu_vram_free_bytes):
        free_gib = free / _BYTES_PER_GIB
        if free < need:
            raise AssertionError(
                f"GPU {i} has {free_gib:.1f} GiB free < required {min_vram_gib} GiB."
            )
        lines.append(f"GPU {i}: {free_gib:.1f} GiB free")
    return lines


def check_disk(free_bytes: int, estimated_bytes: int, path: str = ".") -> str:
    """Assert enough free disk for the sweep's artifacts. Testable via injection."""
    if free_bytes < estimated_bytes:
        raise AssertionError(
            f"Insufficient disk at {path}: {free_bytes / _BYTES_PER_GIB:.1f} GiB free "
            f"< estimated {estimated_bytes / _BYTES_PER_GIB:.1f} GiB for the sweep."
        )
    return (
        f"disk: {free_bytes / _BYTES_PER_GIB:.1f} GiB free "
        f">= {estimated_bytes / _BYTES_PER_GIB:.1f} GiB estimated"
    )


def detect_gpus() -> Tuple[int, List[int]]:
    """Return ``(gpu_count, [free_vram_bytes...])`` from the real hardware.

    Impure wrapper around torch. Returns ``(0, [])`` when CUDA is unavailable, so
    every model falls back to CPU with no exceptions.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return 0, []
        count = torch.cuda.device_count()
        vram: List[int] = []
        for i in range(count):
            try:
                free, _total = torch.cuda.mem_get_info(i)
            except Exception:
                free = int(torch.cuda.get_device_properties(i).total_memory)
            vram.append(int(free))
        return count, vram
    except Exception:
        return 0, []


def run_preflight(
    models: Sequence[str],
    gpu_fraction: float,
    n_runs: int,
    est_bytes_per_run: int = 200 * 1024 ** 2,  # ~200 MiB per run (models+cache+logs)
    concurrency: int = 1,
    path: Optional[str] = None,
    disk_path: str = ".",
) -> Dict[str, object]:
    """Full preflight: detect hardware, resolve resources, assert, print, or abort.

    Returns a dict summary. Raises ``AssertionError`` on any capacity/VRAM/disk
    violation so the caller aborts before run 1.
    """
    gpu_count, gpu_vram = detect_gpus()
    gpu_available = gpu_count > 0

    resolved = {m: R.for_model(m, gpu_available=gpu_available, path=path) for m in models}
    # Apply the requested GPU fraction to GPU models (sequential default = 1.0).
    for m, res in resolved.items():
        if res["num_gpus"] > 0:
            res["num_gpus"] = float(gpu_fraction)

    print("=== PREFLIGHT ===")
    print(f"GPUs detected: {gpu_count}" + ("" if gpu_available else "  → CPU fallback for ALL models"))

    cap_lines = assert_gpu_capacity(resolved, gpu_count=gpu_count, concurrency=concurrency)
    for ln in cap_lines:
        print("  " + ln)

    if gpu_available:
        for ln in check_vram(gpu_vram, R.min_vram_gib(path)):
            print("  " + ln)

    est_total = est_bytes_per_run * n_runs
    free = shutil.disk_usage(disk_path).free
    print("  " + check_disk(free, est_total, disk_path))
    print(f"  planned runs: {n_runs}")
    print("=== PREFLIGHT OK ===")

    return {
        "gpu_count": gpu_count,
        "gpu_available": gpu_available,
        "resolved": resolved,
        "n_runs": n_runs,
        "capacity": cap_lines,
    }
