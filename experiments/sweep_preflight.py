"""Preflight resource checks — fail loudly BEFORE run 1, never mid-sweep.

A previous sweep crashed the GPU because models requested resources
inconsistently. The core guards here are pure functions that take the detected
hardware as arguments (:func:`assert_gpu_capacity`, :func:`check_vram`,
:func:`check_disk`), so they are unit-testable with a mocked GPU count and VRAM
on a box that has no GPU. :func:`detect_gpus` is the thin, impure wrapper that
reads the real hardware; :func:`run_preflight` wires detection to the pure checks.
"""

from __future__ import annotations

import shutil
from typing import Dict, List, Optional, Sequence, Tuple

from experiments import resources as R

GPU_MODELS = ("ffd", "bert_fraud", "fedxgbllr")
_BYTES_PER_GIB = 1024 ** 3


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
