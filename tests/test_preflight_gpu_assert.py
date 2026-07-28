"""Unit tests for the preflight GPU-capacity / VRAM / disk assertions.

This box has no GPU, so the real preflight always takes the CPU-fallback path and
the ``num_gpus × concurrent ≤ 1.0`` logic is never exercised locally. These tests
INJECT a fake GPU count and VRAM so the assertion is verified here, before it
reaches the GPU server. Runnable via ``pytest`` or ``python tests/<file>.py``.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from experiments.sweep_preflight import (  # noqa: E402
    assert_gpu_capacity,
    check_disk,
    check_vram,
)

_GIB = 1024 ** 3


def test_sequential_fraction_one_passes():
    """gpu_fraction=1.0, concurrency=1, 1 GPU → fits, exactly one actor."""
    res = {"ffd": {"num_gpus": 1.0}, "bert_fraud": {"num_gpus": 1.0}}
    lines = assert_gpu_capacity(res, gpu_count=1, concurrency=1)
    assert any("max_concurrent_actors=1" in ln for ln in lines)


def test_oversubscription_raises():
    """0.6 × 2 concurrent = 1.2 > 1 GPU must raise."""
    res = {"bert_fraud": {"num_gpus": 0.6}}
    try:
        assert_gpu_capacity(res, gpu_count=1, concurrency=2)
    except AssertionError as e:
        assert "oversubscription" in str(e)
        return
    raise AssertionError("expected AssertionError for GPU oversubscription")


def test_fraction_above_one_raises():
    """num_gpus_per_client > 1.0 → no actor can schedule (ActorPool empty)."""
    res = {"ffd": {"num_gpus": 1.5}}
    try:
        assert_gpu_capacity(res, gpu_count=1, concurrency=1)
    except AssertionError:
        return
    raise AssertionError("expected AssertionError for fraction > 1.0")


def test_gpu_request_without_gpu_raises():
    """A non-zero GPU request with 0 GPUs detected is misuse (should be CPU-resolved)."""
    res = {"ffd": {"num_gpus": 1.0}}
    try:
        assert_gpu_capacity(res, gpu_count=0, concurrency=1)
    except AssertionError:
        return
    raise AssertionError("expected AssertionError for GPU request with 0 GPUs")


def test_cpu_only_models_never_assert():
    """num_gpus=0 models pass regardless of GPU count (even 0)."""
    res = {"lr": {"num_gpus": 0.0}, "svm": {"num_gpus": 0.0}, "gbm": {"num_gpus": 0.0}}
    lines = assert_gpu_capacity(res, gpu_count=0, concurrency=1)
    assert all("CPU-only" in ln for ln in lines)


def test_vram_below_minimum_raises():
    """Injected VRAM below the minimum must raise."""
    try:
        check_vram([2 * _GIB], min_vram_gib=4)
    except AssertionError as e:
        assert "free" in str(e)
        return
    raise AssertionError("expected AssertionError for insufficient VRAM")


def test_vram_sufficient_passes():
    lines = check_vram([8 * _GIB, 8 * _GIB], min_vram_gib=4)
    assert len(lines) == 2


def test_disk_insufficient_raises():
    try:
        check_disk(free_bytes=1 * _GIB, estimated_bytes=10 * _GIB)
    except AssertionError:
        return
    raise AssertionError("expected AssertionError for insufficient disk")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nAll {len(fns)} preflight tests passed.")
