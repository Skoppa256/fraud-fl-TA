"""Resources must be required, not defaulted.

A missing or incomplete sweep_resources.yaml must raise — a bare per-model
invocation should crash loudly rather than fall back to any built-in value (the
failure mode that let a wrong per-config value escape undetected before).
Runnable via ``pytest`` or ``python tests/<file>.py``.
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from experiments import resources  # noqa: E402


def test_missing_file_raises_filenotfound():
    try:
        resources.load_resources(path="/nonexistent/sweep_resources.yaml")
    except FileNotFoundError:
        return
    raise AssertionError("expected FileNotFoundError for missing resource config")


def test_incomplete_config_raises_valueerror():
    """A config missing a required model must raise ValueError, not default."""
    incomplete = (
        "gpu_fraction_default: 1.0\n"
        "object_store_memory: 1\n"
        "threads_per_actor: 1\n"
        "min_vram_gib: 4\n"
        "xgboost: {device: cpu, tree_method: hist}\n"
        "models:\n"
        "  lr: {device: cpu, num_cpus: 2, num_gpus: 0.0}\n"  # only one model
    )
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        f.write(incomplete)
        p = f.name
    try:
        resources.load_resources(path=p)
    except ValueError as e:
        assert "missing" in str(e).lower()
        return
    finally:
        os.unlink(p)
    raise AssertionError("expected ValueError for incomplete resource config")


def test_model_missing_key_raises():
    """A model missing num_gpus must raise, not silently default it."""
    bad = (
        "gpu_fraction_default: 1.0\n"
        "object_store_memory: 1\n"
        "threads_per_actor: 1\n"
        "min_vram_gib: 4\n"
        "xgboost: {device: cpu, tree_method: hist}\n"
        "models:\n"
        + "".join(
            f"  {m}: {{device: cpu, num_cpus: 2, num_gpus: 0.0}}\n"
            for m in resources.MODELS if m != "ffd"
        )
        + "  ffd: {device: gpu, num_cpus: 4}\n"  # ffd missing num_gpus
    )
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        f.write(bad)
        p = f.name
    try:
        resources.load_resources(path=p)
    except ValueError as e:
        assert "num_gpus" in str(e)
        return
    finally:
        os.unlink(p)
    raise AssertionError("expected ValueError for model missing a required key")


def test_real_config_loads_and_resolves():
    """The shipped sweep_resources.yaml loads and CPU-resolves correctly."""
    cfg = resources.load_resources()  # default path
    assert set(resources.MODELS).issubset(cfg["models"])
    # CPU fallback zeroes GPU for a GPU model.
    r = resources.for_model("ffd", gpu_available=False)
    assert r["num_gpus"] == 0.0 and r["device"] == "cpu"
    # LR is CPU-only regardless of gpu_available.
    r = resources.for_model("lr", gpu_available=True)
    assert r["num_gpus"] == 0.0
    # XGBoost device downgrades to cpu without a GPU.
    assert resources.xgboost_params(gpu_available=False)["device"] == "cpu"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nAll {len(fns)} resource-required tests passed.")
