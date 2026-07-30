"""Memory preflight reproduces the observed FedXGBllr OOM boundary.

The peak is sized from the LARGEST client partition, not the mean, because under
Dirichlet the largest client holds several× the mean. Observed at 5 concurrent
actors on a 15 GB box:

  * PaySim (890k/client)        → OOM   (also OOMs sequential? no — 1 actor fits)
  * BAF iid (140k/client)       → fits (17 rounds completed)
  * BAF α=0.5 (391k largest)    → OOM  (deterministic, both attempts)
  * ULB (≈40k/client)           → fits

The preflight must warn on the OOM cells (including the Dirichlet one the mean
would have passed) and clear the rest. Runnable via ``pytest`` or ``python``.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from experiments import sweep_preflight as P  # noqa: E402

_GIB = 1024 ** 3
_BOX = 15 * _GIB  # the box that OOM'd


def _over(max_partition_rows, feat, nconc, model="fedxgbllr"):
    _peak, _avail, over, _msg = P.memory_preflight(
        model=model, max_partition_rows=max_partition_rows, n_features=feat,
        n_concurrent=nconc, client_num=5, n_estimators_client=50,
        available_bytes=_BOX,
    )
    return over


def test_paysim_5concurrent_warns():
    """PaySim (≈890k largest partition) at 5 concurrent actors must warn."""
    assert _over(890766, 13, 5) is True


def test_paysim_sequential_ok():
    """PaySim at 1 concurrent (--gpu-fraction 1.0) must NOT warn."""
    assert _over(890766, 13, 1) is False


def test_baf_dirichlet_warns_where_mean_would_pass():
    """BAF α=0.5 largest partition (391k) OOMs at 5 concurrent — the case the
    mean (140k) passed. BAF iid (140k largest) fits the same box."""
    assert _over(391355, 55, 5) is True   # Dirichlet peak actor
    assert _over(140000, 55, 5) is False  # iid: max == mean


def test_ulb_5concurrent_ok():
    assert _over(39872, 30, 5) is False


def test_footprint_scales_with_partition_rows():
    small = P.estimate_fl_actor_bytes("fedxgbllr", 100_000, 13)
    big = P.estimate_fl_actor_bytes("fedxgbllr", 1_000_000, 13)
    assert big > 5 * small  # ~linear in partition rows (margin tensor dominates)


def test_unknown_ram_never_warns():
    """When RAM can't be determined (0), the check is skipped, not a false alarm."""
    _p, _a, over, _m = P.memory_preflight(
        model="fedxgbllr", max_partition_rows=890766, n_features=13,
        n_concurrent=5, available_bytes=0,
    )
    assert over is False


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nAll {len(fns)} memory-preflight tests passed.")
