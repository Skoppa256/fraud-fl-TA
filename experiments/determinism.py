"""Every random draw in a run comes from the run seed.

Three things made federated runs differ at the same seed (measured 2026-11-13):

* **Aggregation order.** Flower hands client results to ``aggregate_fit`` in the
  order they finish, which changes every round and every run. Floating-point
  addition is not associative, so the averaged model differs in its last bits, and
  iterative training amplifies that (federated LR: AUPRC 0.7557–0.7580 over five
  runs). :func:`in_client_order` sorts results by client id before any strategy
  averages them; FedAvg is mathematically order-invariant, so only the float
  summation order changes.
* **Unseeded generators.** FFD/BERT federated starting weights and FedXGBllr's CNN
  initialisation, row shuffles and XGBoost subsampling drew from generators no
  seed reached. Entry points call :func:`seed_everything`; per-round, per-client
  draws use :func:`derive_seed`, so they do not depend on which Ray worker
  process happens to run a client.
* **GPU kernels.** Some CUDA kernels are nondeterministic unless told otherwise;
  :func:`torch_deterministic` sets the switches. It warns rather than raises on an
  operation that has no deterministic implementation, so a run is never aborted
  for it — a repeat run of the same seed is what verifies determinism on a GPU box.
"""

from __future__ import annotations

import os
import random
from typing import Any, List, Sequence, Tuple

import numpy as np


def derive_seed(seed: int, *parts: int) -> int:
    """A 32-bit seed for one sub-stream, e.g. ``derive_seed(seed, round, cid)``.

    Built with numpy's SeedSequence, so it is stable across processes and Python
    versions (no ``hash()``, which PYTHONHASHSEED randomises).
    """
    return int(np.random.SeedSequence([int(seed), *map(int, parts)]).generate_state(1)[0])


def seed_everything(seed: int) -> None:
    """Seed Python's ``random``, numpy's global generator and torch (CPU + CUDA)."""
    random.seed(int(seed))
    np.random.seed(int(seed) % 2**32)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def torch_deterministic() -> None:
    """Ask torch for deterministic kernels in this process (call in every Ray actor).

    ``CUBLAS_WORKSPACE_CONFIG`` must be set before CUDA initialises; the sweep
    runner sets it in every child's environment, and this sets it as a default for
    a by-hand run.
    """
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def in_client_order(results: Sequence[Tuple[Any, Any]]) -> List[Tuple[Any, Any]]:
    """Flower ``(ClientProxy, FitRes)`` results sorted by client id.

    Raises if two results share a client id: every strategy here expects one
    result per client per round, and silently keeping both would double-weight it.
    """
    out = sorted(results, key=lambda r: int(r[0].cid))
    cids = [int(p.cid) for p, _ in out]
    if len(set(cids)) != len(cids):
        raise RuntimeError(f"duplicate client ids in one round's results: {cids}")
    return out
