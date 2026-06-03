"""Per-client local ADASYN for FL training.

Mirrors ``preprocessing/smote.py`` so the two oversamplers are interchangeable
behind a single ``--oversampling`` switch.

Notes:
- ``enabled=False`` → ADASYN is skipped on every client.
- ``n_fraud < n_neighbors + 1`` → ADASYN is skipped for that client and a
  warning is printed identifying the client and the shortfall.
- ADASYN can raise ``ValueError`` ("No samples will be generated...") when the
  density-based weighting yields no candidates (e.g., when minority samples
  are all surrounded by other minority samples). In that case the client's
  data is returned unchanged with ``adasyn_applied=False``.
- Otherwise → ADASYN oversamples the minority class to approximately 1:1 with
  the majority class (``sampling_strategy="auto"``).
"""

from __future__ import annotations

from typing import Any, Dict, List, Union

import numpy as np
from imblearn.over_sampling import ADASYN


def apply_adasyn(
    client_data: Dict[str, Any],
    enabled: bool = True,
    sampling_strategy: Union[float, str] = "auto",
    n_neighbors: int = 5,
    base_seed: int = 42,
) -> Dict[str, Any]:
    """Apply local ADASYN to a single client's data partition.

    Parameters
    ----------
    client_data:
        Dict as returned by ``partitioning.dirichlet.get_partition``,
        containing ``x``, ``y``, ``client_id``, ``n_samples``,
        ``n_fraud``, ``fraud_ratio``.
    enabled:
        Master switch — if ``False``, ADASYN is skipped on every call.
    sampling_strategy:
        Passed through to :class:`imblearn.over_sampling.ADASYN`.
    n_neighbors:
        ADASYN neighbour count. The safety guard requires at least
        ``n_neighbors + 1`` minority samples per client.
    base_seed:
        ADASYN ``random_state`` is set to ``base_seed + client_id``.

    Returns
    -------
    dict
        A new dict (input is not mutated) with the original keys plus:

        * ``adasyn_applied``    — ``True`` iff ADASYN actually ran.
        * ``n_samples_after``   — total samples after resampling.
        * ``n_fraud_after``     — fraud samples after resampling.
        * ``fraud_ratio_after`` — fraud ratio after resampling.

        ``x`` is float32 and ``y`` is int32 in all cases.
    """
    out: Dict[str, Any] = dict(client_data)
    client_id = int(out["client_id"])
    x: np.ndarray = out["x"]
    y: np.ndarray = out["y"]
    n_samples = int(len(y))
    n_fraud = int((y == 1).sum())
    min_required = n_neighbors + 1

    skip_reason: str | None = None
    if not enabled:
        skip_reason = "ADASYN disabled (ablation arm)"
    elif n_fraud < min_required:
        skip_reason = (
            f"insufficient fraud samples (have {n_fraud}, "
            f"need >= {min_required} = n_neighbors+1)"
        )
        print(
            f"[adasyn] WARN client {client_id}: skipping ADASYN — {skip_reason}"
        )

    if skip_reason is not None:
        return _no_op_result(out, x, y, n_samples, n_fraud)

    adasyn = ADASYN(
        sampling_strategy=sampling_strategy,
        n_neighbors=n_neighbors,
        random_state=base_seed + client_id,
    )
    try:
        x_res, y_res = adasyn.fit_resample(x, y)
    except ValueError as exc:
        print(
            f"[adasyn] WARN client {client_id}: ADASYN failed — {exc}; "
            f"returning client data unchanged"
        )
        return _no_op_result(out, x, y, n_samples, n_fraud)

    x_res = x_res.astype(np.float32, copy=False)
    y_res = y_res.astype(np.int32, copy=False)
    n_total_after = int(len(y_res))
    n_fraud_after = int((y_res == 1).sum())

    out["x"] = x_res
    out["y"] = y_res
    out["adasyn_applied"] = True
    out["n_samples_after"] = n_total_after
    out["n_fraud_after"] = n_fraud_after
    out["fraud_ratio_after"] = float(n_fraud_after / n_total_after)
    return out


def apply_adasyn_to_all_clients(
    clients: List[Dict[str, Any]],
    enabled: bool = True,
    sampling_strategy: Union[float, str] = "auto",
    n_neighbors: int = 5,
    base_seed: int = 42,
) -> List[Dict[str, Any]]:
    """Run :func:`apply_adasyn` on each client and print a summary table."""
    results: List[Dict[str, Any]] = [
        apply_adasyn(
            c,
            enabled=enabled,
            sampling_strategy=sampling_strategy,
            n_neighbors=n_neighbors,
            base_seed=base_seed,
        )
        for c in clients
    ]
    _print_adasyn_summary(results)
    return results


def _no_op_result(
    out: Dict[str, Any],
    x: np.ndarray,
    y: np.ndarray,
    n_samples: int,
    n_fraud: int,
) -> Dict[str, Any]:
    out["x"] = x.astype(np.float32, copy=False)
    out["y"] = y.astype(np.int32, copy=False)
    out["adasyn_applied"] = False
    out["n_samples_after"] = n_samples
    out["n_fraud_after"] = n_fraud
    out["fraud_ratio_after"] = (
        float(n_fraud / n_samples) if n_samples > 0 else 0.0
    )
    return out


def _print_adasyn_summary(clients: List[Dict[str, Any]]) -> None:
    """Print a compact per-client ADASYN summary table."""
    header = (
        f"  {'cid':>3} | {'n_before':>11} | {'n_after':>11} | "
        f"{'fraud_before':>13} | {'fraud_after':>12} | "
        f"{'ratio_after':>11} | {'applied':>7}"
    )
    sep = "-" * len(header)
    print("\n[adasyn] === per-client ADASYN summary ===")
    print(header)
    print(sep)
    for c in clients:
        print(
            f"  {c['client_id']:>3} | "
            f"{c['n_samples']:>11,} | "
            f"{c['n_samples_after']:>11,} | "
            f"{c['n_fraud']:>13,} | "
            f"{c['n_fraud_after']:>12,} | "
            f"{c['fraud_ratio_after'] * 100:>10.4f}% | "
            f"{str(c['adasyn_applied']):>7}"
        )
    print("[adasyn] === end summary ===\n")
