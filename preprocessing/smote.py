"""Per-client local SMOTE for FL training.

Notes:
- ``enabled=False`` → SMOTE is skipped on every client
- ``n_fraud < k_neighbors + 1`` → SMOTE is skipped for that client and a warning is printed identifying the client and the shortfall
- Otherwise → SMOTE oversamples the minority class to a 1:1 ratio with the majority class (``sampling_strategy="auto"``).
"""

from __future__ import annotations

from typing import Any, Dict, List, Union

import numpy as np
from imblearn.over_sampling import SMOTE


def apply_smote(
    client_data: Dict[str, Any],
    enabled: bool = True,
    sampling_strategy: Union[float, str] = "auto",
    k_neighbors: int = 5,
    base_seed: int = 42,
) -> Dict[str, Any]:
    """Apply local SMOTE to a single client's data partition.

    Parameters
    ----------
    client_data:
        Dict as returned by ``partitioning.dirichlet.get_partition``,
        containing ``x``, ``y``, ``client_id``, ``n_samples``,
        ``n_fraud``, ``fraud_ratio``.
    enabled:
        Master switch — if ``False``, SMOTE is skipped on every call
        (the "without SMOTE" ablation condition).
    sampling_strategy:
        Passed through to :class:`imblearn.over_sampling.SMOTE`.
        ``"auto"`` resamples the minority to match the majority count
        (final ratio 1:1).
    k_neighbors:
        SMOTE neighbour count. The safety guard requires at least
        ``k_neighbors + 1`` minority samples per client.
    base_seed:
        SMOTE ``random_state`` is set to ``base_seed + client_id`` so
        each client is reproducible *and* uses a distinct seed.

    Returns
    -------
    dict
        A new dict (input is not mutated) with the original keys plus:

        * ``smote_applied``    — ``True`` iff SMOTE actually ran.
        * ``n_samples_after``  — total samples after resampling.
        * ``n_fraud_after``    — fraud samples after resampling.
        * ``fraud_ratio_after``— fraud ratio after resampling.

        ``x`` is float32 and ``y`` is int32 in all cases.
    """
    out: Dict[str, Any] = dict(client_data)
    client_id = int(out["client_id"])
    x: np.ndarray = out["x"]
    y: np.ndarray = out["y"]
    n_samples = int(len(y))
    n_fraud = int((y == 1).sum())
    min_required = k_neighbors + 1

    skip_reason: str | None = None
    if not enabled:
        skip_reason = "SMOTE disabled (ablation arm)"
    elif n_fraud < min_required:
        skip_reason = (
            f"insufficient fraud samples (have {n_fraud}, "
            f"need >= {min_required} = k_neighbors+1)"
        )
        print(
            f"[smote] WARN client {client_id}: skipping SMOTE — {skip_reason}"
        )

    if skip_reason is not None:
        out["x"] = x.astype(np.float32, copy=False)
        out["y"] = y.astype(np.int32, copy=False)
        out["smote_applied"] = False
        out["n_samples_after"] = n_samples
        out["n_fraud_after"] = n_fraud
        out["fraud_ratio_after"] = (
            float(n_fraud / n_samples) if n_samples > 0 else 0.0
        )
        return out

    smote = SMOTE(
        sampling_strategy=sampling_strategy,
        k_neighbors=k_neighbors,
        random_state=base_seed + client_id,
    )
    x_res, y_res = smote.fit_resample(x, y)

    x_res = x_res.astype(np.float32, copy=False)
    y_res = y_res.astype(np.int32, copy=False)
    n_total_after = int(len(y_res))
    n_fraud_after = int((y_res == 1).sum())

    out["x"] = x_res
    out["y"] = y_res
    out["smote_applied"] = True
    out["n_samples_after"] = n_total_after
    out["n_fraud_after"] = n_fraud_after
    out["fraud_ratio_after"] = float(n_fraud_after / n_total_after)
    return out


def apply_smote_to_all_clients(
    clients: List[Dict[str, Any]],
    enabled: bool = True,
    sampling_strategy: Union[float, str] = "auto",
    k_neighbors: int = 5,
    base_seed: int = 42,
) -> List[Dict[str, Any]]:
    """Run :func:`apply_smote` on each client and print a summary table.

    Parameters mirror :func:`apply_smote`. ``clients`` is the list
    returned by ``get_partition``.

    Returns
    -------
    list of dict
        One updated client record per input client.
    """
    results: List[Dict[str, Any]] = [
        apply_smote(
            c,
            enabled=enabled,
            sampling_strategy=sampling_strategy,
            k_neighbors=k_neighbors,
            base_seed=base_seed,
        )
        for c in clients
    ]
    _print_smote_summary(results)
    return results


def _print_smote_summary(clients: List[Dict[str, Any]]) -> None:
    """Print a compact per-client SMOTE summary table."""
    header = (
        f"  {'cid':>3} | {'n_before':>11} | {'n_after':>11} | "
        f"{'fraud_before':>13} | {'fraud_after':>12} | "
        f"{'ratio_after':>11} | {'applied':>7}"
    )
    sep = "-" * len(header)
    print("\n[smote] === per-client SMOTE summary ===")
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
            f"{str(c['smote_applied']):>7}"
        )
    print("[smote] === end summary ===\n")
