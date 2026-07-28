"""Per-client local SMOTE for FL training.

Notes:
- ``enabled=False`` → SMOTE is skipped on every client
- ``n_fraud < k_neighbors + 1`` → SMOTE is skipped for that client and a warning is printed identifying the client and the shortfall
- ``sampling_strategy`` is a float *target* and a client already meets/exceeds it → SMOTE is skipped for that client (it only adds minority samples; forcing a lower ratio would require *removing* fraud, which raises ``ValueError``). This clamp keeps non-IID (Dirichlet) runs from crashing when a partition over-concentrates fraud.
- Otherwise → SMOTE oversamples the minority class to the requested ratio (``sampling_strategy="auto"`` → 1:1 with the majority class).
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
        (final ratio 1:1). A float is a minority:majority target; if a
        client already meets/exceeds it, SMOTE is skipped for that client
        (see module docstring) rather than raising ``ValueError``.
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
    n_majority = n_samples - n_fraud
    min_required = k_neighbors + 1

    # skip_category is a machine-readable reason so the ablation analysis can
    # tell the two skips apart — they mean OPPOSITE things: "insufficient_minority"
    # is a sparse client that could NOT be oversampled, while "target_met" is a
    # dense client that did NOT need it. Collapsing them would misread the
    # ablation. None here means SMOTE actually ran.
    skip_reason: str | None = None
    skip_category: str | None = None
    if not enabled:
        skip_reason = "SMOTE disabled (ablation arm)"
        skip_category = "disabled"
    elif n_fraud < min_required:
        skip_reason = (
            f"insufficient fraud samples (have {n_fraud}, "
            f"need >= {min_required} = k_neighbors+1)"
        )
        skip_category = "insufficient_minority"
        print(
            f"[smote] WARN client {client_id}: skipping SMOTE — {skip_reason}"
        )
    elif isinstance(sampling_strategy, float) and n_fraud >= sampling_strategy * n_majority:
        # Float sampling_strategy is a minority:majority TARGET. SMOTE only
        # adds minority points, so a client already at/above the target needs
        # no oversampling — forcing it would require removing fraud, which
        # imblearn rejects with ValueError. Common under non-IID (Dirichlet)
        # partitions that concentrate fraud onto a few clients.
        current_ratio = (n_fraud / n_majority) if n_majority > 0 else float("inf")
        skip_reason = (
            f"target already met (minority:majority {current_ratio:.4f} "
            f">= target {sampling_strategy})"
        )
        skip_category = "target_met"
        print(
            f"[smote] client {client_id}: skipping SMOTE — {skip_reason}"
        )

    if skip_reason is not None:
        out["x"] = x.astype(np.float32, copy=False)
        out["y"] = y.astype(np.int32, copy=False)
        out["smote_applied"] = False
        out["skip_reason"] = skip_category
        out["n_samples_after"] = n_samples
        out["n_fraud_after"] = n_fraud
        out["fraud_ratio_after"] = (
            float(n_fraud / n_samples) if n_samples > 0 else 0.0
        )
        # No synthesis on a skipped client — record zeros so the ablation
        # analysis can tell "SMOTE off here" apart from "SMOTE ran".
        out["n_real_minority"] = n_fraud
        out["n_synthetic"] = 0
        out["synthesis_multiplier"] = 0.0
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

    # Synthesis multiplier: how many synthetic minority points were interpolated
    # per real minority point on this client. This is the direct empirical
    # evidence for the "unrepresentative synthesis" concern (§2.2.6): under a
    # Dirichlet partition a client can hold only a handful of real fraud rows,
    # so a high multiplier means many synthetic points were interpolated among
    # very few real ones. Logged per client (and per round for FFD, which
    # re-SMOTEs each round). Bab 4 quantifies this.
    n_synthetic = n_fraud_after - n_fraud
    synthesis_multiplier = (
        float(n_synthetic / n_fraud) if n_fraud > 0 else 0.0
    )
    print(
        f"[smote] client {client_id}: applied — "
        f"{n_synthetic:,} synthetic from {n_fraud:,} real minority "
        f"(x{synthesis_multiplier:.1f} synthesis multiplier)"
    )

    out["x"] = x_res
    out["y"] = y_res
    out["smote_applied"] = True
    out["skip_reason"] = None
    out["n_samples_after"] = n_total_after
    out["n_fraud_after"] = n_fraud_after
    out["fraud_ratio_after"] = float(n_fraud_after / n_total_after)
    out["n_real_minority"] = n_fraud
    out["n_synthetic"] = n_synthetic
    out["synthesis_multiplier"] = synthesis_multiplier
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
    """Print a compact per-client SMOTE summary table.

    Includes the per-client synthesis multiplier (synthetic:real minority) and
    the ``applied`` flag. Under non-IID (Dirichlet) partitions "SMOTE on" is NOT
    uniform across clients — a client that draws a large share of the minority
    class can exceed the target locally and be skipped while its peers
    oversample. The trailing "applied N/K clients" line makes that explicit so
    the ablation is not misread as a clean on/off across all clients.
    """
    header = (
        f"  {'cid':>3} | {'n_before':>11} | {'n_after':>11} | "
        f"{'fraud_before':>13} | {'fraud_after':>12} | "
        f"{'ratio_after':>11} | {'synth_mult':>10} | {'applied':>7}"
    )
    sep = "-" * len(header)
    print("\n[smote] === per-client SMOTE summary ===")
    print(header)
    print(sep)
    n_applied = 0
    n_skip_insufficient = 0
    n_skip_target = 0
    n_skip_disabled = 0
    for c in clients:
        n_applied += int(bool(c.get("smote_applied")))
        cat = c.get("skip_reason")
        n_skip_insufficient += int(cat == "insufficient_minority")
        n_skip_target += int(cat == "target_met")
        n_skip_disabled += int(cat == "disabled")
        print(
            f"  {c['client_id']:>3} | "
            f"{c['n_samples']:>11,} | "
            f"{c['n_samples_after']:>11,} | "
            f"{c['n_fraud']:>13,} | "
            f"{c['n_fraud_after']:>12,} | "
            f"{c['fraud_ratio_after'] * 100:>10.4f}% | "
            f"{'x' + format(c.get('synthesis_multiplier', 0.0), '.1f'):>10} | "
            f"{str(c['smote_applied']):>7}"
        )
    print(sep)
    # Report the two skip reasons SEPARATELY — "insufficient_minority" (a sparse
    # client that could not be oversampled) and "target_met" (a dense client that
    # did not need it) mean opposite things for the ablation.
    print(
        f"[smote] applied on {n_applied}/{len(clients)} clients | "
        f"skipped: {n_skip_insufficient} insufficient-minority, "
        f"{n_skip_target} target-already-met"
        + (f", {n_skip_disabled} disabled" if n_skip_disabled else "")
    )
    print("[smote] === end summary ===\n")
