"""Dispatch wrapper that selects between SMOTE and ADASYN per client.

Every FL model run script (LR, SVM, GBM, FFD, BERT/FT-Transformer, and
FedXGBllr) and the centralized baselines need to apply one of
{SMOTE, ADASYN, none} to per-client partitions before training. This module
centralises that dispatch so each script only needs a single call.
"""

from __future__ import annotations

from typing import Any, Dict, List

from preprocessing.smote import apply_smote_to_all_clients
from preprocessing.adasyn import apply_adasyn_to_all_clients


VALID_METHODS = ("smote", "adasyn", "none")


def apply_oversampling_to_all_clients(
    clients: List[Dict[str, Any]],
    method: str,
    k_neighbors: int = 5,
    sampling_strategy: str | float = "auto",
    base_seed: int = 42,
) -> List[Dict[str, Any]]:
    """Apply the configured oversampler to every client partition.

    Parameters
    ----------
    clients:
        List of client records returned by ``get_partition``.
    method:
        One of ``"smote"``, ``"adasyn"``, ``"none"`` (case-insensitive).
        ``"none"`` skips oversampling but still attaches the standard
        ``*_applied=False`` / ``n_samples_after`` fields so downstream code
        does not need to special-case it.
    k_neighbors:
        Forwarded as ``k_neighbors`` for SMOTE and ``n_neighbors`` for ADASYN.
    sampling_strategy, base_seed:
        Forwarded to the underlying oversampler unchanged.
    """
    m = (method or "none").lower()
    if m not in VALID_METHODS:
        raise ValueError(
            f"oversampling method must be one of {VALID_METHODS!r}, got {method!r}"
        )

    if m == "smote":
        return apply_smote_to_all_clients(
            clients,
            enabled=True,
            sampling_strategy=sampling_strategy,
            k_neighbors=k_neighbors,
            base_seed=base_seed,
        )
    if m == "adasyn":
        return apply_adasyn_to_all_clients(
            clients,
            enabled=True,
            sampling_strategy=sampling_strategy,
            n_neighbors=k_neighbors,
            base_seed=base_seed,
        )
    # method == "none" — run SMOTE module with enabled=False to keep the
    # downstream contract (smote_applied, n_samples_after, ...) intact.
    return apply_smote_to_all_clients(
        clients,
        enabled=False,
        sampling_strategy=sampling_strategy,
        k_neighbors=k_neighbors,
        base_seed=base_seed,
    )
