"""Level-2 client partitioning for the FL training set.

Partition Schemes:
    - ``partition_iid``       — uniform shuffle + K-way equal split.
    - ``partition_dirichlet`` — per-class Dirichlet(alpha) draw, distributes each class's indices across the K clients in proportion to the draw.

Operated only on ``x_train`` / ``y_train``. 
The server-side ``x_val`` / ``y_val`` and ``x_test`` / ``y_test`` arrays are never
partitioned — they stay on the server for GBM best-model selection, early stopping, and final evaluation.

Each client subset ``D_k`` is passed through downstream training as-is, with no further train/val split inside the client partition.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np


_VALID_SCHEMES = ("iid", "dirichlet")


def partition_iid(
    x_train: np.ndarray,
    y_train: np.ndarray,
    num_clients: int = 5,
    random_state: int = 42,
) -> List[dict]:
    """Uniform IID partition of training data across K clients.

    Shuffles all samples with the given seed and slices the permutation
    into K (approximately) equal contiguous chunks. Label distribution
    across clients is statistically uniform — this is the FL baseline
    that isolates protocol overhead from data heterogeneity.

    Returns
    -------
    list of dict
        ``K`` client records with keys:
        ``x``
        ``y``
        ``client_id``
        ``n_samples``
        ``n_fraud``
        ``fraud_ratio``
    """
    rng = np.random.default_rng(random_state)
    perm = rng.permutation(len(y_train))
    shards = np.array_split(perm, num_clients)

    # NEXT: local SMOTE applied per-client before model training
    return [_build_client_record(k, idx, x_train, y_train) for k, idx in enumerate(shards)]


def partition_dirichlet(
    x_train: np.ndarray,
    y_train: np.ndarray,
    alpha: float,
    num_clients: int = 5,
    random_state: int = 42,
) -> List[dict]:
    """Non-IID Dirichlet partition of training data across K clients.

    For each class ``c`` in ``y_train``, draws a proportion vector
    ``p ~ Dir(alpha)`` of length ``K`` and routes a corresponding slice
    of class-c indices to each client.
    - Smaller ``alpha`` concentrates each class on fewer clients (stronger heterogeneity)
    - Larger ``alpha`` spreads classes evenly (approaches IID)

    Parameters
    ----------
    alpha:
        Dirichlet concentration. Conventional values for this study:
        ``0.5`` (strong Non-IID), ``1.0`` (moderate), ``5.0`` (near-IID).

    Returns
    -------
    list of dict
        Same structure as :func:`partition_iid`.

    Notes
    -----
    Some clients may receive zero fraud samples under small ``alpha``
    given PaySim's ~0.13% positive rate. That outcome is *intentional*:
    SMOTE downstream is responsible for handling clients with no or
    very few minority samples.
    """
    rng = np.random.default_rng(random_state)
    classes = np.unique(y_train)
    per_client_chunks: List[List[np.ndarray]] = [[] for _ in range(num_clients)]

    for c in classes:
        class_idx = np.where(y_train == c)[0]
        rng.shuffle(class_idx)

        proportions = rng.dirichlet(alpha=np.full(num_clients, alpha))
        split_points = (np.cumsum(proportions) * len(class_idx)).astype(int)[:-1]
        groups = np.split(class_idx, split_points)

        for k, group in enumerate(groups):
            per_client_chunks[k].append(group)

    # NEXT: local SMOTE applied per-client before model training
    clients: List[dict] = []
    for k in range(num_clients):
        if per_client_chunks[k]:
            idx_k = np.concatenate(per_client_chunks[k])
        else:
            idx_k = np.array([], dtype=np.int64)
        rng.shuffle(idx_k)
        clients.append(_build_client_record(k, idx_k, x_train, y_train))
    return clients


def get_partition(
    x_train: np.ndarray,
    y_train: np.ndarray,
    scheme: str,
    alpha: Optional[float] = None,
    num_clients: int = 5,
    random_state: int = 42,
) -> List[dict]:
    """Unified entry point used by every FL model in the study.

    Parameters
    ----------
    scheme:
        ``"iid"`` or ``"dirichlet"``.
    alpha:
        Required when ``scheme == "dirichlet"``; ignored otherwise.

    Raises
    ------
    ValueError
        If ``scheme`` is not one of the supported options, or if
        ``scheme == "dirichlet"`` and ``alpha`` is ``None``.
    """
    if scheme == "iid":
        return partition_iid(
            x_train, y_train, num_clients=num_clients, random_state=random_state
        )
    if scheme == "dirichlet":
        if alpha is None:
            raise ValueError(
                "alpha must be provided when scheme='dirichlet' "
                "(e.g. alpha=0.5 for strong Non-IID)"
            )
        return partition_dirichlet(
            x_train,
            y_train,
            alpha=alpha,
            num_clients=num_clients,
            random_state=random_state,
        )
    raise ValueError(
        f"scheme must be one of {_VALID_SCHEMES!r}, got {scheme!r}"
    )


def _build_client_record(
    client_id: int,
    indices: np.ndarray,
    x_train: np.ndarray,
    y_train: np.ndarray,
) -> dict:
    """Slice the training arrays for one client and package the metadata."""
    x_k = x_train[indices]
    y_k = y_train[indices]
    n_samples = int(len(y_k))
    n_fraud = int(y_k.sum())
    fraud_ratio = float(n_fraud / n_samples) if n_samples > 0 else 0.0
    return {
        "x": x_k,
        "y": y_k,
        "client_id": client_id,
        "n_samples": n_samples,
        "n_fraud": n_fraud,
        "fraud_ratio": fraud_ratio,
    }
