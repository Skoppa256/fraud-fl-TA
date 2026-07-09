"""Dataset dispatcher — the single entry point every model uses to load data.

Adds a second dataset (``creditcard``) alongside PaySim without touching any
model logic. Each per-dataset preprocessor returns the identical interface
(``x_train/y_train, x_val/y_val, x_test/y_test, feature_names, scaler``), so
downstream code stays dataset-agnostic and reads the feature dimension at
runtime from ``x_train.shape[1]`` / ``feature_names``.

Backward compatibility: ``load_dataset("paysim", random_state=seed)`` calls
``load_paysim(random_state=seed)`` with its default data path — byte-identical
to the pre-change behaviour.
"""

from __future__ import annotations

from typing import Dict

DATASETS = ("paysim", "creditcard")
DEFAULT_DATASET = "paysim"


def load_dataset(
    name: str = DEFAULT_DATASET,
    random_state: int = 42,
) -> Dict[str, object]:
    """Load a named dataset and return the canonical split dict.

    Parameters
    ----------
    name:
        ``"paysim"`` (default) or ``"creditcard"``.
    random_state:
        Seed forwarded to the per-dataset stratified split.
    """
    key = str(name).lower()
    if key == "paysim":
        from preprocessing.paysim import load_paysim

        return load_paysim(random_state=random_state)
    if key == "creditcard":
        from preprocessing.creditcard import load_creditcard

        return load_creditcard(random_state=random_state)
    raise ValueError(
        f"unknown dataset {name!r}; expected one of {DATASETS}"
    )
