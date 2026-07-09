"""ULB credit-card preprocessing pipeline (Kaggle mlg-ulb/creditcardfraud).

Parallel to :mod:`preprocessing.paysim`. Produces the *identical output
interface* (same dict keys, same dtypes, same split semantics) so every model
and centralized baseline can consume it without any change to model logic.

Key differences from PaySim (by design — the raw schema is already model-ready):
    - NO identifier columns to drop.
    - NO one-hot encoding (there is no categorical column).
    - NO balance-error feature engineering.
    - V1..V28 are already PCA components (zero-mean, unit-ish scale) — they are
      left UNTOUCHED. Only ``Time`` and ``Amount`` are standardized, with the
      StandardScaler fit on the training split only (no leakage).
    - Same stratified 70/15/15 train/val/test split and the same
      ``random_state`` handling as PaySim.

Raw schema (31 columns): ``Time, V1..V28, Amount, Class``.
    - 284,807 rows, 492 fraud (0.172%). Target column is ``Class``.
    - ``feature_names`` is returned as ``['Time','V1',...,'V28','Amount']`` so
      SHAP still keys on named columns (V1..V28 are anonymized — expected).
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


_TARGET_COLUMN: str = "Class"
# Only these two columns are on a raw scale; V1..V28 are already PCA-standardized.
_SCALE_COLUMNS: Tuple[str, ...] = ("Time", "Amount")


class CreditCardPreprocessor:
    """Stateful credit-card preprocessor.

    Parameters
    ----------
    data_path:
        Path to ``creditcard.csv``.
    random_state:
        Seed used for both train/val/test splits. Defaults to 42.
    """

    def __init__(self, data_path: str, random_state: int = 42) -> None:
        self.data_path: str = data_path
        self.random_state: int = random_state
        self.scaler: StandardScaler | None = None
        self.feature_names: List[str] = []

    def load_and_split(self) -> Dict[str, object]:
        """Run the pipeline and return all splits.

        Returns
        -------
        dict
            Keys:
            ``x_train`` (numpy arrays)
            ``y_train`` (numpy arrays)
            ``x_val`` (numpy arrays)
            ``y_val`` (numpy arrays)
            ``x_test`` (numpy arrays)
            ``y_test`` (numpy arrays)
            ``feature_names`` (list of str)
            ``scaler`` (the fitted StandardScaler)
        """
        print(f"[creditcard] loading {self.data_path}")
        df = pd.read_csv(self.data_path)
        n_total = len(df)
        print(f"[creditcard] loaded {n_total:,} rows, {df.shape[1]} columns")

        x_df, y_arr = self._split_features_and_target(df)
        self.feature_names = list(x_df.columns)

        x_train, x_val, x_test, y_train, y_val, y_test = self._stratified_split(
            x_df.to_numpy(dtype=np.float32), y_arr
        )

        x_train, x_val, x_test = self._fit_and_apply_scaler(x_train, x_val, x_test)

        y_train = y_train.astype(np.int32)
        y_val = y_val.astype(np.int32)
        y_test = y_test.astype(np.int32)

        self._print_summary(
            n_total=n_total,
            x_train=x_train,
            x_val=x_val,
            x_test=x_test,
            y_train=y_train,
            y_val=y_val,
            y_test=y_test,
        )

        return {
            "x_train": x_train,
            "y_train": y_train,
            "x_val": x_val,
            "y_val": y_val,
            "x_test": x_test,
            "y_test": y_test,
            "feature_names": self.feature_names,
            "scaler": self.scaler,
        }

    @staticmethod
    def _split_features_and_target(
        df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        y = df[_TARGET_COLUMN].to_numpy()
        x = df.drop(columns=[_TARGET_COLUMN])
        return x, y

    def _stratified_split(
        self, x: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        x_train, x_temp, y_train, y_temp = train_test_split(
            x,
            y,
            test_size=0.30,
            stratify=y,
            random_state=self.random_state,
        )
        x_val, x_test, y_val, y_test = train_test_split(
            x_temp,
            y_temp,
            test_size=0.50,
            stratify=y_temp,
            random_state=self.random_state,
        )
        return x_train, x_val, x_test, y_train, y_val, y_test

    def _scale_column_indices(self) -> List[int]:
        """Column positions of ``Time`` and ``Amount`` in ``feature_names``."""
        return [self.feature_names.index(c) for c in _SCALE_COLUMNS]

    def _fit_and_apply_scaler(
        self,
        x_train: np.ndarray,
        x_val: np.ndarray,
        x_test: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Standardize ONLY Time and Amount; V1..V28 pass through untouched.

        The scaler is fit on the training split only (no leakage), matching the
        PaySim pipeline's train-only fit.
        """
        idx = self._scale_column_indices()
        self.scaler = StandardScaler()
        x_train = x_train.copy()
        x_val = x_val.copy()
        x_test = x_test.copy()
        x_train[:, idx] = self.scaler.fit_transform(x_train[:, idx]).astype(np.float32)
        x_val[:, idx] = self.scaler.transform(x_val[:, idx]).astype(np.float32)
        x_test[:, idx] = self.scaler.transform(x_test[:, idx]).astype(np.float32)
        return (
            x_train.astype(np.float32),
            x_val.astype(np.float32),
            x_test.astype(np.float32),
        )

    def _print_summary(
        self,
        *,
        n_total: int,
        x_train: np.ndarray,
        x_val: np.ndarray,
        x_test: np.ndarray,
        y_train: np.ndarray,
        y_val: np.ndarray,
        y_test: np.ndarray,
    ) -> None:
        """Print a single end-of-pipeline data summary."""
        print("\n[creditcard] === data summary ===")
        print(f"  total rows loaded : {n_total:,}")
        print(f"  feature count     : {len(self.feature_names)}")
        print(f"  feature names     : {self.feature_names}")
        print(
            f"  x_train: {x_train.shape}  dtype={x_train.dtype}"
            f"   |  y_train: {y_train.shape}  dtype={y_train.dtype}"
        )
        print(
            f"  x_val  : {x_val.shape}  dtype={x_val.dtype}"
            f"   |  y_val  : {y_val.shape}  dtype={y_val.dtype}"
        )
        print(
            f"  x_test : {x_test.shape}  dtype={x_test.dtype}"
            f"   |  y_test : {y_test.shape}  dtype={y_test.dtype}"
        )

        for name, y in (("train", y_train), ("val", y_val), ("test", y_test)):
            n_pos = int(y.sum())
            ratio_pct = n_pos / len(y) * 100
            print(
                f"  fraud in {name:<5}: {n_pos:,} / {len(y):,}"
                f"  ({ratio_pct:.4f}%)"
            )
        print("[creditcard] === end summary ===\n")


def load_creditcard(
    data_path: str = "data/creditcard/creditcard.csv",
    random_state: int = 42,
) -> Dict[str, object]:
    """Convenience wrapper around :class:`CreditCardPreprocessor`.

    Mirrors :func:`preprocessing.paysim.load_paysim` — same signature shape and
    same return dict, so it is a drop-in for any model's data-loading call::

        from preprocessing.creditcard import load_creditcard
        data = load_creditcard()
        x_train, y_train = data["x_train"], data["y_train"]
        x_val,   y_val   = data["x_val"],   data["y_val"]
        x_test,  y_test  = data["x_test"],  data["y_test"]
    """
    prep = CreditCardPreprocessor(data_path=data_path, random_state=random_state)
    return prep.load_and_split()
