"""PaySim preprocessing pipeline shared by all four FL models.

Implements Level 1 of the partitioning scheme: 
    - load the raw CSV
    - drop irrelevant identifiers (CURRENT: 'isFlaggedFraud' dropped)
    - engineer balance-error features
    - one-hot encode the transaction type
    - fit a StandardScaler on the training split only
    - split stratified 70/15/15 train/val/test arrays

The output of this module is the input to Level 2 (client partitioning, which lives in ``partitioning/dirichlet.py``).
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


_DROP_COLUMNS: Tuple[str, ...] = ("nameOrig", "nameDest", "isFlaggedFraud")
_TYPE_CATEGORIES: Tuple[str, ...] = (
    "CASH_IN",
    "CASH_OUT",
    "DEBIT",
    "PAYMENT",
    "TRANSFER",
)
_TARGET_COLUMN: str = "isFraud"


class PaySimPreprocessor:
    """Stateful PaySim preprocessor.

    Parameters
    ----------
    data_path:
        Path to ``paysim.csv``.
    random_state:
        Seed used for both train/val/test splits. Defaults to 42.
    """

    def __init__(self, data_path: str, random_state: int = 42) -> None:
        self.data_path: str = data_path
        self.random_state: int = random_state
        self.scaler: StandardScaler | None = None
        self.feature_names: List[str] = []


    def load_and_split(self) -> Dict[str, object]:
        """Run the full 6-step pipeline and return all splits.

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
        print(f"[paysim] loading {self.data_path}")
        df = pd.read_csv(self.data_path)
        n_total = len(df)
        print(f"[paysim] loaded {n_total:,} rows, {df.shape[1]} columns")

        df = self._drop_identifier_columns(df)
        df = self._add_balance_error_features(df)
        df = self._one_hot_encode_type(df)

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
    def _drop_identifier_columns(df: pd.DataFrame) -> pd.DataFrame:
        return df.drop(columns=list(_DROP_COLUMNS))

    
    @staticmethod
    def _add_balance_error_features(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["errorBalanceOrig"] = (
            df["newbalanceOrig"] - df["oldbalanceOrg"] + df["amount"]
        )
        df["errorBalanceDest"] = (
            df["oldbalanceDest"] + df["amount"] - df["newbalanceDest"]
        )
        return df


    @staticmethod
    def _one_hot_encode_type(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["type"] = pd.Categorical(df["type"], categories=list(_TYPE_CATEGORIES))
        dummies = pd.get_dummies(
            df["type"], prefix="type", drop_first=False, dtype=np.float32
        )
        df = df.drop(columns=["type"])
        return pd.concat([df, dummies], axis=1)


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

    
    def _fit_and_apply_scaler(
        self,
        x_train: np.ndarray,
        x_val: np.ndarray,
        x_test: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        self.scaler = StandardScaler()
        x_train = self.scaler.fit_transform(x_train).astype(np.float32)
        x_val = self.scaler.transform(x_val).astype(np.float32)
        x_test = self.scaler.transform(x_test).astype(np.float32)
        return x_train, x_val, x_test

    
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
        print("\n[paysim] === data summary ===")
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
        print("[paysim] === end summary ===\n")


def load_paysim(
    data_path: str = "data/paysim/paysim.csv",
    random_state: int = 42,
) -> Dict[str, object]:
    """Convenience wrapper around :class:`PaySimPreprocessor`.

    All four FL models import this function as the canonical entry point::

        from preprocessing.paysim import load_paysim
        data = load_paysim()
        x_train, y_train = data["x_train"], data["y_train"]
        x_val,   y_val   = data["x_val"],   data["y_val"]
        x_test,  y_test  = data["x_test"],  data["y_test"]
    """
    prep = PaySimPreprocessor(data_path=data_path, random_state=random_state)
    return prep.load_and_split()
