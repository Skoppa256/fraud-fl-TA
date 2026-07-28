"""Bank Account Fraud (BAF) preprocessing pipeline.

Feedzai's BAF suite (Jesus et al., NeurIPS 2022). This module consumes the
``Base`` variant (``data/baf/baf.csv``) and produces the *identical output
interface* as :mod:`preprocessing.paysim` and :mod:`preprocessing.creditcard`
(same dict keys, same dtypes, same stratified 70/15/15 split semantics) so every
model and centralized baseline consumes it without any change to model logic.

BAF's path is PaySim-like (categorical encoding + scaling), not ULB-like. It is
also the only dataset in the study with real, named, semantically meaningful
features, which is why the (currently stubbed) SHAP analysis will carry its
interpretive weight here.

Preprocessing steps
-------------------
1. Load the raw CSV. Target column is ``fraud_bool`` (1,000,000 rows, ~1.10%
   fraud on Base).
2. Drop ``device_fraud_count`` — it is constant 0 across the whole dataset
   (zero variance; a known BAF quirk), so it carries no signal.
3. Hold out ``month`` as a NON-feature side column. BAF's fraud prevalence
   drifts across the eight months and our split is stratified-random (not
   temporal), so leaving ``month`` in the feature set would let a model learn
   month-level prevalence and apply it to same-month test rows — inflating
   AUPRC. It is returned under ``month_{train,val,test}`` for later temporal
   analysis without a re-run, but is never fed to a model.
4. Missing-value sentinels: five columns encode "missing" as exactly ``-1``
   (see ``_SENTINEL_COLUMNS``). For each, add a binary ``<col>_missing``
   indicator, replace the ``-1`` with NaN, then median-impute using the TRAIN
   split only (no leakage). Genuinely-signed columns (``intended_balcon_amount``,
   ``velocity_6h``, ``credit_risk_score``) are left untouched — their negatives
   are real values, not sentinels.
5. One-hot encode the five categorical columns with FIXED category lists so the
   column set is byte-identical across every client and split (required for
   FedAvg parameter-vector alignment and stable SHAP feature names).
6. StandardScaler fit on the training split only, applied to the full feature
   matrix (mirrors the PaySim pipeline's whole-matrix scaling).
7. Stratified 70/15/15 train/val/test split with the same ``random_state``
   handling as PaySim/ULB.

Final feature width: 24 numeric + 5 missing-indicator + 26 one-hot = 50 + 5 =
**55** named features.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


_TARGET_COLUMN: str = "fraud_bool"
# Constant (all-zero) across BAF — dropped as a zero-variance feature.
_DROP_COLUMNS: Tuple[str, ...] = ("device_fraud_count",)
# Held out of the feature matrix (see step 3) but returned as a side column.
_SIDE_COLUMN: str = "month"

# Columns whose ONLY negative value is exactly -1, which BAF uses as a "missing"
# sentinel. Verified against the data: each has nunique(negatives) == 1 and the
# underlying quantity cannot legitimately be negative. Handled via
# indicator + train-median imputation.
_SENTINEL_COLUMNS: Tuple[str, ...] = (
    "prev_address_months_count",     # ~71.3% missing
    "bank_months_count",             # ~25.4% missing
    "current_address_months_count",  # ~0.4% missing
    "session_length_in_minutes",     # ~0.2% missing
    "device_distinct_emails_8w",     # ~0.02% missing
)

# Fixed category lists (frozen from the observed cardinalities of Base) so that
# one-hot columns are identical across clients and splits.
_CATEGORICAL_CATEGORIES: Dict[str, Tuple[str, ...]] = {
    "payment_type": ("AA", "AB", "AC", "AD", "AE"),
    "employment_status": ("CA", "CB", "CC", "CD", "CE", "CF", "CG"),
    "housing_status": ("BA", "BB", "BC", "BD", "BE", "BF", "BG"),
    "source": ("INTERNET", "TELEAPP"),
    "device_os": ("linux", "macintosh", "other", "windows", "x11"),
}


class BAFPreprocessor:
    """Stateful BAF preprocessor.

    Parameters
    ----------
    data_path:
        Path to ``baf.csv`` (the Base variant).
    random_state:
        Seed used for both train/val/test splits. Defaults to 42.
    """

    def __init__(self, data_path: str, random_state: int = 42) -> None:
        self.data_path: str = data_path
        self.random_state: int = random_state
        self.scaler: StandardScaler | None = None
        self.feature_names: List[str] = []

    def load_and_split(self) -> Dict[str, object]:
        """Run the full pipeline and return all splits.

        Returns
        -------
        dict
            Keys:
            ``x_train`` / ``y_train`` / ``x_val`` / ``y_val`` / ``x_test`` /
            ``y_test`` (numpy arrays), ``feature_names`` (list of str),
            ``scaler`` (the fitted StandardScaler), and the non-feature side
            arrays ``month_train`` / ``month_val`` / ``month_test``.
        """
        print(f"[baf] loading {self.data_path}")
        df = pd.read_csv(self.data_path)
        n_total = len(df)
        print(f"[baf] loaded {n_total:,} rows, {df.shape[1]} columns")

        y_arr = df[_TARGET_COLUMN].to_numpy()
        month_arr = df[_SIDE_COLUMN].to_numpy()

        feat_df = self._build_feature_frame(df)
        self.feature_names = list(feat_df.columns)

        # to_numpy keeps NaN in the sentinel columns; imputation happens after
        # the split so the fill value is a train-only statistic.
        x_all = feat_df.to_numpy(dtype=np.float32)

        (
            x_train, x_val, x_test,
            y_train, y_val, y_test,
            month_train, month_val, month_test,
        ) = self._stratified_split(x_all, y_arr, month_arr)

        x_train, x_val, x_test = self._impute_sentinels(x_train, x_val, x_test)
        x_train, x_val, x_test = self._fit_and_apply_scaler(x_train, x_val, x_test)

        y_train = y_train.astype(np.int32)
        y_val = y_val.astype(np.int32)
        y_test = y_test.astype(np.int32)

        self._print_summary(
            n_total=n_total,
            x_train=x_train, x_val=x_val, x_test=x_test,
            y_train=y_train, y_val=y_val, y_test=y_test,
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
            # Non-feature side column (never fed to a model), aligned per split.
            "month_train": month_train,
            "month_val": month_val,
            "month_test": month_test,
        }

    def _build_feature_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop non-features, add missing-indicators, NaN the sentinels, one-hot.

        Returns a DataFrame whose columns are, in fixed order:
        ``<24 numeric> + <5 missing-indicator> + <26 one-hot>``. Sentinel
        columns still hold NaN where the raw value was ``-1``; imputation is
        deferred to :meth:`_impute_sentinels` (train-only).
        """
        df = df.drop(columns=[_TARGET_COLUMN, _SIDE_COLUMN, *_DROP_COLUMNS])

        categorical = list(_CATEGORICAL_CATEGORIES.keys())
        numeric_cols = [c for c in df.columns if c not in categorical]

        # Missing-indicator columns (float 0/1), then NaN the sentinel values.
        flag_cols: List[str] = []
        numeric_frame = df[numeric_cols].copy()
        for col in _SENTINEL_COLUMNS:
            flag = f"{col}_missing"
            numeric_frame[flag] = (numeric_frame[col] == -1).astype(np.float32)
            numeric_frame.loc[numeric_frame[col] == -1, col] = np.nan
            flag_cols.append(flag)

        # One-hot encode categoricals with fixed category lists.
        dummy_frames: List[pd.DataFrame] = []
        for col, cats in _CATEGORICAL_CATEGORIES.items():
            cat = pd.Categorical(df[col], categories=list(cats))
            dummy_frames.append(
                pd.get_dummies(cat, prefix=col, drop_first=False, dtype=np.float32)
            )

        # Fixed column order: numeric (24) → missing flags (5) → one-hot (26).
        ordered = pd.concat(
            [numeric_frame[numeric_cols], numeric_frame[flag_cols], *dummy_frames],
            axis=1,
        )
        return ordered

    def _sentinel_indices(self) -> List[int]:
        """Positions of the sentinel columns in ``feature_names``."""
        return [self.feature_names.index(c) for c in _SENTINEL_COLUMNS]

    def _stratified_split(
        self, x: np.ndarray, y: np.ndarray, month: np.ndarray
    ) -> Tuple[np.ndarray, ...]:
        """Stratified 70/15/15 split carried out on row indices.

        Splitting on indices (rather than on ``x`` directly) keeps the ``month``
        side column aligned to each split and lets imputation compute a
        train-only median.
        """
        idx = np.arange(len(y))
        idx_train, idx_temp = train_test_split(
            idx, test_size=0.30, stratify=y, random_state=self.random_state
        )
        idx_val, idx_test = train_test_split(
            idx_temp, test_size=0.50, stratify=y[idx_temp],
            random_state=self.random_state,
        )
        return (
            x[idx_train], x[idx_val], x[idx_test],
            y[idx_train], y[idx_val], y[idx_test],
            month[idx_train], month[idx_val], month[idx_test],
        )

    def _impute_sentinels(
        self, x_train: np.ndarray, x_val: np.ndarray, x_test: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Median-impute the sentinel columns using TRAIN medians only."""
        x_train, x_val, x_test = x_train.copy(), x_val.copy(), x_test.copy()
        for j in self._sentinel_indices():
            median = float(np.nanmedian(x_train[:, j]))
            for arr in (x_train, x_val, x_test):
                col = arr[:, j]
                col[np.isnan(col)] = median
        return x_train, x_val, x_test

    def _fit_and_apply_scaler(
        self, x_train: np.ndarray, x_val: np.ndarray, x_test: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Standardize the full feature matrix; scaler fit on train only."""
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
        print("\n[baf] === data summary ===")
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
        print("[baf] === end summary ===\n")


def load_baf(
    data_path: str = "data/baf/baf.csv",
    random_state: int = 42,
) -> Dict[str, object]:
    """Convenience wrapper around :class:`BAFPreprocessor`.

    Mirrors :func:`preprocessing.paysim.load_paysim` — same signature shape and
    same return dict (plus ``month_*`` side arrays), so it is a drop-in for any
    model's data-loading call::

        from preprocessing.baf import load_baf
        data = load_baf()
        x_train, y_train = data["x_train"], data["y_train"]
    """
    prep = BAFPreprocessor(data_path=data_path, random_state=random_state)
    return prep.load_and_split()
