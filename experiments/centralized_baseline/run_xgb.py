"""Centralized baseline — XGBoost Classifier.

Represents the centralized upper bound for FedXGBllr.
FedXGBllr internally uses XGBClassifier on each client — this
script trains a single XGBClassifier on the full dataset.

Methodological note:
    Uses XGBClassifier with the same hyperparameters as
    models/fedxgbllr/ (n_estimators=50, max_depth=6,
    learning_rate=0.1, subsample=0.8). FedXGBllr distributes
    50 trees per client across K=5 clients and aggregates via
    a 1D CNN. Here, a single model trains 50 trees on the full
    data. This is the centralized upper bound for tree-based FL.

    Global oversampling (when enabled via --oversampling smote|adasyn)
    is applied once on the full x_train using imblearn
    (sampling_strategy="auto", k_neighbors=5 for SMOTE,
    n_neighbors=5 for ADASYN). This represents the upper bound for
    imbalance handling.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
from imblearn.over_sampling import ADASYN, SMOTE
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)
from xgboost import XGBClassifier

from preprocessing.paysim import load_paysim


MODEL_NAME: str = "xgb"
OVERSAMPLING_CHOICES = ("smote", "adasyn", "none")


def _str2bool(v) -> bool:
    if isinstance(v, bool):
        return v
    s = str(v).lower()
    if s in ("yes", "true", "t", "y", "1"):
        return True
    if s in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError(f"boolean expected, got {v!r}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=f"Centralized baseline ({MODEL_NAME})"
    )
    p.add_argument(
        "--oversampling",
        choices=list(OVERSAMPLING_CHOICES),
        default="smote",
        help="Global oversampler applied to x_train (smote | adasyn | none).",
    )
    p.add_argument("--random_seed", type=int, default=42)
    p.add_argument("--use_wandb", type=_str2bool, default=False)
    p.add_argument(
        "--wandb_project", type=str, default="hfedxgboost-paysim"
    )
    return p.parse_args()


def _apply_global_oversampling(
    x_train: np.ndarray, y_train: np.ndarray, seed: int, method: str
) -> tuple[np.ndarray, np.ndarray, bool, str]:
    """Apply global SMOTE/ADASYN once on full x_train. Returns (x, y, applied, note)."""
    if method == "none":
        return x_train, y_train, False, "oversampling disabled"
    if method == "smote":
        sampler = SMOTE(sampling_strategy="auto", k_neighbors=5, random_state=seed)
    elif method == "adasyn":
        sampler = ADASYN(
            sampling_strategy="auto", n_neighbors=5, random_state=seed
        )
    else:
        raise ValueError(f"unknown oversampling method: {method!r}")
    try:
        x_res, y_res = sampler.fit_resample(x_train, y_train)
    except ValueError as exc:
        return (
            x_train,
            y_train,
            False,
            f"{method.upper()} failed: {exc}; falling back to raw data",
        )
    return (
        x_res.astype(np.float32, copy=False),
        y_res.astype(np.int32, copy=False),
        True,
        "",
    )


def _fraud_ratio(y: np.ndarray) -> float:
    return float((y == 1).sum()) / max(len(y), 1)


def _metrics(y_true: np.ndarray, scores: np.ndarray, preds: np.ndarray) -> dict:
    return {
        "auprc": float(average_precision_score(y_true, scores)),
        "f1": float(f1_score(y_true, preds, zero_division=0)),
        "precision": float(precision_score(y_true, preds, zero_division=0)),
        "recall": float(recall_score(y_true, preds, zero_division=0)),
    }


def main() -> None:
    args = _parse_args()
    seed = int(args.random_seed)

    data = load_paysim(random_state=seed)
    x_train, y_train = data["x_train"], data["y_train"]
    x_val, y_val = data["x_val"], data["y_val"]
    x_test, y_test = data["x_test"], data["y_test"]

    ratio_before = _fraud_ratio(y_train)

    print(f"=== CENTRALIZED BASELINE — {MODEL_NAME.upper()} ===")
    print(
        f"Dataset: PaySim | x_train: {x_train.shape} | "
        f"fraud ratio: {ratio_before * 100:.4f}%"
    )

    oversampling = str(args.oversampling).lower()
    x_train, y_train, applied, note = _apply_global_oversampling(
        x_train, y_train, seed, oversampling
    )
    if applied:
        ratio_after = _fraud_ratio(y_train)
        print(
            f"Oversampling: {oversampling.upper()} | "
            f"x_train after: {x_train.shape} | "
            f"fraud ratio: {ratio_after * 100:.2f}%"
        )
    else:
        print(f"Oversampling: {oversampling} | x_train: {x_train.shape} | {note}")

    wandb_run = None
    if bool(args.use_wandb):
        import wandb

        run_name = f"centralized_{MODEL_NAME}_seed{seed}"
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=run_name,
            config={
                "model": MODEL_NAME,
                "oversampling": oversampling,
                "random_seed": seed,
                "n_estimators": 50,
                "max_depth": 6,
                "learning_rate": 0.1,
                "subsample": 0.8,
            },
        )

    model = XGBClassifier(
        n_estimators=50,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        objective="binary:logistic",
        eval_metric="aucpr",
        random_state=seed,
        use_label_encoder=False,
    )

    t0 = time.time()
    model.fit(x_train, y_train)
    train_time = time.time() - t0

    val_scores = model.predict_proba(x_val)[:, 1]
    val_preds = model.predict(x_val)
    v = _metrics(y_val, val_scores, val_preds)

    test_scores = model.predict_proba(x_test)[:, 1]
    test_preds = model.predict(x_test)
    t = _metrics(y_test, test_scores, test_preds)

    print(
        f"[VAL]  auprc={v['auprc']:.4f} | f1={v['f1']:.4f} | "
        f"precision={v['precision']:.4f} | recall={v['recall']:.4f}"
    )
    print(
        f"[TEST] auprc={t['auprc']:.4f} | f1={t['f1']:.4f} | "
        f"precision={t['precision']:.4f} | recall={t['recall']:.4f}"
    )
    print(f"Training time: {train_time:.2f}s")

    if wandb_run is not None:
        wandb_run.log(
            {
                "val_auprc": v["auprc"],
                "val_f1": v["f1"],
                "val_precision": v["precision"],
                "val_recall": v["recall"],
                "test_auprc": t["auprc"],
                "test_f1": t["f1"],
                "test_precision": t["precision"],
                "test_recall": t["recall"],
                "training_time_s": train_time,
            }
        )
        wandb_run.summary.update(
            {
                "val_auprc": v["auprc"],
                "test_auprc": t["auprc"],
                "test_f1": t["f1"],
                "test_precision": t["precision"],
                "test_recall": t["recall"],
                "training_time_s": train_time,
            }
        )
        wandb_run.finish()


if __name__ == "__main__":
    main()
