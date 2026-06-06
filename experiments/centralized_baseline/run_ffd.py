"""Centralized baseline — FFD (1D CNN, Yang et al. 2019).

Centralized upper bound for the FFD federated pipeline: trains the same
:class:`FFDModel` once on the full ``x_train`` (with optional global
SMOTE/ADASYN), using the same SGD hyperparameters as the local FL clients
(lr=0.01, batch_size=80) but for ``--num_epochs`` epochs on all the data.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch
import torch.nn as nn
from imblearn.over_sampling import ADASYN, SMOTE
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader, TensorDataset

from models.ffd.model import FFDModel
from evaluation.results_writer import (
    build_centralized_run_name,
    write_centralized_results,
)
from preprocessing.paysim import load_paysim


MODEL_NAME: str = "ffd"
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
    p.add_argument(
        "--sampling_strategy",
        type=str,
        default="auto",
        help=(
            "Passed to imblearn's sampling_strategy. 'auto' = 1:1 fraud:non-fraud. "
            "A float sets the post-resample minority/majority ratio "
            "(e.g. 0.01 for 1:100 fraud:non-fraud)."
        ),
    )
    p.add_argument("--random_seed", type=int, default=42)
    p.add_argument("--num_epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=80)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--use_wandb", type=_str2bool, default=False)
    p.add_argument(
        "--wandb_project", type=str, default="fraud-fl-TA"
    )
    return p.parse_args()


def _parse_sampling_strategy(s: str):
    try:
        return float(s)
    except (TypeError, ValueError):
        return s


def _apply_global_oversampling(
    x_train: np.ndarray,
    y_train: np.ndarray,
    seed: int,
    method: str,
    sampling_strategy="auto",
) -> tuple[np.ndarray, np.ndarray, bool, str]:
    if method == "none":
        return x_train, y_train, False, "oversampling disabled"
    if method == "smote":
        sampler = SMOTE(
            sampling_strategy=sampling_strategy, k_neighbors=5, random_state=seed
        )
    elif method == "adasyn":
        sampler = ADASYN(
            sampling_strategy=sampling_strategy, n_neighbors=5, random_state=seed
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
    num_epochs = int(args.num_epochs)
    batch_size = int(args.batch_size)
    lr = float(args.lr)

    torch.manual_seed(seed)
    np.random.seed(seed)

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
    sampling_strategy = _parse_sampling_strategy(args.sampling_strategy)
    x_train, y_train, applied, note = _apply_global_oversampling(
        x_train, y_train, seed, oversampling, sampling_strategy=sampling_strategy
    )
    if applied:
        ratio_after = _fraud_ratio(y_train)
        print(
            f"Oversampling: {oversampling.upper()} | "
            f"sampling_strategy={sampling_strategy!r} | "
            f"x_train after: {x_train.shape} | "
            f"fraud ratio: {ratio_after * 100:.2f}%"
        )
    else:
        print(f"Oversampling: {oversampling} | x_train: {x_train.shape} | {note}")

    run_name = build_centralized_run_name(MODEL_NAME, oversampling, seed)
    wandb_run = None
    if bool(args.use_wandb):
        import wandb

        wandb_run = wandb.init(
            project=args.wandb_project,
            name=run_name,
            config={
                "model": MODEL_NAME,
                "oversampling": oversampling,
                "random_seed": seed,
                "num_epochs": num_epochs,
                "batch_size": batch_size,
                "lr": lr,
            },
        )

    n_features = int(x_train.shape[1])
    model = FFDModel(input_dim=n_features)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    dataset = TensorDataset(
        torch.from_numpy(x_train.astype(np.float32)),
        torch.from_numpy(y_train.astype(np.int64)),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    t0 = time.time()
    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for xb, yb in loader:
            xb, yb = xb.to(model.device), yb.to(model.device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())
            n_batches += 1
        avg_loss = epoch_loss / max(n_batches, 1)

        val_scores = model.predict_proba(x_val)[:, 1]
        val_preds = (val_scores >= 0.5).astype(np.int32)
        v = _metrics(y_val, val_scores, val_preds)

        print(
            f"[epoch {epoch:>3}/{num_epochs}] loss={avg_loss:.4f} | "
            f"val_auprc={v['auprc']:.4f} | val_f1={v['f1']:.4f} | "
            f"val_precision={v['precision']:.4f} | val_recall={v['recall']:.4f}"
        )
        if wandb_run is not None:
            wandb_run.log(
                {
                    "epoch": epoch,
                    "train_loss": avg_loss,
                    "val_auprc": v["auprc"],
                    "val_f1": v["f1"],
                    "val_precision": v["precision"],
                    "val_recall": v["recall"],
                }
            )

    train_time = time.time() - t0

    val_scores = model.predict_proba(x_val)[:, 1]
    val_preds = (val_scores >= 0.5).astype(np.int32)
    v = _metrics(y_val, val_scores, val_preds)

    test_scores = model.predict_proba(x_test)[:, 1]
    test_preds = (test_scores >= 0.5).astype(np.int32)
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

    write_centralized_results(
        model=MODEL_NAME,
        oversampling=oversampling,
        seed=seed,
        val_metrics={
            "val_auprc": v["auprc"],
            "val_f1": v["f1"],
            "val_precision": v["precision"],
            "val_recall": v["recall"],
        },
        test_metrics={
            "test_auprc": t["auprc"],
            "test_f1": t["f1"],
            "test_precision": t["precision"],
            "test_recall": t["recall"],
        },
        duration_seconds=train_time,
    )

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
