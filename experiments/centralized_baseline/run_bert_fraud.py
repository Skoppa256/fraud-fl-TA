"""Centralized baseline — BertFraud (FT-Transformer, Gorishniy et al. 2021).

Centralized upper bound for the bert_fraud federated pipeline: trains the same
:class:`BertFraudModel` once on the full ``x_train`` (with optional global
SMOTE/ADASYN), using AdamW with the same hyperparameters as the FL clients
(lr=0.001, weight_decay=1e-4, batch_size=64) for ``--num_epochs`` epochs.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch
import torch.nn as nn
from imblearn.over_sampling import ADASYN, SMOTE
from evaluation.metrics import (
    best_f1_threshold,
    metrics_at_threshold,
    tuned_metrics,
)
from torch.utils.data import DataLoader, TensorDataset

from models.bert_fraud.model import BertFraudModel
from evaluation.results_writer import (
    build_centralized_run_name,
    write_centralized_results,
)
from preprocessing.loader import DATASETS, load_dataset


MODEL_NAME: str = "bert_fraud"
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


def _parse_sampling_strategy(s: str):
    try:
        return float(s)
    except (TypeError, ValueError):
        return s


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=f"Centralized baseline ({MODEL_NAME})"
    )
    p.add_argument("--dataset", choices=list(DATASETS), default="paysim")
    p.add_argument(
        "--oversampling",
        choices=list(OVERSAMPLING_CHOICES),
        default="smote",
        help="Global oversampler applied to x_train (smote | adasyn | none).",
    )
    p.add_argument(
        "--sampling_strategy",
        type=str,
        default="0.01",
        help=(
            "Passed to imblearn's sampling_strategy. 'auto' = 1:1 fraud:non-fraud. "
            "A float sets the post-resample minority/majority ratio."
        ),
    )
    p.add_argument("--random_seed", type=int, default=42)
    p.add_argument("--num_epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument(
        "--device",
        type=str,
        default=None,
        help="Training device: 'cpu' or 'cuda' (default: auto-detect).",
    )
    # Transformer architecture (must match FL config if comparing directly)
    p.add_argument("--d_model", type=int, default=64)
    p.add_argument("--nhead", type=int, default=4)
    p.add_argument("--num_layers", type=int, default=2)
    p.add_argument("--dim_feedforward", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--use_wandb", type=_str2bool, default=False)
    p.add_argument("--wandb_project", type=str, default="fraud-fl-TA")
    return p.parse_args()


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


def main() -> None:
    args = _parse_args()
    seed = int(args.random_seed)
    num_epochs = int(args.num_epochs)
    batch_size = int(args.batch_size)
    lr = float(args.lr)
    weight_decay = float(args.weight_decay)

    torch.manual_seed(seed)
    np.random.seed(seed)

    dataset = str(args.dataset).lower()
    data = load_dataset(dataset, random_state=seed)
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
                "weight_decay": weight_decay,
                "d_model": args.d_model,
                "nhead": args.nhead,
                "num_layers": args.num_layers,
                "dim_feedforward": args.dim_feedforward,
                "dropout": args.dropout,
            },
        )

    n_features = int(x_train.shape[1])
    model = BertFraudModel(
        input_dim=n_features,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        device=args.device,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
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
            xb = xb.to(model.device)
            yb = yb.to(model.device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())
            n_batches += 1
        avg_loss = epoch_loss / max(n_batches, 1)

        val_scores = model.predict_proba(x_val)[:, 1]
        ep_threshold = best_f1_threshold(y_val, val_scores)
        v = metrics_at_threshold(y_val, val_scores, ep_threshold)

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
    test_scores = model.predict_proba(x_test)[:, 1]
    threshold, v, t = tuned_metrics(y_val, val_scores, y_test, test_scores)

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
        dataset=dataset,
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
