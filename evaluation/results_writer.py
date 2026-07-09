"""Shared CSV results writer for all FL and centralized baseline runs.

Writes structured CSVs under ``results/logs/<dataset>/<subdir>/`` so that every
model in the comparison emits identical schema, and PaySim and creditcard runs
never collide on disk. Two files per FL run:

* ``<run_name>.csv``        — single-row summary
* ``<run_name>_rounds.csv`` — per-round val metrics

Centralized baselines emit only the summary CSV (no FL rounds).

Also exposes :func:`build_run_name` / :func:`build_centralized_run_name`
so every model and the downstream collector agree on the run identifier.
The canonical FL format is::

    <model>_<scheme>_alpha<alpha>_<oversampling>_seed<seed>

where ``<alpha>`` is the literal ``-`` for IID runs.
"""

from __future__ import annotations

import csv
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence


SUMMARY_COLUMNS: Sequence[str] = (
    "dataset",
    "model",
    "scheme",
    "alpha",
    "oversampling",
    "random_seed",
    "num_rounds",
    "num_clients",
    "best_round",
    "best_val_auprc",
    "best_val_f1",
    "best_val_precision",
    "best_val_recall",
    "test_auprc",
    "test_f1",
    "test_precision",
    "test_recall",
    "timestamp",
    "duration_seconds",
    "run_name",
)

ROUND_COLUMNS: Sequence[str] = (
    "round",
    "val_auprc",
    "val_f1",
    "val_precision",
    "val_recall",
    "train_loss",
)


def _alpha_token(alpha: Any) -> str:
    return "-" if alpha is None else str(alpha)


def build_run_name(
    model: str, scheme: str, alpha: Any, oversampling: str, seed: int
) -> str:
    """Canonical FL run name. Used for both W&B and CSV filenames."""
    return (
        f"{model}_{scheme}_alpha{_alpha_token(alpha)}_"
        f"{oversampling}_seed{int(seed)}"
    )


def build_centralized_run_name(model: str, oversampling: str, seed: int) -> str:
    """Canonical centralized-baseline run name."""
    return f"centralized_{model}_{oversampling}_seed{int(seed)}"


def _logs_dir(subdir: str, root: str = "results/logs") -> str:
    path = os.path.join(root, subdir)
    os.makedirs(path, exist_ok=True)
    return path


def _utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _write_csv(
    path: str, columns: Sequence[str], rows: List[Dict[str, Any]]
) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=list(columns), extrasaction="ignore"
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c, "") for c in columns})


def _best_val_metrics(
    history: List[Dict[str, Any]], best_round: int
) -> Dict[str, Any]:
    """Pull val_{f1,precision,recall} from the history row matching best_round.

    Returns empty strings if the round isn't present (e.g. early-stop edge case).
    """
    target = int(best_round) if best_round is not None else -999
    for entry in history or []:
        try:
            if int(entry.get("round", -999)) == target:
                return {
                    "best_val_f1": entry.get("val_f1", ""),
                    "best_val_precision": entry.get("val_precision", ""),
                    "best_val_recall": entry.get("val_recall", ""),
                }
        except (TypeError, ValueError):
            continue
    return {
        "best_val_f1": "",
        "best_val_precision": "",
        "best_val_recall": "",
    }


def write_fl_results(
    *,
    model: str,
    scheme: str,
    alpha: Optional[float],
    oversampling: str,
    seed: int,
    num_rounds: int,
    num_clients: int,
    best_round: int,
    best_val_auprc: float,
    history: List[Dict[str, Any]],
    final_test: Optional[Dict[str, float]],
    duration_seconds: float,
    dataset: str = "paysim",
    subdir: Optional[str] = None,
) -> Dict[str, str]:
    """Write summary and per-round CSVs for a federated learning run.

    Parameters
    ----------
    model
        Canonical short name: ``ffd``, ``lr``, ``svm``, ``gbm``, ``fedxgbllr``.
    dataset
        Dataset identifier (``paysim`` or ``creditcard``). Recorded in the
        ``dataset`` column and used to namespace the output directory so runs
        on different datasets never collide. Defaults to ``paysim`` — an
        unchanged PaySim run keeps its exact ``run_name`` and CSV schema
        (bar the new column) and only gains a ``paysim/`` parent level.
    scheme
        ``iid`` or ``dirichlet``.
    alpha
        Dirichlet concentration, or ``None`` for IID.
    history
        List of dicts, one per round: ``{round, val_auprc, val_f1,
        val_precision, val_recall}`` (and optional ``train_loss``).
    final_test
        ``{test_auprc, test_f1, test_precision, test_recall}`` or ``None``
        (e.g. if the run was interrupted before the final round).
    subdir
        Subdirectory under ``results/logs/`` (defaults to ``model``).

    Returns
    -------
    dict
        ``{"summary": <path>, "rounds": <path>, "run_name": <name>}``.
    """
    run_name = build_run_name(model, scheme, alpha, oversampling, seed)
    out_dir = _logs_dir(os.path.join(dataset, subdir or model))

    history = history or []
    best_val = _best_val_metrics(history, best_round)
    final_test = final_test or {}

    summary = {
        "dataset": dataset,
        "model": model,
        "scheme": scheme,
        "alpha": "" if alpha is None else alpha,
        "oversampling": oversampling,
        "random_seed": int(seed),
        "num_rounds": int(num_rounds),
        "num_clients": int(num_clients),
        "best_round": "" if best_round in (None, -1) else int(best_round),
        "best_val_auprc": (
            ""
            if best_val_auprc in (None, -1.0, -1)
            else float(best_val_auprc)
        ),
        **best_val,
        "test_auprc": final_test.get("test_auprc", ""),
        "test_f1": final_test.get("test_f1", ""),
        "test_precision": final_test.get("test_precision", ""),
        "test_recall": final_test.get("test_recall", ""),
        "timestamp": _utc_iso(),
        "duration_seconds": round(float(duration_seconds), 3),
        "run_name": run_name,
    }
    summary_path = os.path.join(out_dir, f"{run_name}.csv")
    _write_csv(summary_path, SUMMARY_COLUMNS, [summary])

    round_rows: List[Dict[str, Any]] = []
    for entry in history:
        try:
            r = int(entry.get("round", 0))
        except (TypeError, ValueError):
            continue
        round_rows.append(
            {
                "round": r,
                "val_auprc": entry.get("val_auprc", ""),
                "val_f1": entry.get("val_f1", ""),
                "val_precision": entry.get("val_precision", ""),
                "val_recall": entry.get("val_recall", ""),
                "train_loss": entry.get("train_loss", ""),
            }
        )
    rounds_path = os.path.join(out_dir, f"{run_name}_rounds.csv")
    _write_csv(rounds_path, ROUND_COLUMNS, round_rows)

    print(f"[results] wrote summary CSV → {summary_path}")
    print(f"[results] wrote rounds CSV  → {rounds_path}")
    return {
        "summary": summary_path,
        "rounds": rounds_path,
        "run_name": run_name,
    }


def write_centralized_results(
    *,
    model: str,
    oversampling: str,
    seed: int,
    val_metrics: Dict[str, float],
    test_metrics: Dict[str, float],
    duration_seconds: float,
    dataset: str = "paysim",
    subdir: str = "centralized",
) -> Dict[str, str]:
    """Write a single-row summary CSV for a centralized baseline.

    The schema matches the FL summary so a single collector can ingest both.
    Val metrics are duplicated into ``best_val_*`` columns and FL-only fields
    (``best_round``, ``num_rounds``) are left blank.

    ``val_metrics`` / ``test_metrics`` keys are read as ``val_auprc``,
    ``val_f1``, ``val_precision``, ``val_recall`` (and ``test_*`` likewise).
    """
    run_name = build_centralized_run_name(model, oversampling, seed)
    out_dir = _logs_dir(os.path.join(dataset, subdir))

    summary = {
        "dataset": dataset,
        "model": model,
        "scheme": "centralized",
        "alpha": "",
        "oversampling": oversampling,
        "random_seed": int(seed),
        "num_rounds": "",
        "num_clients": 1,
        "best_round": "",
        "best_val_auprc": val_metrics.get("val_auprc", ""),
        "best_val_f1": val_metrics.get("val_f1", ""),
        "best_val_precision": val_metrics.get("val_precision", ""),
        "best_val_recall": val_metrics.get("val_recall", ""),
        "test_auprc": test_metrics.get("test_auprc", ""),
        "test_f1": test_metrics.get("test_f1", ""),
        "test_precision": test_metrics.get("test_precision", ""),
        "test_recall": test_metrics.get("test_recall", ""),
        "timestamp": _utc_iso(),
        "duration_seconds": round(float(duration_seconds), 3),
        "run_name": run_name,
    }
    summary_path = os.path.join(out_dir, f"{run_name}.csv")
    _write_csv(summary_path, SUMMARY_COLUMNS, [summary])
    print(f"[results] wrote summary CSV → {summary_path}")
    return {"summary": summary_path, "run_name": run_name}
