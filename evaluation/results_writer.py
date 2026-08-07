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
    # Configured FL round budget (was "num_rounds"): 20 for LR/SVM/GBM/FFD/BERT,
    # 50 for FedXGBllr (Flower hfedxgboost baseline). Pairs with rounds_completed.
    # "n/a" for centralized. Renamed for an unambiguous completed-vs-configured pair.
    "rounds_configured",
    "num_clients",
    "best_round",
    "best_val_auprc",
    "best_val_f1",
    "best_val_precision",
    "best_val_recall",
    # Recall@5%FPR on validation: recall reachable while holding the false-alarm
    # rate at <=5%. A threshold-specific operational metric complementing the
    # threshold-free AUPRC. See evaluation.metrics.recall_at_fpr.
    "best_val_recall_at_fpr",
    "test_auprc",
    "test_f1",
    "test_precision",
    "test_recall",
    # Recall@5%FPR on the central test set, plus the operating point it was read
    # at: the score cut-off (test_threshold_at_fpr) and the FPR actually achieved
    # there (test_actual_fpr, typically just under 0.05). See recall_at_fpr.
    # test_threshold_at_fpr is on the model's own score scale — a probability for
    # LR/GBM/FFD/BERT/FedXGBllr, a decision_function MARGIN for SVM (see the
    # `threshold` note below); large negative SVM values are expected.
    "test_recall_at_fpr",
    "test_threshold_at_fpr",
    "test_actual_fpr",
    # Calibration (van den Goorbergh et al. 2022). "NA" for models that emit
    # margins, not probabilities (e.g. SVM decision_function). See
    # evaluation.metrics.calibration_metrics.
    "test_brier",
    "test_cal_intercept",
    "test_cal_slope",
    # Threshold policy (Decision 2): tuned per arm on the central validation set.
    # IMPORTANT: `threshold` is expressed on each model's OWN score scale — a
    # probability in [0, 1] for LR/GBM/FFD/BERT/FedXGBllr, but a signed
    # `decision_function` MARGIN for SVM (hinge, no probability). So an SVM
    # threshold of e.g. -128.08 (creditcard) or -49.14 (BAF) is a normal
    # large-magnitude margin, NOT a bug, and is not comparable to a
    # probability-scale threshold such as GBM's ~0.0014. Same scale caveat
    # applies to `test_threshold_at_fpr` below.
    "threshold_policy",
    "threshold",
    # Comparability proof (Part 3): hashes of the data/partition the model
    # actually consumed, emitted by the child itself.
    "data_hash",
    "partition_hash",
    # Interpretation context (Part 4). rounds_completed = federated FIT rounds
    # actually executed, EXCLUDING the round-0 initial evaluation — defined
    # consistently across all models so the column means the same thing for LR
    # (20) and FedXGBllr (its early-stopped count). Pairs with rounds_configured.
    "rounds_completed",
    # GBM validation-set iteration selection: boosting-prefix length kept
    # (max-AUPRC on central val). "n/a" for every non-GBM model. See
    # models.gbm_bestmodel.iteration_selection.
    "n_iter_selected",
    "n_clients_below_smote_floor",
    "baseline_auprc",
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
    # Recall@5%FPR per round (threshold-free operational metric; see
    # evaluation.metrics.recall_at_fpr).
    "val_recall_at_fpr",
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


# Repo root, so every model writes to the SAME results/logs/ regardless of the
# cwd it runs in. FedXGBllr runs with cwd=models/fedxgbllr (Hydra needs its conf/),
# so a cwd-relative "results/logs" would silently scatter its results into
# models/fedxgbllr/results/logs/ where the collector and verify_hashes (which walk
# the repo results/logs/) would never find them.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _logs_dir(subdir: str, root: str = None) -> str:
    root = root or os.path.join(_REPO_ROOT, "results", "logs")
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


def _round_ge_one(entry: Dict[str, Any]) -> bool:
    """True if a history entry is a fit round (round >= 1), i.e. not the round-0
    initial evaluation. Non-numeric/missing round → treated as a fit round."""
    try:
        return int(entry.get("round", 1)) >= 1
    except (TypeError, ValueError):
        return True


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
                    "best_val_recall_at_fpr": entry.get("val_recall_at_fpr", ""),
                }
        except (TypeError, ValueError):
            continue
    return {
        "best_val_f1": "",
        "best_val_precision": "",
        "best_val_recall": "",
        "best_val_recall_at_fpr": "",
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
    data_hash: str = "",
    partition_hash: str = "",
    threshold_policy: str = "val_f1_tuned",
    threshold: object = "",
    n_clients_below_smote_floor: object = "",
    baseline_auprc: object = "",
    n_iter_selected: object = "n/a",
) -> Dict[str, str]:
    """Write summary and per-round CSVs for a federated learning run.

    ``rounds_completed`` is DERIVED here (not passed) as the number of federated
    fit rounds with ``round >= 1`` — the round-0 initial evaluation is excluded so
    the column is consistent across models. The per-round CSV likewise contains
    only fit rounds (1..N).

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

    # Fit rounds only (exclude the round-0 initial evaluation), used for both the
    # rounds_completed count and the per-round CSV so they agree with each other
    # and with the runner's row-count.
    fit_history = [h for h in history if _round_ge_one(h)]
    rounds_completed = len(fit_history)

    summary = {
        "dataset": dataset,
        "model": model,
        "scheme": scheme,
        "alpha": "" if alpha is None else alpha,
        "oversampling": oversampling,
        "random_seed": int(seed),
        "rounds_configured": int(num_rounds),
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
        # Recall@5%FPR operating point (blank when the arm did not supply it).
        "test_recall_at_fpr": final_test.get("test_recall_at_fpr", ""),
        "test_threshold_at_fpr": final_test.get("test_threshold_at_fpr", ""),
        "test_actual_fpr": final_test.get("test_actual_fpr", ""),
        # Calibration — "NA" when the arm did not supply it (e.g. SVM margins,
        # or an arm whose eval has not yet been wired to compute calibration).
        "test_brier": final_test.get("test_brier", "NA"),
        "test_cal_intercept": final_test.get("test_cal_intercept", "NA"),
        "test_cal_slope": final_test.get("test_cal_slope", "NA"),
        "threshold_policy": threshold_policy,
        "threshold": final_test.get("threshold", threshold),
        "data_hash": data_hash,
        "partition_hash": partition_hash,
        "rounds_completed": rounds_completed,
        "n_iter_selected": n_iter_selected,
        "n_clients_below_smote_floor": n_clients_below_smote_floor,
        "baseline_auprc": baseline_auprc,
        "timestamp": _utc_iso(),
        "duration_seconds": round(float(duration_seconds), 3),
        "run_name": run_name,
    }
    summary_path = os.path.join(out_dir, f"{run_name}.csv")
    _write_csv(summary_path, SUMMARY_COLUMNS, [summary])

    round_rows: List[Dict[str, Any]] = []
    for entry in fit_history:
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
                "val_recall_at_fpr": entry.get("val_recall_at_fpr", ""),
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
    data_hash: str = "",
    threshold_policy: str = "val_f1_tuned",
    threshold: object = "",
    baseline_auprc: object = "",
    n_iter_selected: object = "n/a",
) -> Dict[str, str]:
    """Write a single-row summary CSV for a centralized baseline.

    The schema matches the FL summary so a single collector can ingest both.
    Val metrics are duplicated into ``best_val_*`` columns; FL-only fields
    (``best_round``) are left blank and ``rounds_configured``/``rounds_completed``
    are ``n/a`` (centralized training has no federated rounds).

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
        "rounds_configured": "n/a",
        "num_clients": 1,
        "best_round": "",
        "best_val_auprc": val_metrics.get("val_auprc", ""),
        "best_val_f1": val_metrics.get("val_f1", ""),
        "best_val_precision": val_metrics.get("val_precision", ""),
        "best_val_recall": val_metrics.get("val_recall", ""),
        "best_val_recall_at_fpr": val_metrics.get("val_recall_at_fpr", ""),
        "test_auprc": test_metrics.get("test_auprc", ""),
        "test_f1": test_metrics.get("test_f1", ""),
        "test_precision": test_metrics.get("test_precision", ""),
        "test_recall": test_metrics.get("test_recall", ""),
        "test_recall_at_fpr": test_metrics.get("test_recall_at_fpr", ""),
        "test_threshold_at_fpr": test_metrics.get("test_threshold_at_fpr", ""),
        "test_actual_fpr": test_metrics.get("test_actual_fpr", ""),
        "test_brier": test_metrics.get("test_brier", "NA"),
        "test_cal_intercept": test_metrics.get("test_cal_intercept", "NA"),
        "test_cal_slope": test_metrics.get("test_cal_slope", "NA"),
        "threshold_policy": threshold_policy,
        "threshold": test_metrics.get("threshold", threshold),
        "data_hash": data_hash,
        "partition_hash": "n/a (centralized)",
        "rounds_completed": "n/a",
        "n_iter_selected": n_iter_selected,
        "n_clients_below_smote_floor": "n/a",
        "baseline_auprc": baseline_auprc,
        "timestamp": _utc_iso(),
        "duration_seconds": round(float(duration_seconds), 3),
        "run_name": run_name,
    }
    summary_path = os.path.join(out_dir, f"{run_name}.csv")
    _write_csv(summary_path, SUMMARY_COLUMNS, [summary])
    print(f"[results] wrote summary CSV → {summary_path}")
    return {"summary": summary_path, "run_name": run_name}
