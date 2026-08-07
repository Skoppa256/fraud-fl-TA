"""Aggregate per-run CSVs into a single master table and print Markdown.

Walks ``results/logs/**/*.csv`` (skipping the per-round files), concatenates
every summary row into ``results/summary_table.csv``, and prints a Markdown
table grouped by ``(model, scheme, alpha, oversampling)`` ready to paste into
the README results section.

Usage
-----
    python -m experiments.collect_results
    python -m experiments.collect_results --out results/summary_table.csv
    python -m experiments.collect_results --markdown-only  # skip CSV write

The schema is whatever :mod:`evaluation.results_writer` writes; see
``SUMMARY_COLUMNS`` there for the authoritative list.
"""

from __future__ import annotations

import argparse
import csv
import os
from typing import Any, Dict, List, Sequence


LOGS_ROOT = "results/logs"
DEFAULT_OUT = "results/summary_table.csv"

# Random-classifier AUPRC equals the positive-class prevalence, so AUPRC's floor
# scales with the fraud rate and is NOT comparable across datasets: a BAF score
# looks better than a PaySim score for free. Reporting this baseline alongside
# every test_auprc (rather than a normalized metric) keeps the raw, standard
# AUPRC while making the cross-dataset floor explicit. Values are the test-set
# fraud prevalence per dataset (stratified split preserves the overall rate).
DATASET_BASELINE_AUPRC = {
    "paysim": 0.00129,
    "creditcard": 0.00173,
    "baf": 0.01103,
}


def _baseline_auprc(dataset: str) -> str:
    """Random-baseline AUPRC (= prevalence) for a dataset, or '' if unknown."""
    v = DATASET_BASELINE_AUPRC.get((dataset or "").lower())
    return "" if v is None else f"{v:.5f}"

PRIMARY_COLUMNS: Sequence[str] = (
    "dataset",
    "model",
    "scheme",
    "alpha",
    "oversampling",
    "random_seed",
    "best_round",
    "best_val_auprc",
    "test_auprc",
    "test_f1",
    "test_precision",
    "test_recall",
    "test_recall_at_fpr",
    "duration_seconds",
    "timestamp",
    "run_name",
)


def _is_round_file(path: str) -> bool:
    return path.endswith("_rounds.csv")


def _read_summary_rows(root: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    if not os.path.isdir(root):
        return rows
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            if not name.endswith(".csv"):
                continue
            full = os.path.join(dirpath, name)
            if _is_round_file(full):
                continue
            try:
                with open(full, newline="") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        rows.append(row)
            except (OSError, csv.Error) as exc:
                print(f"[collect] WARN: failed reading {full}: {exc}")
    return rows


def _all_columns(rows: List[Dict[str, str]]) -> List[str]:
    seen: List[str] = []
    seen_set = set()
    for row in rows:
        for k in row.keys():
            if k not in seen_set:
                seen.append(k)
                seen_set.add(k)
    return seen


def _sort_key(row: Dict[str, str]) -> tuple:
    def _f(name: str, default: float) -> float:
        try:
            return float(row.get(name) or default)
        except ValueError:
            return default

    return (
        row.get("dataset", ""),
        row.get("model", ""),
        row.get("scheme", ""),
        _f("alpha", 1e9),
        row.get("oversampling", ""),
        _f("random_seed", 0),
    )


def write_summary_csv(rows: List[Dict[str, str]], out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    columns = _all_columns(rows)
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        for row in sorted(rows, key=_sort_key):
            w.writerow(row)
    print(f"[collect] wrote master CSV: {out_path} ({len(rows)} rows)")


def _fmt(val: Any, digits: int = 4) -> str:
    if val in (None, ""):
        return "-"
    try:
        return f"{float(val):.{digits}f}"
    except (TypeError, ValueError):
        return str(val)


def _alpha_label(alpha: Any) -> str:
    if alpha in (None, "", "None"):
        return "—"
    try:
        return f"α={float(alpha):g}"
    except (TypeError, ValueError):
        return str(alpha)


def print_markdown_table(rows: List[Dict[str, str]]) -> None:
    """Print a Markdown table grouped by (model, scheme, alpha, oversampling).

    Each row is a single seed result; multi-seed runs print one row per seed
    so you can spot variance directly. Sort order is stable to keep diffs
    minimal between successive collector invocations.
    """
    if not rows:
        print("(no runs found)")
        return

    header = (
        "| Dataset | Model | Scheme | Oversampling | Seed | "
        "test_auprc | baseline (prev.) | test_f1 | test_precision | test_recall | "
        "recall@5%fpr | brier | cal_int | cal_slope | best_round | duration (s) |"
    )
    sep = (
        "|---------|-------|--------|--------------|------|"
        "------------|------------------|---------|----------------|-------------|"
        "-------------|-------|---------|-----------|------------|--------------|"
    )
    print(header)
    print(sep)
    for row in sorted(rows, key=_sort_key):
        scheme = row.get("scheme", "")
        if scheme == "dirichlet":
            scheme_label = f"Dirichlet {_alpha_label(row.get('alpha'))}"
        elif scheme == "iid":
            scheme_label = "IID"
        else:
            scheme_label = scheme or "—"
        print(
            "| "
            f"{row.get('dataset', '-') or '-'} | "
            f"{row.get('model', '-')} | "
            f"{scheme_label} | "
            f"{row.get('oversampling', '-')} | "
            f"{row.get('random_seed', '-')} | "
            f"{_fmt(row.get('test_auprc'))} | "
            f"{_fmt(row.get('baseline_auprc') or _baseline_auprc(row.get('dataset', '')))} | "
            f"{_fmt(row.get('test_f1'))} | "
            f"{_fmt(row.get('test_precision'))} | "
            f"{_fmt(row.get('test_recall'))} | "
            f"{_fmt(row.get('test_recall_at_fpr'))} | "
            f"{_fmt(row.get('test_brier'))} | "
            f"{_fmt(row.get('test_cal_intercept'))} | "
            f"{_fmt(row.get('test_cal_slope'))} | "
            f"{row.get('best_round', '-') or '-'} | "
            f"{_fmt(row.get('duration_seconds'), digits=1)} |"
        )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Aggregate per-run CSVs into a master table + Markdown."
    )
    p.add_argument(
        "--logs-root",
        default=LOGS_ROOT,
        help=f"Directory to walk (default: {LOGS_ROOT}).",
    )
    p.add_argument(
        "--out",
        default=DEFAULT_OUT,
        help=f"Master CSV output path (default: {DEFAULT_OUT}).",
    )
    p.add_argument(
        "--markdown-only",
        action="store_true",
        help="Print Markdown only — skip writing the master CSV.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    rows = _read_summary_rows(args.logs_root)
    # Stamp the random-baseline AUPRC (= prevalence) onto every row so both the
    # master CSV and the Markdown table carry the cross-dataset floor.
    for row in rows:
        row["baseline_auprc"] = _baseline_auprc(row.get("dataset", ""))
    if not args.markdown_only:
        write_summary_csv(rows, args.out)
    print()
    print_markdown_table(rows)


if __name__ == "__main__":
    main()
