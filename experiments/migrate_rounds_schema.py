"""One-off, idempotent migration of results CSVs to the standardized rounds schema.

Rows written before the rounds standardization carry the OLD convention:
``rounds_completed`` includes the round-0 initial evaluation, and a ``num_rounds``
column holds the configured budget. New rows use ``rounds_completed`` = federated
FIT rounds (round >= 1) and ``rounds_configured`` in place of ``num_rounds``.
Mixing the two makes the column mean different things depending on when the row
was written. This script rewrites the old rows in place:

  * summary CSVs (``results/logs/<ds>/<subdir>/<run>.csv``): rewrite to the current
    SUMMARY schema — ``rounds_completed`` recomputed as the count of fit rounds
    (round >= 1) from the sibling per-round CSV, ``num_rounds`` → ``rounds_configured``
    (filled where absent: 50 FedXGBllr, 20 other FL, ``n/a`` centralized), and any
    newly-added column (e.g. ``n_iter_selected``) defaulted.
  * per-round CSVs (``<run>_rounds.csv``): drop the round-0 row.
  * ``results/sweep/sweep_master.csv``: same ``rounds_completed`` fix + add
    ``rounds_configured``.

The new ``rounds_completed`` is derived as ``count(round >= 1)`` in the per-round
CSV rather than blindly subtracting 1, because the round-0 offset is NOT uniform:
GBM logs only fit rounds (offset 0) while LR/SVM/FFD/BERT/FedXGBllr log round 0
(offset 1). ``count(round >= 1)`` is correct for both and is invariant to whether
the per-round CSV has already had its round-0 row dropped — so the migration is
robust to ordering and safe to re-run.

Idempotent: a summary/master already carrying ``rounds_configured`` is skipped; a
per-round CSV with no round-0 row is skipped.

Usage::

    python -m experiments.migrate_rounds_schema [--results-root results] [--dry-run]
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import List, Optional, Tuple

from evaluation.results_writer import SUMMARY_COLUMNS
from experiments.run_sweep import _MASTER_COLUMNS, _child_run_name, RunSpec

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _configured(model: str, scheme_or_condition: str) -> object:
    """rounds_configured for a model: n/a centralized, 50 FedXGBllr, 20 other FL."""
    if scheme_or_condition == "centralized":
        return "n/a"
    return 50 if model == "fedxgbllr" else 20


def _read_csv(path: Path) -> Tuple[List[str], List[dict]]:
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        return list(r.fieldnames or []), list(r)


def _write_csv(path: Path, header: List[str], rows: List[dict]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({c: row.get(c, "") for c in header})
    os.replace(tmp, path)


def _round_val(row: dict) -> Optional[int]:
    try:
        return int(float(str(row.get("round", "")).strip()))
    except (TypeError, ValueError):
        return None


def _fit_round_count(rounds_csv: Path) -> Optional[int]:
    """Count rows with round >= 1 (fit rounds). None if the file is absent."""
    if not rounds_csv.is_file():
        return None
    _hdr, rows = _read_csv(rounds_csv)
    return sum(1 for r in rows if (_round_val(r) or 0) >= 1)


def _new_completed(scheme: str, rounds_csv: Optional[Path], old_val, model: str) -> object:
    """New rounds_completed: n/a for centralized; else count(round>=1) from the
    per-round CSV. Falls back to (old - model offset) only if that CSV is absent."""
    if scheme == "centralized":
        return "n/a"
    s = ("" if old_val is None else str(old_val)).strip()
    if rounds_csv is not None:
        n = _fit_round_count(rounds_csv)
        if n is not None:
            return n
    if s in ("", "n/a"):
        return s or "n/a"
    off = 0 if model == "gbm" else 1  # fallback only: GBM never logs round 0
    try:
        return max(int(float(s)) - off, 0)
    except ValueError:
        return s


def _sibling_rounds(summary_path: Path) -> Path:
    # foo.csv -> foo_rounds.csv
    return summary_path.with_name(summary_path.stem + "_rounds.csv")


# --------------------------------------------------------------------------- #
# migrators (return an action label)
# --------------------------------------------------------------------------- #
def migrate_summary(path: Path, dry: bool) -> str:
    header, rows = _read_csv(path)
    if not header:
        return "skip-empty"
    if "rounds_configured" in header:
        return "skip-new"
    if "rounds_completed" not in header and "num_rounds" not in header:
        return "skip-unrelated"
    sib = _sibling_rounds(path)
    out: List[dict] = []
    for old in rows:
        model = (old.get("model") or "").strip()
        scheme = (old.get("scheme") or "").strip()
        nr: dict = {}
        for c in SUMMARY_COLUMNS:
            if c == "rounds_configured":
                v = (old.get("num_rounds") or "").strip()
                nr[c] = ("n/a" if scheme == "centralized"
                         else v if v not in ("", "n/a")
                         else _configured(model, scheme))
            elif c == "rounds_completed":
                nr[c] = _new_completed(scheme, sib, old.get("rounds_completed"), model)
            elif c == "n_iter_selected":
                nr[c] = old.get("n_iter_selected") or "n/a"
            else:
                nr[c] = old.get(c, "")
        out.append(nr)
    if not dry:
        _write_csv(path, list(SUMMARY_COLUMNS), out)
    return "migrated"


def migrate_rounds(path: Path, dry: bool) -> str:
    header, rows = _read_csv(path)
    if not header or "round" not in header:
        return "skip-unrelated"
    kept = [r for r in rows if _round_val(r) != 0]
    if len(kept) == len(rows):
        return "skip-noround0"
    if not dry:
        _write_csv(path, header, kept)
    return "migrated"


def _master_rounds_csv(row: dict, results_root: Path) -> Optional[Path]:
    """Reconstruct the child per-round CSV path for a master row (FL only)."""
    model = (row.get("model") or "").strip()
    cond = (row.get("condition") or "").strip()
    dataset = (row.get("dataset") or "").strip()
    if cond == "centralized" or not (model and dataset):
        return None
    a_raw = str(row.get("alpha", "")).strip()
    try:
        alpha = float(a_raw) if a_raw not in ("", "n/a") else None
    except ValueError:
        alpha = None
    try:
        spec = RunSpec(dataset, model, (row.get("smote_arm") or "no-smote").strip(),
                       cond, seed=int(float(row.get("seed", 42) or 42)), alpha=alpha)
        subdir = "fedxgbllr" if model == "fedxgbllr" else model
        return (results_root / "logs" / dataset / subdir
                / f"{_child_run_name(spec)}_rounds.csv")
    except Exception:  # noqa: BLE001 — bad row → no lookup, fall back
        return None


def migrate_master(path: Path, dry: bool, results_root: Path) -> str:
    header, rows = _read_csv(path)
    if not header:
        return "skip-empty"
    if "rounds_configured" in header:
        return "skip-new"
    if "rounds_completed" not in header:
        return "skip-unrelated"
    out: List[dict] = []
    for old in rows:
        model = (old.get("model") or "").strip()
        cond = (old.get("condition") or "").strip()
        nr: dict = {}
        for c in _MASTER_COLUMNS:
            if c == "rounds_configured":
                nr[c] = _configured(model, cond)
            elif c == "rounds_completed":
                nr[c] = _new_completed(cond, _master_rounds_csv(old, results_root),
                                       old.get("rounds_completed"), model)
            else:
                nr[c] = old.get(c, "")
        out.append(nr)
    if not dry:
        _write_csv(path, list(_MASTER_COLUMNS), out)
    return "migrated"


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def run_migration(results_root: Path, dry: bool) -> dict:
    logs = results_root / "logs"
    summaries, rounds_files = [], []
    if logs.is_dir():
        for p in sorted(logs.rglob("*.csv")):
            if "_quarantine" in p.parts:
                continue
            (rounds_files if p.name.endswith("_rounds.csv") else summaries).append(p)

    tally: dict = {}

    def _bump(kind: str, action: str) -> None:
        tally[f"{kind}:{action}"] = tally.get(f"{kind}:{action}", 0) + 1

    # Summaries and master first (they read the per-round CSVs); per-round last.
    # (count(round>=1) is order-invariant, so this is belt-and-suspenders.)
    for p in summaries:
        act = migrate_summary(p, dry)
        _bump("summary", act)
        if act == "migrated":
            print(f"  [summary ] {act}: {p.relative_to(results_root)}")

    master = results_root / "sweep" / "sweep_master.csv"
    if master.is_file():
        act = migrate_master(master, dry, results_root)
        _bump("master", act)
        print(f"  [master  ] {act}: {master.relative_to(results_root)}")

    for p in rounds_files:
        act = migrate_rounds(p, dry)
        _bump("rounds", act)
        if act == "migrated":
            print(f"  [rounds  ] {act}: {p.relative_to(results_root)}")

    return tally


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-root", default=str(PROJECT_ROOT / "results"),
                    help="root containing logs/ and sweep/ (default: <repo>/results)")
    ap.add_argument("--dry-run", action="store_true",
                    help="report actions without writing any file")
    args = ap.parse_args(argv)
    root = Path(args.results_root)
    print(f"=== rounds-schema migration ({'DRY RUN' if args.dry_run else 'WRITING'}) "
          f"| root={root} ===")
    tally = run_migration(root, args.dry_run)
    print("\n=== summary ===")
    for k in sorted(tally):
        print(f"  {k}: {tally[k]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
