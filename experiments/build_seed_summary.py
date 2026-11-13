"""Build results/seeds/seed<N>/clean_summary.csv the way results/clean_summary.csv was built.

results/clean_summary.csv (seed 42, frozen) is the 96 per-run summary CSVs under
results/logs/ stacked with pandas, columns in the writer's SUMMARY_COLUMNS order;
it differs from those files only by pandas' float formatting and by reading "n/a"
as a blank. This script applies the same recipe to one seed's tree, so the two
summaries compare column for column, and puts the rows in the frozen file's order
so they can be read side by side. Applied to results/logs/ it reproduces every
one of results/clean_summary.csv's 3,360 cells; 60 print the same float with
different digits (worst relative difference 2e-16). It does not use
experiments/collect_results.py, which replaces each row's exact baseline_auprc
with a rounded constant.

Fails loudly — non-zero exit, naming the path — unless the tree holds exactly one
summary row for every cell of the frozen summary, all at the requested seed.

    python -m experiments.build_seed_summary --seed 123
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.results_paths import assert_writable, seed_root  # noqa: E402
from evaluation.results_writer import SUMMARY_COLUMNS  # noqa: E402

FROZEN_SUMMARY = PROJECT_ROOT / "results" / "clean_summary.csv"


def stack(logs: Path) -> pd.DataFrame:
    """Every per-run summary CSV under ``logs`` (not the _rounds files), stacked,
    columns in SUMMARY_COLUMNS order. The seed-42 per-run files carry the
    backfilled Recall@5%FPR columns at the end; the summary never did."""
    files = sorted(p for p in Path(logs).rglob("*.csv") if not p.name.endswith("_rounds.csv"))
    if not files:
        raise SystemExit(f"FATAL: no per-run summary CSVs under {logs}")
    df = pd.concat([pd.read_csv(p) for p in files], ignore_index=True)
    missing = [c for c in SUMMARY_COLUMNS if c not in df.columns]
    extra = [c for c in df.columns if c not in SUMMARY_COLUMNS]
    if missing or extra:
        raise SystemExit(f"FATAL: per-run CSVs under {logs} do not match SUMMARY_COLUMNS "
                         f"— missing: {missing}, unexpected: {extra}")
    return df[list(SUMMARY_COLUMNS)]


def cells(df: pd.DataFrame) -> list:
    """One cell key per row. alpha as text, so IID/centralized (NaN) compare equal."""
    alpha = df["alpha"].map(lambda a: "" if pd.isna(a) else f"{float(a):g}")
    return list(zip(df["dataset"], df["model"], df["scheme"], alpha, df["oversampling"]))


def in_frozen_order(df: pd.DataFrame, frozen: pd.DataFrame, where: str) -> pd.DataFrame:
    """``df`` reordered to the frozen summary's rows; raises unless the cells match 1:1."""
    have, want = cells(df), cells(frozen)
    dup = sorted({k for k in have if have.count(k) > 1})
    missing = [k for k in want if k not in set(have)]
    extra = [k for k in have if k not in set(want)]
    if dup or missing or extra:
        raise SystemExit(
            f"FATAL: {where} does not hold the frozen summary's {len(want)} cells "
            f"one-to-one — duplicated: {dup}, missing: {missing}, unexpected: {extra}")
    order = {k: i for i, k in enumerate(want)}
    return df.iloc[sorted(range(len(df)), key=lambda i: order[have[i]])]


def build(seed: int) -> Path:
    root = seed_root(seed)
    df = stack(root / "logs")
    wrong = df[df["random_seed"] != seed]
    if len(wrong):
        raise SystemExit(f"FATAL: {len(wrong)} row(s) under {root / 'logs'} are not "
                         f"seed {seed}: {list(wrong['run_name'])}")
    df = in_frozen_order(df, pd.read_csv(FROZEN_SUMMARY), str(root / "logs"))
    out = root / "clean_summary.csv"
    assert_writable(out)
    df.to_csv(out, index=False)
    print(f"wrote {out} ({len(df)} rows, seed {seed})")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, required=True)
    args = ap.parse_args(argv)
    build(args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
