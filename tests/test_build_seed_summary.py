"""experiments.build_seed_summary: a seed's clean_summary.csv, same recipe as seed 42's.

The seed tree is synthesised from the frozen results/clean_summary.csv (read only):
its 96 rows, re-labelled to another seed, written as per-run CSVs in shuffled
order. The builder must put them back in the frozen order, and must refuse a tree
that is missing a cell, holds another seed, or is empty — never write a partial
summary. Runnable via ``pytest`` or ``python tests/<file>.py``.
"""

import contextlib
import csv
import os
import random
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from evaluation import results_paths as P  # noqa: E402
from evaluation.results_writer import SUMMARY_COLUMNS  # noqa: E402
from experiments import build_seed_summary as B  # noqa: E402

SEED = 123


@contextlib.contextmanager
def _seed_tree(seed_of_row=lambda i: SEED, drop=None):
    """A throwaway results/seeds/ holding seed-SEED per-run CSVs for the 96 frozen cells."""
    frozen = list(csv.DictReader(open(B.FROZEN_SUMMARY)))
    prev = P.SEEDS_ROOT
    with tempfile.TemporaryDirectory() as d:
        P.SEEDS_ROOT = Path(d)
        try:
            logs = P.seed_root(SEED) / "logs"
            order = list(range(len(frozen)))
            random.Random(0).shuffle(order)
            for i in order:
                if i == drop:
                    continue
                row = dict(frozen[i])
                row["random_seed"] = str(seed_of_row(i))
                row["run_name"] = row["run_name"].replace("_seed42", f"_seed{seed_of_row(i)}")
                out = logs / row["dataset"] / f"{row['run_name']}.csv"
                out.parent.mkdir(parents=True, exist_ok=True)
                with open(out, "w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=list(SUMMARY_COLUMNS))
                    w.writeheader()
                    w.writerow(row)
            yield frozen
        finally:
            P.SEEDS_ROOT = prev


def _refuses(fragment):
    try:
        B.build(SEED)
    except SystemExit as e:
        assert fragment in str(e), str(e)
        assert not (P.seed_root(SEED) / "clean_summary.csv").exists()
        return
    raise AssertionError("build_seed_summary wrote a summary it should have refused")


def test_rebuilds_the_tree_in_frozen_order():
    with _seed_tree() as frozen:
        out = B.build(SEED)
        rows = list(csv.DictReader(open(out)))
        assert list(rows[0]) == list(SUMMARY_COLUMNS)
        assert len(rows) == len(frozen) == 96
        assert [r["run_name"] for r in rows] == \
            [r["run_name"].replace("_seed42", f"_seed{SEED}") for r in frozen]
        assert {r["random_seed"] for r in rows} == {str(SEED)}


def test_refuses_a_missing_cell():
    with _seed_tree(drop=7):
        _refuses("missing: [(")    # a non-empty list of missing cells


def test_refuses_another_seed_in_the_tree():
    with _seed_tree(seed_of_row=lambda i: 42 if i == 3 else SEED):
        _refuses(f"are not seed {SEED}")


def test_refuses_an_empty_tree():
    prev = P.SEEDS_ROOT
    with tempfile.TemporaryDirectory() as d:
        P.SEEDS_ROOT = Path(d)
        try:
            _refuses("no per-run summary CSVs")
        finally:
            P.SEEDS_ROOT = prev


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nAll {len(fns)} seed-summary tests passed.")
