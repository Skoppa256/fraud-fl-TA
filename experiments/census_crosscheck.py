"""Cross-check cached partitions against the independent minority census.

``notebooks/minority_census.py`` produced per-client minority counts (seed 42)
directly from ``get_partition``. The cached partitions (recovered via the
index-proxy trick and round-tripped through disk) must reproduce those counts
EXACTLY for every overlapping cell. If the two disagree, one is wrong and we need
to know before 96 runs depend on it.

Overlap: the census sweeps ``{iid, dirichlet a=0.5/1.0/5.0}``; the cache exposes
``iid`` and ``noniid`` (dirichlet at a chosen alpha). This checks every census
cell at seed 42 by mapping scheme+alpha to the cache's (condition, alpha).

Run: ``python -m experiments.census_crosscheck`` — prints a per-cell PASS/FAIL
table and exits non-zero on any mismatch.
"""

from __future__ import annotations

import csv
import os
import sys
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from experiments import data_cache  # noqa: E402

CENSUS_CSV = os.path.join(
    os.path.dirname(__file__), "..", "results", "analysis", "minority_census.csv"
)
SEED = 42


def _census_counts(path: str) -> Dict[Tuple[str, str], Dict[int, int]]:
    """Return {(dataset, scheme_alpha_key): {client_id: n_minority}} for seed 42."""
    out: Dict[Tuple[str, str], Dict[int, int]] = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if int(row["seed"]) != SEED:
                continue
            scheme = row["scheme"]
            alpha = row.get("alpha", "")
            key = "iid" if scheme == "iid" else f"dirichlet_a{alpha}"
            out.setdefault((row["dataset"], key), {})[int(row["client_id"])] = int(
                row["n_minority"]
            )
    return out


def _cache_counts(dataset: str, scheme_key: str) -> Dict[int, int]:
    if scheme_key == "iid":
        return data_cache.partition_minority_counts(dataset, SEED, None, "iid")
    alpha = float(scheme_key.split("_a")[1])
    return data_cache.partition_minority_counts(dataset, SEED, alpha, "noniid")


def crosscheck() -> List[dict]:
    census = _census_counts(os.path.abspath(CENSUS_CSV))
    results: List[dict] = []
    for (dataset, scheme_key), cen in sorted(census.items()):
        cache = _cache_counts(dataset, scheme_key)
        match = cen == cache
        results.append(
            {
                "dataset": dataset,
                "cell": scheme_key,
                "match": match,
                "census": cen,
                "cache": cache,
            }
        )
    return results


def main() -> int:
    print(f"=== census cross-check (seed {SEED}) — cached partitions vs minority_census.csv ===")
    rows = crosscheck()
    n_fail = 0
    for r in rows:
        status = "PASS" if r["match"] else "FAIL"
        if not r["match"]:
            n_fail += 1
        print(
            f"  [{status}] {r['dataset']:<11} {r['cell']:<16} "
            f"census={[r['census'][k] for k in sorted(r['census'])]} "
            f"cache={[r['cache'][k] for k in sorted(r['cache'])]}"
        )
    print(f"=== {len(rows) - n_fail}/{len(rows)} cells match ===")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
