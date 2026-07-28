"""Post-sweep comparability verification (Part 3).

Two assertions over every per-run summary CSV in ``results/logs/``:

1. **Child == cache.** Each row's child-emitted ``data_hash`` (and
   ``partition_hash`` for federated rows) must equal the hash the cache produces
   for that ``(dataset, seed[, scheme, alpha])`` — proving the model consumed the
   cached data, not something it recomputed differently.
2. **Cross-run equality.** All runs sharing a ``(dataset, seed, scheme, alpha)``
   cell must carry identical hashes — proving every model saw byte-identical data
   and partition.

Rows with an empty ``data_hash`` (e.g. an entry point not yet wired to emit it —
currently FedXGBllr) are reported as UNVERIFIED rather than silently passing.

Run: ``python -m experiments.verify_hashes`` — exits non-zero on any mismatch.
"""

from __future__ import annotations

import csv
import os
import sys
from collections import defaultdict
from typing import Dict, List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from experiments import data_cache  # noqa: E402

LOGS_ROOT = os.path.join(os.path.dirname(__file__), "..", "results", "logs")


def _rows() -> List[dict]:
    out = []
    for dp, _d, fs in os.walk(os.path.abspath(LOGS_ROOT)):
        for fn in fs:
            if fn.endswith(".csv") and not fn.endswith("_rounds.csv"):
                try:
                    out.extend(csv.DictReader(open(os.path.join(dp, fn))))
                except (OSError, csv.Error):
                    pass
    return out


def _scheme_to_condition(scheme: str):
    if scheme == "iid":
        return "iid", None
    if scheme == "dirichlet":
        return "noniid", "alpha"
    return None, None  # centralized — no partition


def verify() -> int:
    rows = _rows()
    n_ok = n_fail = n_unverified = 0
    fails: List[str] = []
    groups: Dict[tuple, List[dict]] = defaultdict(list)

    for r in rows:
        ds, seed = r.get("dataset", ""), r.get("random_seed", "")
        dh = (r.get("data_hash") or "").strip()
        if not dh:
            n_unverified += 1
            continue
        try:
            seed_i = int(seed)
        except (TypeError, ValueError):
            continue
        # (1a) child data_hash == cache data_hash
        _data, cache_dh = data_cache.get_preprocessed(ds, seed_i)
        if dh != cache_dh:
            n_fail += 1
            fails.append(f"{r.get('run_name')}: data_hash {dh[:12]} != cache {cache_dh[:12]}")
        else:
            n_ok += 1
        # (1b) partition hash for federated rows
        cond, need_alpha = _scheme_to_condition(r.get("scheme", ""))
        ph = (r.get("partition_hash") or "").strip()
        if cond and ph and ph not in ("n/a (centralized)",):
            alpha = float(r["alpha"]) if (need_alpha and r.get("alpha")) else None
            _idx, cache_ph = data_cache.get_partition_indices(ds, seed_i, alpha, cond)
            if ph != cache_ph:
                n_fail += 1
                fails.append(f"{r.get('run_name')}: partition_hash {ph[:12]} != cache {cache_ph[:12]}")
        groups[(ds, seed, r.get("scheme"), r.get("alpha"))].append(r)

    # (2) cross-run equality within each cell
    for key, grp in groups.items():
        dhs = {(g.get("data_hash") or "").strip() for g in grp if (g.get("data_hash") or "").strip()}
        if len(dhs) > 1:
            n_fail += 1
            fails.append(f"cell {key}: {len(dhs)} distinct data_hash across {len(grp)} runs")

    print("=== HASH VERIFICATION (Part 3) ===")
    print(f"  child==cache checks passed : {n_ok}")
    print(f"  unverified (no hash emitted): {n_unverified}")
    print(f"  failures                    : {n_fail}")
    for f in fails:
        print("   FAIL " + f)
    print("=== " + ("OK" if n_fail == 0 else "FAILED") + " ===")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(verify())
