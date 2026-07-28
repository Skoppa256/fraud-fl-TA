"""
Per-client minority census (DIAGNOSTIC, read-only).

Sweeps every dataset x partition scheme x seed x client and records the local
minority-class situation each client faces, plus what SMOTE (at the uniform
sampling_strategy=0.01 rule) actually does to it. Reuses the real preprocessors,
partitioner, and SMOTE logic (preprocessing.smote.apply_smote) so the numbers
are byte-identical to training — nothing is reimplemented.

Sweep: 3 datasets x {IID, alpha=0.5, alpha=1.0, alpha=5.0} x 3 seeds x 5 clients.

Per client-instance it records:
  n_total, n_minority, minority_fraction, local_ratio (min/maj),
  events_per_parameter (= n_minority / n_features; REPORTED ONLY, never a gate),
  smote_fires, skip_reason (insufficient_minority / target_met / None),
  n_synthetic, synthesis_multiplier, and flags n_minority<10 / n_minority==0.

Outputs (results/analysis/):
  - minority_census.csv           (one row per client-instance)
  - minority_census_summary.md    (per dataset x scheme aggregate)

No training, no config writes, no output outside results/analysis/.
"""

from __future__ import annotations

import contextlib
import io
import os
import sys
import time

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from preprocessing.loader import load_dataset
from partitioning.dirichlet import get_partition
from preprocessing.smote import apply_smote


DATASETS = ("paysim", "creditcard", "baf")
SEEDS = (42, 123, 2024)
SCHEMES = (("iid", None), ("dirichlet", 0.5), ("dirichlet", 1.0), ("dirichlet", 5.0))
NUM_CLIENTS = 5
SAMPLING_STRATEGY = 0.01   # the uniform rule (Part A)
K_NEIGHBORS = 5

OUT_DIR = os.path.join(PROJECT_ROOT, "results/analysis")
CSV_COLUMNS = [
    "dataset", "seed", "scheme", "alpha", "client_id", "n_features",
    "n_total", "n_minority", "minority_fraction", "local_ratio",
    "events_per_parameter", "smote_fires", "skip_reason",
    "n_synthetic", "synthesis_multiplier", "flag_lt10", "flag_zero",
]


def census_one_client(client: dict, n_features: int, seed: int) -> dict:
    """Run apply_smote exactly as training does and record the census row."""
    y = np.asarray(client["y"])
    n_total = int(len(y))
    n_min = int((y == 1).sum())
    n_maj = n_total - n_min
    # apply_smote prints per client; silence it for a clean census log.
    with contextlib.redirect_stdout(io.StringIO()):
        out = apply_smote(
            client, enabled=True, sampling_strategy=SAMPLING_STRATEGY,
            k_neighbors=K_NEIGHBORS, base_seed=seed,
        )
    fires = bool(out.get("smote_applied"))
    return {
        "n_features": n_features,
        "n_total": n_total,
        "n_minority": n_min,
        "minority_fraction": (n_min / n_total) if n_total else 0.0,
        "local_ratio": (n_min / n_maj) if n_maj else float("inf"),
        "events_per_parameter": (n_min / n_features) if n_features else 0.0,
        "smote_fires": fires,
        "skip_reason": "" if fires else (out.get("skip_reason") or ""),
        "n_synthetic": int(out.get("n_synthetic", 0)),
        "synthesis_multiplier": float(out.get("synthesis_multiplier", 0.0)),
        "flag_lt10": n_min < 10,
        "flag_zero": n_min == 0,
    }


def collect_rows() -> list[dict]:
    rows: list[dict] = []
    for name in DATASETS:
        for seed in SEEDS:
            t0 = time.time()
            with contextlib.redirect_stdout(io.StringIO()):
                data = load_dataset(name, random_state=seed)
            x = np.asarray(data["x_train"])
            y = np.asarray(data["y_train"])
            n_features = int(x.shape[1])
            print(f"[census] {name} seed={seed}: x_train={x.shape} "
                  f"fraud={int(y.sum()):,} ({y.mean()*100:.4f}%) "
                  f"loaded in {time.time()-t0:.1f}s")
            for scheme, alpha in SCHEMES:
                clients = get_partition(x, y, scheme=scheme, alpha=alpha,
                                        num_clients=NUM_CLIENTS, random_state=seed)
                for c in clients:
                    row = {
                        "dataset": name, "seed": seed, "scheme": scheme,
                        "alpha": "" if alpha is None else alpha,
                        "client_id": int(c["client_id"]),
                    }
                    row.update(census_one_client(c, n_features, seed))
                    rows.append(row)
    return rows


def write_csv(rows: list[dict], path: str) -> None:
    import csv
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[census] wrote {path} ({len(rows)} rows)")


def _scheme_label(scheme: str, alpha) -> str:
    return "IID" if scheme == "iid" else f"Dirichlet a={alpha}"


def write_summary(rows: list[dict], path: str) -> None:
    """Per dataset x scheme aggregate markdown + the two headline claims."""
    lines: list[str] = []
    lines.append("# Per-client minority census (sampling_strategy = 0.01)\n")
    lines.append(f"Sweep: {len(DATASETS)} datasets x {len(SCHEMES)} schemes x "
                 f"{len(SEEDS)} seeds x {NUM_CLIENTS} clients "
                 f"= {len(rows)} client-instances.\n")
    lines.append("Per dataset x scheme: aggregated over "
                 f"{len(SEEDS)*NUM_CLIENTS} client-instances "
                 f"({len(SEEDS)} seeds x {NUM_CLIENTS} clients).\n")

    header = ("| Dataset | Scheme | clients<10 | clients=0 | min | median | max "
              "| worst mult | fires | skip:insuff | skip:target |")
    sep = ("|---------|--------|-----------|-----------|-----|--------|-----|"
           "-----------|-------|-------------|-------------|")
    lines.append(header)
    lines.append(sep)

    def group(name, scheme, alpha):
        return [r for r in rows if r["dataset"] == name and r["scheme"] == scheme
                and str(r["alpha"]) == ("" if alpha is None else str(alpha))]

    for name in DATASETS:
        for scheme, alpha in SCHEMES:
            g = group(name, scheme, alpha)
            if not g:
                continue
            mins = [r["n_minority"] for r in g]
            mult = [r["synthesis_multiplier"] for r in g]
            lines.append(
                f"| {name} | {_scheme_label(scheme, alpha)} | "
                f"{sum(r['flag_lt10'] for r in g)} | "
                f"{sum(r['flag_zero'] for r in g)} | "
                f"{min(mins)} | {int(np.median(mins))} | {max(mins)} | "
                f"x{max(mult):.0f} | "
                f"{sum(r['smote_fires'] for r in g)} | "
                f"{sum(r['skip_reason']=='insufficient_minority' for r in g)} | "
                f"{sum(r['skip_reason']=='target_met' for r in g)} |"
            )

    # Claim 1: ULB the most exposed under alpha=0.5.
    lines.append("\n## Claim 1 — most-exposed dataset (alpha=0.5)\n")
    for name in DATASETS:
        g = group(name, "dirichlet", 0.5)
        if not g:
            continue
        zeros = sum(r["flag_zero"] for r in g)
        lt10 = sum(r["flag_lt10"] for r in g)
        lines.append(f"- **{name}** a=0.5: {zeros} zero-minority, {lt10} <10-minority "
                     f"client-instances (of {len(g)}); min minority "
                     f"= {min(r['n_minority'] for r in g)}.")

    # Claim 2: target_met dominance where base prevalence exceeds the target.
    lines.append("\n## Claim 2 — `target_met` skips per dataset x scheme\n")
    for name in DATASETS:
        for scheme, alpha in SCHEMES:
            g = group(name, scheme, alpha)
            if not g:
                continue
            tm = sum(r["skip_reason"] == "target_met" for r in g)
            lines.append(f"- {name} {_scheme_label(scheme, alpha)}: "
                         f"{tm}/{len(g)} skip via target_met.")

    # Identity flag: cells where SMOTE fires on 0 clients (SMOTE == no-SMOTE).
    lines.append("\n## Cells where SMOTE fires on ZERO clients "
                 "(SMOTE arm == no-SMOTE arm)\n")
    any_identity = False
    for name in DATASETS:
        for scheme, alpha in SCHEMES:
            g = group(name, scheme, alpha)
            if g and sum(r["smote_fires"] for r in g) == 0:
                any_identity = True
                lines.append(f"- {name} {_scheme_label(scheme, alpha)}: "
                             f"0/{len(g)} clients oversample.")
    if not any_identity:
        lines.append("- (none)")

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[census] wrote {path}")
    # Also echo the summary to stdout.
    print("\n" + "\n".join(lines))


def main() -> None:
    t0 = time.time()
    rows = collect_rows()
    write_csv(rows, os.path.join(OUT_DIR, "minority_census.csv"))
    write_summary(rows, os.path.join(OUT_DIR, "minority_census_summary.md"))
    print(f"\n[census] total elapsed {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
