"""Backfill Recall@5%FPR into frozen result CSVs — RUN ON THE BOX where the
``results/models/`` artifacts live.

READ-ONLY against every frozen input: it only *reads* the persisted final models
under ``results/models/`` and the preprocessed data cache, exactly like
``experiments/shap_stage0_probe.py`` (whose verified loaders it reuses). It never
refits, retrains, or edits any model artifact or the data cache.

The ONLY writes it performs are into result CSVs, and only into the blank
Recall@5%FPR columns added by ``evaluation.results_writer`` —
``test_recall_at_fpr`` / ``test_threshold_at_fpr`` / ``test_actual_fpr`` (plus
``best_val_recall_at_fpr`` for centralized runs). It refuses to overwrite any
non-blank cell and asserts every pre-existing value is byte-identical after the
rewrite, backing each CSV up to ``<file>.bak-recallfpr`` first. Re-running is
idempotent (already-filled cells are left untouched).

Why this reproduces the frozen numbers rather than inventing them
----------------------------------------------------------------
Recall@5%FPR is threshold-free: it needs only ``(y_test, test_scores)``. The
persisted model IS the exact final global model whose ``test_*`` metrics were
logged, so recomputing its test scores reproduces the operating point. As a hard
guard, each row is filled ONLY if the recomputed AUPRC matches the row's stored
``test_auprc`` within ``--auprc-tol`` (default 2e-3). A mismatch means the
model/data pairing is wrong for that row, so the row is skipped, never written.

``best_val_recall_at_fpr`` is only backfilled for CENTRALIZED runs, where val and
test are scored by the same persisted model. For FL runs the ``best_val_*`` value
came from the best-round model (not persisted), so it is left blank — forward runs
populate it. Per-round ``val_recall_at_fpr`` likewise cannot be backfilled (no
per-round models are persisted).

Usage
-----
    python experiments/backfill_recall_at_fpr.py                 # DRY RUN: print plan, write nothing
    python experiments/backfill_recall_at_fpr.py --apply         # fill results/clean_summary.csv
    python experiments/backfill_recall_at_fpr.py --apply --per-run   # also upgrade+fill per-run summary CSVs
    python experiments/backfill_recall_at_fpr.py --apply --auprc-tol 5e-3
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "models" / "fedxgbllr"))  # hfedxgboost.*

from evaluation.metrics import auprc, recall_at_fpr  # noqa: E402
from experiments import shap_stage0_probe as probe  # noqa: E402 — reuse read-only loaders

CLEAN_SUMMARY = PROJECT_ROOT / "results" / "clean_summary.csv"
NEW_COLS = ("test_recall_at_fpr", "test_threshold_at_fpr", "test_actual_fpr")
BEST_VAL_COL = "best_val_recall_at_fpr"


# --------------------------------------------------------------------------- #
# READ-ONLY scoring: reload one artifact and return (val_scores, test_scores).
# Mirrors exactly how each model was scored during the run.
# --------------------------------------------------------------------------- #
def _xgb_force_cpu(estimator) -> None:
    """Pin an XGBoost estimator's booster to CPU so predicting on CPU arrays does
    not trip the "booster on cuda:0, data on cpu" fallback. Best-effort across
    xgboost versions; the AUPRC guard still validates the output either way."""
    try:
        estimator.set_params(device="cpu")
    except Exception:  # noqa: BLE001 — older xgboost has no `device` param
        pass
    try:
        estimator.get_booster().set_param({"device": "cpu"})
    except Exception:  # noqa: BLE001
        pass


def score_artifact(art, data):
    """Return ``(val_scores | None, test_scores)`` for a reloaded frozen model.

    Pure reads: loads the persisted model and scores the in-memory cache splits.
    """
    from evaluation import model_persistence as mp

    model = art["model"]
    xva = np.asarray(data["x_val"], np.float32)
    xte = np.asarray(data["x_test"], np.float32)

    if model in ("lr", "gbm", "xgb"):
        m = mp.load_sklearn(art["dir"])
        if model == "xgb":
            # The booster may have been trained on GPU (device=cuda:0); predicting
            # on CPU arrays otherwise triggers XGBoost's "mismatched devices"
            # fallback (correct, but noisy). Pin it to CPU to match the data.
            _xgb_force_cpu(m)
        return m.predict_proba(xva)[:, 1], m.predict_proba(xte)[:, 1]

    if model == "svm":
        m = mp.load_sklearn(art["dir"])  # margin is the ranking score (no predict_proba)
        return m.decision_function(xva), m.decision_function(xte)

    if model in ("ffd", "bert_fraud"):
        import torch
        if model == "ffd":
            from models.ffd.model import FFDModel as Cls
        else:
            from models.bert_fraud.model import BertFraudModel as Cls
        m = mp.load_torch(Cls, art["dir"], device="cpu")
        # Port of the shap_stage0_probe device fix. load_torch reconstructs the
        # module with its DEFAULT device (cuda when the box has a GPU) and merely
        # copies weights in, so after reload params/buffers sit on CUDA. Two moves
        # are needed to score on CPU arrays without a device clash:
        #   1. .cpu() moves params AND registered buffers to CPU — this is what
        #      fixes BERT (feature_weights / feature_biases / cls_token) reporting
        #      mixed devices.
        #   2. reset self.device — the cached attribute predict_proba uses to place
        #      the INPUT tensor (`x.to(self.device)`). Without it inputs go to CUDA
        #      while weights are on CPU — FFD's "data on CUDA, weights on CPU".
        m = m.cpu().eval()
        m.device = torch.device("cpu")
        return m.predict_proba(xva)[:, 1], m.predict_proba(xte)[:, 1]

    if model == "fedxgbllr":
        # Reuses the probe's verified CPU loader. Its per-client XGBoost boosters
        # may emit the same benign device-mismatch fallback as centralized xgb;
        # it does not affect output (the AUPRC guard validates every row).
        predict_proba, _ = probe.load_fedxgbllr_composed(art)
        return predict_proba(xva), predict_proba(xte)

    raise ValueError(f"unknown model kind {model!r} for {art['run_name']}")


def compute_row_values(art, data, y_val, y_test, auprc_tol):
    """Recompute the Recall@5%FPR triple for one artifact, guarded by an AUPRC
    cross-check against the stored value. Returns ``(values_dict, note)`` where
    ``values_dict`` is ``None`` if the row must be skipped."""
    val_scores, test_scores = score_artifact(art, data)
    rec_auprc = auprc(y_test, test_scores)

    stored = probe._stored_metric(art["dataset"], art["model"], art["scheme"], art["arm"], "test_auprc")
    if stored in (None, ""):
        return None, f"no stored test_auprc row to cross-check (skip)"
    if abs(rec_auprc - float(stored)) > auprc_tol:
        return None, (f"AUPRC mismatch recomputed={rec_auprc:.4f} vs stored={float(stored):.4f} "
                      f"(> {auprc_tol}); model/data pairing wrong — SKIP")

    t = recall_at_fpr(y_test, test_scores)
    vals = {
        "test_recall_at_fpr": t["recall_at_fpr"],
        "test_threshold_at_fpr": t["threshold_at_fpr"],
        "test_actual_fpr": t["actual_fpr"],
    }
    # best_val only where val & test share the persisted model (centralized).
    if art["scheme"] == "centralized":
        v = recall_at_fpr(y_val, val_scores)
        vals[BEST_VAL_COL] = v["recall_at_fpr"]
    return vals, f"OK (recomputed AUPRC {rec_auprc:.4f} ≈ stored {float(stored):.4f})"


def _fmt(x):
    return "" if isinstance(x, str) else repr(round(x, 6))


# --------------------------------------------------------------------------- #
# CSV writing — surgical, backed-up, verified. Fills ONLY blank new-metric cells.
# --------------------------------------------------------------------------- #
def _load_csv(path):
    with open(path, newline="") as f:
        r = csv.reader(f)
        rows = list(r)
    return rows[0], rows[1:]


def _write_csv(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def fill_csv(path, by_key, *, add_missing_cols, apply):
    """Fill the Recall@5%FPR columns for matching ``(dataset, run_name)`` rows.

    Keyed on ``(dataset, run_name)`` because the same ``run_name`` recurs across
    datasets (paysim/creditcard/baf) with different scores. Only blank cells are
    written; a non-blank cell is never touched. Every pre-existing value is
    asserted unchanged. Returns a list of ``(key, action)`` for the report.
    Writes nothing unless ``apply`` is True.
    """
    if not path.is_file():
        return [("-", f"{path} not found (skip)")]
    header, rows = _load_csv(path)
    orig = [list(r) for r in rows]  # snapshot for the no-clobber assertion

    if add_missing_cols:
        for c in NEW_COLS + (BEST_VAL_COL,):
            if c not in header:
                header.append(c)
                for r in rows:
                    r.append("")
    idx = {c: header.index(c) for c in header}
    rn_i = idx.get("run_name")
    ds_i = idx.get("dataset")
    if rn_i is None or ds_i is None:
        return [("-", f"{path} lacks a dataset/run_name column (skip)")]

    actions = []
    for r in rows:
        key = (r[ds_i], r[rn_i])
        vals = by_key.get(key)
        if vals is None:
            continue
        wrote = []
        for c, v in vals.items():
            if c not in idx:
                continue  # column not present and not adding it here
            ci = idx[c]
            cur = r[ci]
            cell = "" if isinstance(v, str) else f"{v:.6f}"  # NA stays "", floats fixed-format
            if isinstance(v, str):
                cell = v  # NA sentinel
            if cur.strip() == "":
                r[ci] = cell
                wrote.append(c)
            # else: already populated — leave as-is (idempotent)
        actions.append((key, "filled: " + ", ".join(wrote) if wrote else "already filled"))

    # No-clobber guarantee: every cell that existed before is unchanged except the
    # blank new-metric cells we filled.
    fillable = {idx[c] for c in (BEST_VAL_COL, *NEW_COLS) if c in idx}
    for r_new, r_old in zip(rows, orig):
        for j, old_v in enumerate(r_old):
            if j in fillable:
                continue
            assert r_new[j] == old_v, f"REFUSING: non-metric cell changed in {path} col {header[j]!r}"

    if apply and actions:
        bak = path.with_suffix(path.suffix + ".bak-recallfpr")
        if not bak.exists():
            bak.write_bytes(path.read_bytes())
        _write_csv(path, header, rows)
    return actions


def _summary_csv_path(art):
    subdir = ("centralized" if art["scheme"] == "centralized"
              else "fedxgbllr" if art["model"] == "fedxgbllr" else art["model"])
    return PROJECT_ROOT / "results" / "logs" / art["dataset"] / subdir / f"{art['run_name']}.csv"


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apply", action="store_true", help="Write cells (default: dry run, writes nothing).")
    ap.add_argument("--per-run", action="store_true",
                    help="Also upgrade+fill the per-run summary CSVs under results/logs/ (adds the columns).")
    ap.add_argument("--auprc-tol", type=float, default=2e-3,
                    help="Max |recomputed − stored| test_auprc to accept a row (default 2e-3).")
    ap.add_argument("--seed", type=int, default=42, help="Data-cache seed (default 42, matches the runs).")
    args = ap.parse_args()

    print("=" * 72)
    print("Backfill Recall@5%FPR — READ-ONLY on models/data; "
          + ("APPLY (will write CSV cells)" if args.apply else "DRY RUN (writes nothing)"))
    print("=" * 72)

    arts = probe.discover()
    if not arts:
        print("\nNO artifacts under results/models/ — run this ON THE BOX where the frozen "
              "models live. Nothing to backfill.")
        return 2

    # Load + cache each dataset's splits once; assert data_hash matches manifests.
    by_dataset = {}
    computed = {}   # run_name -> values dict
    for a in sorted(arts, key=lambda x: (x["dataset"], x["run_name"])):
        ds = a["dataset"]
        if ds not in by_dataset:
            from experiments import data_cache
            data, dh = data_cache.get_preprocessed(ds, args.seed)
            by_dataset[ds] = (data, dh, np.asarray(data["y_val"]).astype(int),
                              np.asarray(data["y_test"]).astype(int))
        data, dh, y_val, y_test = by_dataset[ds]

        man = json.loads((a["dir"] / "manifest.json").read_text())
        if man.get("data_hash", "")[:16] != dh[:16]:
            print(f"  [{ds}/{a['run_name']}] data_hash mismatch "
                  f"(manifest {man.get('data_hash','')[:12]} vs cache {dh[:12]}) — SKIP")
            continue
        try:
            vals, note = compute_row_values(a, data, y_val, y_test, args.auprc_tol)
        except Exception as exc:  # noqa: BLE001 — a broken artifact must not abort the sweep
            print(f"  [{ds}/{a['run_name']}] ERROR: {type(exc).__name__}: {str(exc)[:140]} — SKIP")
            continue
        if vals is None:
            print(f"  [{ds}/{a['run_name']}] {note}")
            continue
        computed[(ds, a["run_name"])] = vals
        print(f"  [{ds}/{a['run_name']}] {note}")
        print(f"      test_recall_at_fpr={_fmt(vals['test_recall_at_fpr'])} "
              f"thr={_fmt(vals['test_threshold_at_fpr'])} fpr={_fmt(vals['test_actual_fpr'])}"
              + (f" best_val_recall_at_fpr={_fmt(vals[BEST_VAL_COL])}" if BEST_VAL_COL in vals else ""))

    print(f"\nComputed values for {len(computed)}/{len(arts)} artifacts.")

    # 1) clean_summary.csv (already carries the columns; fill blanks).
    print("\n-- results/clean_summary.csv --")
    for key, act in fill_csv(CLEAN_SUMMARY, computed, add_missing_cols=False, apply=args.apply):
        print(f"    {key}: {act}")

    # 2) Optional: the per-run summary CSVs (old schema — add the columns, then fill).
    if args.per_run:
        print("\n-- per-run summary CSVs (results/logs/**) --")
        for (ds, rn), vals in computed.items():
            art = next(a for a in arts if a["dataset"] == ds and a["run_name"] == rn)
            p = _summary_csv_path(art)
            for _, act in fill_csv(p, {(ds, rn): vals}, add_missing_cols=True, apply=args.apply):
                print(f"    {p.relative_to(PROJECT_ROOT)}: {act}")

    print("\n" + "=" * 72)
    if args.apply:
        print("APPLIED. Backups written as <file>.bak-recallfpr. Re-run collect_results to refresh aggregates.")
    else:
        print("DRY RUN complete — nothing written. Re-run with --apply to fill the cells.")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
