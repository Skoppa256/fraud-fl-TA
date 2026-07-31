"""Rounds-schema migration: correct per-model offset, idempotent, fills configured.

Builds a synthetic OLD-format results tree in a temp dir and checks the migration
rewrites it to the new schema. The round-0 offset is per-model (GBM 0, others 1),
so the test covers LR (offset 1), GBM (offset 0), FedXGBllr (offset 1),
centralized (n/a), and an already-new row (skipped). Runnable via pytest/python.
"""

import csv
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from evaluation.results_writer import SUMMARY_COLUMNS, ROUND_COLUMNS  # noqa: E402
from experiments.run_sweep import _MASTER_COLUMNS, _child_run_name, RunSpec  # noqa: E402
from experiments import migrate_rounds_schema as M  # noqa: E402

_OLD_SUMMARY_HEADER = [
    ("num_rounds" if c == "rounds_configured" else c)
    for c in SUMMARY_COLUMNS if c != "n_iter_selected"
]
_OLD_MASTER_HEADER = [c for c in _MASTER_COLUMNS if c != "rounds_configured"]


def _write(path: Path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in header})


def _read(path: Path):
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        return list(r.fieldnames or []), list(r)


def _fl_summary(model, scheme, num_rounds, rounds_completed_old):
    return {"dataset": "creditcard", "model": model, "scheme": scheme, "alpha": "",
            "num_rounds": num_rounds, "num_clients": 5, "rounds_completed": rounds_completed_old}


def _rounds_rows(lo, hi):
    return [{"round": r, "val_auprc": 0.5, "val_f1": 0.1,
             "val_precision": 0.1, "val_recall": 0.1, "train_loss": ""}
            for r in range(lo, hi + 1)]


def _spec(model, scheme):
    cond = "iid" if scheme == "iid" else scheme
    return RunSpec("creditcard", model, "no-smote", cond, seed=42, alpha=None)


def _build_tree(root: Path):
    logs = root / "logs" / "creditcard"
    # FL cells at canonical paths so both summary sibling and master lookup resolve.
    specs = {
        "lr": (_spec("lr", "iid"), "iid", 20, 21, (0, 20)),        # offset 1 → 20
        "gbm": (_spec("gbm", "iid"), "iid", 20, 15, (1, 15)),      # offset 0 → 15
        "fedxgbllr": (_spec("fedxgbllr", "iid"), "iid", 50, 38, (0, 37)),  # offset 1 → 37
    }
    for model, (spec, scheme, nr, rc_old, (lo, hi)) in specs.items():
        subdir = "fedxgbllr" if model == "fedxgbllr" else model
        run = _child_run_name(spec)
        _write(logs / subdir / f"{run}.csv", _OLD_SUMMARY_HEADER,
               [_fl_summary(model, scheme, nr, rc_old)])
        _write(logs / subdir / f"{run}_rounds.csv", list(ROUND_COLUMNS),
               _rounds_rows(lo, hi))
    # centralized summary (old: num_rounds blank, rounds_completed n/a, no sibling)
    _write(logs / "centralized" / "centralized_lr_none_seed42.csv", _OLD_SUMMARY_HEADER,
           [{"dataset": "creditcard", "model": "lr", "scheme": "centralized",
             "num_rounds": "", "rounds_completed": "n/a"}])
    # already-new summary → must be skipped untouched
    _write(logs / "svm" / "svm_iid_alpha-_none_seed42.csv", list(SUMMARY_COLUMNS),
           [{"dataset": "creditcard", "model": "svm", "scheme": "iid",
             "rounds_configured": 20, "rounds_completed": 20, "n_iter_selected": "n/a"}])
    # old master
    master_rows = [
        {"run_name": "c_lr", "dataset": "creditcard", "model": "lr",
         "smote_arm": "no-smote", "condition": "iid", "alpha": "", "seed": 42,
         "status": "success", "rounds_completed": 21},
        {"run_name": "c_gbm", "dataset": "creditcard", "model": "gbm",
         "smote_arm": "no-smote", "condition": "iid", "alpha": "", "seed": 42,
         "status": "success", "rounds_completed": 15},
        {"run_name": "c_cen", "dataset": "creditcard", "model": "lr",
         "smote_arm": "no-smote", "condition": "centralized", "alpha": "", "seed": 42,
         "status": "success", "rounds_completed": "n/a"},
    ]
    _write(root / "sweep" / "sweep_master.csv", _OLD_MASTER_HEADER, master_rows)


def _summary_row(root, model):
    subdir = "fedxgbllr" if model == "fedxgbllr" else model
    run = _child_run_name(_spec(model, "iid"))
    hdr, rows = _read(root / "logs" / "creditcard" / subdir / f"{run}.csv")
    return hdr, rows[-1]


def test_migration_end_to_end_and_idempotent():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        _build_tree(root)
        M.run_migration(root, dry=False)

        # --- summaries: correct per-model rounds_completed + rounds_configured ---
        hdr, lr = _summary_row(root, "lr")
        assert "num_rounds" not in hdr and "rounds_configured" in hdr
        assert hdr == list(SUMMARY_COLUMNS)          # rewritten to current schema
        assert int(lr["rounds_completed"]) == 20     # 21 → 20 (offset 1)
        assert int(lr["rounds_configured"]) == 20
        assert lr["n_iter_selected"] == "n/a"        # new column filled

        _hdr, gbm = _summary_row(root, "gbm")
        assert int(gbm["rounds_completed"]) == 15    # unchanged (offset 0)
        assert int(gbm["rounds_configured"]) == 20

        _hdr, fx = _summary_row(root, "fedxgbllr")
        assert int(fx["rounds_completed"]) == 37     # 38 → 37 (offset 1)
        assert int(fx["rounds_configured"]) == 50

        _hdr, cen = _read(root / "logs" / "creditcard" / "centralized"
                          / "centralized_lr_none_seed42.csv")
        cen = cen[-1]
        assert cen["rounds_completed"] == "n/a" and cen["rounds_configured"] == "n/a"

        # --- per-round CSVs: round-0 dropped for LR/FedXGBllr, GBM unchanged ---
        def _rounds(model):
            subdir = "fedxgbllr" if model == "fedxgbllr" else model
            run = _child_run_name(_spec(model, "iid"))
            _h, rr = _read(root / "logs" / "creditcard" / subdir / f"{run}_rounds.csv")
            return [int(x["round"]) for x in rr]
        assert min(_rounds("lr")) == 1 and len(_rounds("lr")) == 20
        assert min(_rounds("fedxgbllr")) == 1 and len(_rounds("fedxgbllr")) == 37
        assert min(_rounds("gbm")) == 1 and len(_rounds("gbm")) == 15  # never had round 0

        # --- master: rounds_configured added, rounds_completed fixed ---
        mhdr, mrows = _read(root / "sweep" / "sweep_master.csv")
        assert "rounds_configured" in mhdr
        by_model = {r["model"] + ":" + r["condition"]: r for r in mrows}
        assert int(by_model["lr:iid"]["rounds_completed"]) == 20
        assert int(by_model["lr:iid"]["rounds_configured"]) == 20
        assert int(by_model["gbm:iid"]["rounds_completed"]) == 15
        assert by_model["lr:centralized"]["rounds_completed"] == "n/a"
        assert by_model["lr:centralized"]["rounds_configured"] == "n/a"

        # --- idempotent: second run changes nothing ---
        before = {p: p.read_bytes() for p in root.rglob("*.csv")}
        tally = M.run_migration(root, dry=False)
        after = {p: p.read_bytes() for p in root.rglob("*.csv")}
        assert before == after, "second migration must be a no-op"
        assert all(not k.endswith(":migrated") for k in tally), tally


def test_already_new_summary_skipped():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        p = root / "logs" / "creditcard" / "svm" / "svm_iid_alpha-_none_seed42.csv"
        _write(p, list(SUMMARY_COLUMNS),
               [{"model": "svm", "scheme": "iid", "rounds_configured": 20,
                 "rounds_completed": 20, "n_iter_selected": "n/a"}])
        assert M.migrate_summary(p, dry=False) == "skip-new"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nAll {len(fns)} migration tests passed.")
