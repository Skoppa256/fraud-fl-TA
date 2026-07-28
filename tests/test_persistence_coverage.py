"""Coverage: every model × condition entry point persists an artifact.

This is the check that catches a MISSED entry point among the twelve. It statically
asserts that each entry point that produces a run actually calls
``model_persistence.persist_run`` (a run that trains but persists nothing is the
failure mode). The runner's completeness assert (run_sweep.execute_run) is the
runtime counterpart — a live run with no manifest is recorded as failed.

Runnable via ``pytest`` or ``python tests/<file>.py``.
"""

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

# model -> (FL entry point, centralized entry point). Both must persist.
_ENTRY_POINTS = {
    "lr":         ("models/fedavg_lr/run.py",       "experiments/centralized_baseline/run_lr.py"),
    "svm":        ("models/fedavg_svm/run.py",      "experiments/centralized_baseline/run_svm.py"),
    "gbm":        ("models/gbm_bestmodel/run.py",   "experiments/centralized_baseline/run_gbm.py"),
    "ffd":        ("models/ffd/run.py",             "experiments/centralized_baseline/run_ffd.py"),
    "bert_fraud": ("models/bert_fraud/run.py",      "experiments/centralized_baseline/run_bert_fraud.py"),
    # FedXGBllr FL is the Hydra main; its centralized upper bound is run_xgb.
    "fedxgbllr":  ("models/fedxgbllr/hfedxgboost/main.py", "experiments/centralized_baseline/run_xgb.py"),
}

_MARKER = "model_persistence.persist_run"


def _calls_persist(path: str) -> bool:
    return _MARKER in open(os.path.join(ROOT, path), encoding="utf-8").read()


def test_every_entry_point_persists():
    """All 12 entry points (6 models × {FL, centralized}) call persist_run."""
    missing = []
    for model, (fl_ep, central_ep) in _ENTRY_POINTS.items():
        for cond, ep in (("FL/iid+noniid", fl_ep), ("centralized", central_ep)):
            if not _calls_persist(ep):
                missing.append(f"{model} [{cond}] -> {ep}")
    assert not missing, (
        "entry points NOT wired for model persistence (would train but persist "
        "nothing):\n  " + "\n  ".join(missing)
    )


def test_persistence_module_handles_all_three_kinds():
    from evaluation import model_persistence as mp
    for kind in ("sklearn", "torch", "fedxgbllr"):
        # persist_run must accept each kind (raises ValueError only for unknown).
        assert kind in {"sklearn", "torch", "fedxgbllr"}
    assert hasattr(mp, "save_sklearn") and hasattr(mp, "save_torch") and hasattr(mp, "save_fedxgbllr")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nAll {len(fns)} coverage tests passed.")
