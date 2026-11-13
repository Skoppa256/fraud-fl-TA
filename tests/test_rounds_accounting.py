"""rounds_completed / rounds_configured are consistent across models.

rounds_completed = federated FIT rounds (round >= 1), EXCLUDING the round-0
initial evaluation. rounds_configured = the FL round budget (20 for most, 50 for
FedXGBllr, "n/a" for centralized). The per-round CSV contains only fit rounds.
Runnable via ``pytest`` or ``python tests/<file>.py``.
"""

import contextlib
import csv
import os
import sys
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from evaluation.results_paths import RESULTS_ROOT_ENV  # noqa: E402
from evaluation.results_writer import (  # noqa: E402
    SUMMARY_COLUMNS,
    write_centralized_results,
    write_fl_results,
)

_DS = "__test_rounds__"


@contextlib.contextmanager
def _tmp_results_root():
    """A throwaway output tree: the default results/logs/ is frozen and refuses writes."""
    prev = os.environ.get(RESULTS_ROOT_ENV)
    with tempfile.TemporaryDirectory() as d:
        os.environ[RESULTS_ROOT_ENV] = d
        try:
            yield d
        finally:
            if prev is None:
                os.environ.pop(RESULTS_ROOT_ENV, None)
            else:
                os.environ[RESULTS_ROOT_ENV] = prev


def _rows(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def _fl_history(last_round):
    # rounds 0..last_round inclusive (round 0 = initial eval).
    return [
        {"round": r, "val_auprc": 0.5, "val_f1": 0.1,
         "val_precision": 0.1, "val_recall": 0.1}
        for r in range(0, last_round + 1)
    ]


def test_schema_has_rounds_configured_not_num_rounds():
    assert "rounds_configured" in SUMMARY_COLUMNS
    assert "num_rounds" not in SUMMARY_COLUMNS


def test_fl_excludes_round0_and_records_configured():
    """LR: history rounds 0..20 → completed 20 (not 21), configured 20."""
    with _tmp_results_root():
        out = write_fl_results(
            model="lr", scheme="iid", alpha=None, oversampling="none", seed=42,
            num_rounds=20, num_clients=5, best_round=20, best_val_auprc=0.5,
            history=_fl_history(20), final_test={"test_auprc": 0.5},
            duration_seconds=1.0, dataset=_DS, data_hash="d", partition_hash="p",
        )
        srow = _rows(out["summary"])[-1]
        assert int(srow["rounds_completed"]) == 20   # round 0 excluded (was 21)
        assert int(srow["rounds_configured"]) == 20
        rrows = _rows(out["rounds"])
        assert len(rrows) == 20 and all(int(r["round"]) >= 1 for r in rrows)


def test_fedxgbllr_early_stopped_completed_vs_configured():
    """FedXGBllr early-stopped at fit round 12 (history 0..12) → completed 12,
    configured 50 — the asymmetric budget is explicit, not conflated."""
    with _tmp_results_root():
        out = write_fl_results(
            model="fedxgbllr", scheme="dirichlet", alpha=0.5, oversampling="none",
            seed=42, num_rounds=50, num_clients=5, best_round=11, best_val_auprc=0.7,
            history=_fl_history(12), final_test={"test_auprc": 0.7},
            duration_seconds=1.0, dataset=_DS, data_hash="d", partition_hash="p",
        )
        srow = _rows(out["summary"])[-1]
        assert int(srow["rounds_completed"]) == 12
        assert int(srow["rounds_configured"]) == 50


def test_centralized_rounds_are_na():
    with _tmp_results_root():
        out = write_centralized_results(
            model="lr", oversampling="none", seed=42,
            val_metrics={"val_auprc": 0.5}, test_metrics={"test_auprc": 0.5},
            duration_seconds=1.0, dataset=_DS,
        )
        srow = _rows(out["summary"])[-1]
        assert srow["rounds_completed"] == "n/a"
        assert srow["rounds_configured"] == "n/a"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nAll {len(fns)} rounds-accounting tests passed.")
