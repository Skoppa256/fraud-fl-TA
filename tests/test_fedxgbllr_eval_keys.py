"""FedXGBllr's server-side eval reads only keys that metrics_at_threshold returns.

serverside_eval indexes the dicts ``vm`` / ``tm`` that
evaluation.metrics.metrics_at_threshold returns. A key it does not return raises
KeyError inside Flower's evaluate_fn at round 0, which Flower reports as an
out-of-memory crash: every FedXGBllr run fails with no results row. That happened
with ``tm["test_threshold_at_fpr"]`` (found 2026-11-13; latent since Recall@5%FPR
was added, after the seed-42 sweep, whose values were backfilled instead). The
function needs a trained ensemble + CNN to call, so this checks the source, the
same way test_persistence_coverage does. Runnable via ``pytest`` or
``python tests/<file>.py``.
"""

import os
import re
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from evaluation.metrics import metrics_at_threshold  # noqa: E402

SERVER = os.path.join(ROOT, "models", "fedxgbllr", "hfedxgboost", "server.py")


def test_server_reads_only_returned_keys():
    returned = set(metrics_at_threshold([0, 1, 0, 1], [0.1, 0.9, 0.2, 0.8], 0.5))
    src = open(SERVER, encoding="utf-8").read()
    assert re.search(r"\b[tv]m\s*=\s*fair_metrics\.metrics_at_threshold\(", src), \
        "vm/tm no longer come from metrics_at_threshold — update this test"
    read = set(re.findall(r'\b[tv]m\[\s*"([A-Za-z_]+)"\s*\]', src))
    assert {"recall_at_fpr", "threshold_at_fpr", "actual_fpr"} <= read
    assert read <= returned, f"keys read but never returned: {sorted(read - returned)}"


if __name__ == "__main__":
    test_server_reads_only_returned_keys()
    print("PASS test_server_reads_only_returned_keys")
