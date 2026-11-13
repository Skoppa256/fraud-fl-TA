"""experiments.shap_rq3 explains one training seed, inside that seed's tree only.

``--seed`` is required, and configure() must point every input and output at
results/seeds/seed<N>/: the models it reads, the split/partition/backgrounds it
explains (shap_analysis.SEED), and the shap_v2/ it writes. Unconfigured, it must
refuse to write rather than fall back to the frozen results/shap_v2/. No shap and
no model artifacts needed. Runnable via ``pytest`` or ``python tests/<file>.py``.
"""

import contextlib
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from evaluation import results_paths as P  # noqa: E402
from experiments import shap_analysis as SA  # noqa: E402
from experiments import shap_rq3 as R  # noqa: E402

ART = {"dataset": "paysim", "model": "ffd", "condition": "iid", "arm": "none",
       "alpha": None, "dir": "unused"}


@contextlib.contextmanager
def _restored():
    """Leave shap_rq3 / shap_analysis module state as the test found it."""
    saved = (R.SEED, R.MODELS, R.OUT, R.SUMMARY, SA.SEED)
    try:
        yield
    finally:
        R.SEED, R.MODELS, R.OUT, R.SUMMARY, SA.SEED = saved


def test_seed_is_required():
    try:
        R.main(["--stage", "main"])
    except SystemExit as e:
        assert e.code == 2           # argparse usage error, before any work
    else:
        raise AssertionError("shap_rq3 ran without --seed")


def test_configure_points_everything_at_the_seed_tree():
    with _restored():
        for seed in (123, 42):
            R.configure(seed)
            tree = P.seed_root(seed)
            assert R.SEED == SA.SEED == seed
            assert R.MODELS == tree / "models"
            assert R.OUT == tree / "shap_v2"
            assert R.SUMMARY == tree / "shap_v2" / "shap_summary_v2.csv"
            assert R.cell_dir(ART, "local") == tree / "shap_v2" / "paysim" / "ffd" / "iid_none"
            P.assert_writable(R.SUMMARY)     # never the frozen results/shap_v2/


def test_unconfigured_runner_refuses_to_write():
    with _restored():
        R.OUT = R.SUMMARY = R.MODELS = None
        for call in (lambda: R.cell_dir(ART, "local"), lambda: R.write_summary([]),
                     lambda: R._discover()):
            try:
                call()
            except RuntimeError as e:
                assert "not configured" in str(e)
            else:
                raise AssertionError("an unconfigured shap_rq3 call went through")


def test_discover_keeps_only_the_requested_seed():
    names = ["lr_iid_alpha-_none_seed123", "centralized_lr_none_seed123",
             "lr_iid_alpha-_none_seed42", "lr_iid_alpha-_none_seed1234"]
    with tempfile.TemporaryDirectory() as d:
        for n in names:
            (Path(d) / "paysim" / n).mkdir(parents=True)
            (Path(d) / "paysim" / n / "manifest.json").write_text(json.dumps({}))
        got = sorted(a["run_name"] for a in SA.discover(root=d, seed=123))
        assert got == ["centralized_lr_none_seed123", "lr_iid_alpha-_none_seed123"]
        assert len(SA.discover(root=d)) == 4    # no seed: everything, as before


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nAll {len(fns)} RQ3 seed tests passed.")
