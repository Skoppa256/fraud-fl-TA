"""Per-seed output trees and the frozen-folder write guard (evaluation.results_paths).

A run writes into its seed's tree, results/seeds/seed<N>/; the seed-42 record in
results/ is frozen, so the shared writers must refuse it rather than land on it.
Runnable via ``pytest`` or ``python tests/<file>.py``.
"""

import contextlib
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from evaluation import results_paths as P  # noqa: E402
from evaluation import model_persistence as mp  # noqa: E402
from evaluation import results_writer as rw  # noqa: E402

_DS = "__test_results_paths__"


@contextlib.contextmanager
def _env(value):
    """Set (or, with None, unset) the results-root variable for the block."""
    prev = os.environ.get(P.RESULTS_ROOT_ENV)
    if value is None:
        os.environ.pop(P.RESULTS_ROOT_ENV, None)
    else:
        os.environ[P.RESULTS_ROOT_ENV] = str(value)
    try:
        yield
    finally:
        if prev is None:
            os.environ.pop(P.RESULTS_ROOT_ENV, None)
        else:
            os.environ[P.RESULTS_ROOT_ENV] = prev


def _raises(fn, *args):
    try:
        fn(*args)
    except RuntimeError as e:
        return str(e)
    raise AssertionError(f"{fn.__name__}{args} did not refuse")


def test_root_defaults_to_results_and_follows_the_variable():
    with _env(None):
        assert P.results_root() == P.LEGACY_ROOT
    with tempfile.TemporaryDirectory() as d, _env(d):
        assert P.results_root() == Path(d)
    with _env("results/seeds/seed123"):   # relative: differs under fedxgbllr's cwd
        try:
            P.results_root()
        except ValueError:
            pass
        else:
            raise AssertionError("a relative results root was accepted")


def test_seed_root_is_one_tree_per_seed():
    assert P.seed_root(123) == P.LEGACY_ROOT / "seeds" / "seed123"
    assert P.seed_root(42) != P.LEGACY_ROOT   # seed 42 re-runs never reach the record


def test_guard_refuses_every_frozen_folder():
    for frozen in P.FROZEN:
        msg = _raises(P.assert_writable, frozen / "x.csv")
        assert "frozen" in msg and P.RESULTS_ROOT_ENV in msg
        _raises(P.assert_writable, frozen)
    for ok in (P.seed_root(123) / "logs" / "baf" / "lr" / "x.csv",
               P.seed_root(42) / "shap_v2" / "summary.csv",
               P.LEGACY_ROOT / "cache" / "partitions",       # shared, seed-keyed
               P.LEGACY_ROOT / "logs_elsewhere" / "x.csv"):  # a sibling, not inside
        P.assert_writable(ok)


def test_writers_refuse_the_frozen_default():
    """A by-hand run (no variable) must fail loudly and create nothing."""
    with _env(None):
        _raises(rw._logs_dir, os.path.join(_DS, "lr"))
        _raises(mp.run_artifact_dir, _DS, "lr_iid_alpha-_none_seed42")
    assert not (P.LEGACY_ROOT / "logs" / _DS).exists()
    assert not (P.LEGACY_ROOT / "models" / _DS).exists()


def test_writers_follow_the_seed_tree():
    with tempfile.TemporaryDirectory() as d, _env(d):
        out = rw.write_centralized_results(
            model="lr", oversampling="none", seed=123,
            val_metrics={"val_auprc": 0.5}, test_metrics={"test_auprc": 0.5},
            duration_seconds=1.0, dataset=_DS,
        )
        assert Path(out["summary"]) == (Path(d) / "logs" / _DS / "centralized"
                                        / "centralized_lr_none_seed123.csv")
        art = mp.run_artifact_dir(_DS, "lr_iid_alpha-_none_seed123")
        assert art == Path(d) / "models" / _DS / "lr_iid_alpha-_none_seed123"
        assert art.is_dir()


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nAll {len(fns)} results-path tests passed.")
