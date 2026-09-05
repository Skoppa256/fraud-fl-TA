"""The exact tier must be measured, not asserted in prose.

``experiments/shap_rq3.py`` reports ``seed_delta_max`` — max|delta phi| between
the two SHAP-only coalition seeds — for every cell. The properties under test:

* the kernel path populates it at all (it used to write NaN, so the PaySim
  exact-grouped tier's zero-noise claim rested on a hand check of
  ``importance_per_client_seed.csv``);
* it is exactly 0.0 once ``nsamples`` enumerates all 2^M - 2 coalitions, which is
  what makes ``_det_stats``' hard-coded floor of 1.0 legitimate;
* a cell that claims determinism but does not reproduce aborts the run rather
  than publishing a floor it did not earn;
* enumeration is decided by ``nsamples``, never by the presence of a grouping —
  grouping only lowers the bar to 2^M_eff - 2, it does not clear it.

Needs shap (KernelExplainer is the object under test) but no model artifacts:
the predictor, backgrounds and feature names are injected.
"""

import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

pytest.importorskip("shap")

from experiments import shap_rq3 as R  # noqa: E402

M = 4                      # 2^4 - 2 = 14 coalitions: enumerable in a unit test
EXACT_NSAMPLES = 2 ** M - 2
ART = {"dataset": "paysim", "model": "ffd", "condition": "iid", "arm": "none",
       "alpha": None, "dir": "unused"}


def _predict(x):
    """Deterministic, non-additive, non-degenerate — every feature contributes."""
    x = np.atleast_2d(np.asarray(x, float))
    return (x[:, 0] + 0.7 * x[:, 1] - 0.4 * x[:, 2] * x[:, 3]
            + 0.3 * np.tanh(x[:, 1] * x[:, 2]))


@pytest.fixture
def cell(monkeypatch):
    """Two clients with genuinely different backgrounds, one shared X."""
    rng = np.random.default_rng(0)
    bgs = [rng.normal(0, 1, (40, M)), rng.normal(1.5, 2.0, (40, M))]
    monkeypatch.setattr(R, "client_backgrounds", lambda art, rng_: (bgs, "phash"))
    monkeypatch.setattr(R, "load_predictor", lambda art: (_predict, "kernel"))
    monkeypatch.setattr(R, "feature_names", lambda art: [f"f{j}" for j in range(M)])
    monkeypatch.setattr(R, "manifest_hash", lambda d: "0" * 64)
    return rng.normal(0.2, 1.0, (6, M))


def _run(X, tmp_path, nsamples):
    row = R.process_cell(ART, "local", X, "d" * 32, nsamples=nsamples, root=tmp_path)
    payload = json.loads((R.cell_dir(ART, "local", tmp_path) / "stability.json").read_text())
    return row, payload


def test_full_enumeration_is_bit_identical(cell, tmp_path):
    row, payload = _run(cell, tmp_path, EXACT_NSAMPLES)
    assert payload["deterministic"] is True
    assert payload["seed_delta_max"] == 0.0        # not None, not NaN, not ~0
    assert row["seed_delta_max"] == 0.0            # and it reaches the summary CSV


def test_sampled_cell_reports_a_finite_nonzero_delta(cell, tmp_path):
    row, payload = _run(cell, tmp_path, 8)         # 8 < 14: coalitions are drawn
    assert payload["deterministic"] is False
    d = payload["seed_delta_max"]
    assert d is not None and np.isfinite(d) and d > 0.0
    assert row["seed_delta_max"] != "undefined"


def test_grouping_alone_does_not_certify_exactness(cell, tmp_path):
    """M_eff = 2 needs nsamples >= 2, but the cell is run at 1: still sampled."""
    groups, gnames = [[0, 1], [2, 3]], ["g01", "g23"]
    row = R.process_cell(ART, "local", cell, "d" * 32, nsamples=1, groups=groups,
                         gnames=gnames, root=tmp_path)
    payload = json.loads((R.cell_dir(ART, "local", tmp_path) / "stability.json").read_text())
    assert payload["deterministic"] is False, "grouping was read as enumeration"
    assert row["n_clients"] == 2


def test_nonreproducible_deterministic_cell_aborts(cell, tmp_path, monkeypatch):
    """Perturb the second seed's attributions: the exactness guard must fire."""
    real = R.kernel_g
    calls = {"n": 0}

    def flaky(obj, bg_km, X, seed, nsamples):
        g, nz, sv = real(obj, bg_km, X, seed, nsamples)
        calls["n"] += 1
        if calls["n"] % 2 == 0:                    # every client's second seed
            sv = sv + 1e-9
        return g, nz, sv

    monkeypatch.setattr(R, "kernel_g", flaky)
    with pytest.raises(RuntimeError, match="zero estimator noise"):
        _run(cell, tmp_path, EXACT_NSAMPLES)
