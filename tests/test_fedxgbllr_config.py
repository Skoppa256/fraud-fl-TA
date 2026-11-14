"""FedXGBllr's settings are the seed-42 record's, plus base_score 0.5 (2026-11-13).

The hyperparameters are the ones BAB 3 lists (tab-3-6), per dataset; the trees are
fixed after round 0 (tests/test_determinism.py). base_score 0.5 is the default of
xgboost < 2.0. xgboost 2.0 estimates an intercept per client instead, and every
single-tree margin — every input of the CNN — then carries it. That is checked on a
booster trained through the real config, not only in the YAML. Runnable via
``pytest`` or ``python tests/<file>.py``.
"""

import json
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "models", "fedxgbllr"))

from hydra import compose, initialize_config_dir  # noqa: E402

CONF = os.path.join(ROOT, "models", "fedxgbllr", "hfedxgboost", "conf")
ROUNDS = {"paysim": 20, "creditcard": 50, "baf": 50}


def _cfg(dataset):
    with initialize_config_dir(CONF, version_base=None):
        return compose("base", overrides=[f"dataset={dataset}", f"clients={dataset}_5_clients"])


def test_settings_are_the_seed42_records():
    for ds, rounds in ROUNDS.items():
        c = _cfg(ds)
        assert (c.n_estimators_client, c.XGBoost.max_depth, c.clients.CNN.lr) == (50, 6, 0.0005), ds
        assert (c.run_experiment.fit_config.num_iterations, c.run_experiment.batch_size) == (50, 32), ds
        assert (c.run_experiment.num_rounds, c.dataset.early_stop_patience_rounds) == (rounds, 10), ds
        assert c.XGBoost.base_score == 0.5, ds


def test_wandb_mode_is_left_to_the_runner():
    """An explicit mode beats WANDB_MODE, so --offline would not reach FedXGBllr."""
    assert "mode" not in _cfg("creditcard").wandb.setup


def test_trained_trees_carry_no_intercept():
    from hfedxgboost.models import fit_xgboost
    rng = np.random.default_rng(0)
    x = rng.normal(size=(400, 3)).astype(np.float32)
    y = (rng.random(400) < 0.05).astype(np.float32)          # imbalanced, like the fraud data
    c = _cfg("creditcard")
    tree = fit_xgboost(c, "BINARY", x, y, c.n_estimators_client)
    learner = json.loads(tree.get_booster().save_config())["learner"]
    assert float(learner["learner_model_param"]["base_score"]) == 0.5


if __name__ == "__main__":
    for fn in (test_settings_are_the_seed42_records, test_wandb_mode_is_left_to_the_runner,
               test_trained_trees_carry_no_intercept):
        fn()
        print(f"PASS {fn.__name__}")
