"""Round-trip tests for model persistence (commit 5): save -> reload -> assert.

Bitwise for sklearn and tree models; within tolerance for deep models. FedXGBllr
is the one to watch — its artifact spans two stages (per-client boosters + CNN),
so we verify the reloaded ensemble+CNN reproduces the composed prediction, using
the real margin builder, not just that the files load.

Runnable via ``pytest`` or ``python tests/<file>.py``.
"""

import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "fedxgbllr")))

from evaluation import model_persistence as mp  # noqa: E402


def test_sklearn_bitwise():
    from sklearn.linear_model import LogisticRegression
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 8)).astype(np.float32)
    y = (rng.random(200) < 0.3).astype(np.int32)
    model = LogisticRegression(max_iter=500).fit(X, y)
    sample = X[:32]
    before = model.predict_proba(sample)
    with tempfile.TemporaryDirectory() as d:
        mp.save_sklearn(model, d)
        reloaded = mp.load_sklearn(d)
    mp.assert_bitwise(before, reloaded.predict_proba(sample), "sklearn LR")


def test_torch_within_tol():
    import torch
    from models.ffd.model import FFDModel
    torch.manual_seed(0)
    m = FFDModel(input_dim=12, device="cpu")
    sample = np.random.default_rng(1).normal(size=(32, 12)).astype(np.float32)
    before = m.predict_proba(sample)
    with tempfile.TemporaryDirectory() as d:
        mp.save_torch(m, {"input_dim": 12, "device": "cpu"}, d)
        reloaded = mp.load_torch(FFDModel, d)
    max_delta = mp.assert_within_tol(before, reloaded.predict_proba(sample), "torch FFD")
    print(f"  [torch FFD] max|Δ| after reload = {max_delta:.3e} (rtol={mp.RTOL}, atol={mp.ATOL})")


def _tiny_cfg(n_estimators, client_num):
    from omegaconf import OmegaConf
    return OmegaConf.create(
        {"n_estimators_client": n_estimators, "client_num": client_num,
         "dataset": {"task": {"task_type": "BINARY"}}}
    )


def test_fedxgbllr_two_stage_composed():
    """Reloaded (boosters + CNN) must reproduce the composed prediction."""
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from xgboost import XGBClassifier
    from hfedxgboost.models import CNN
    from hfedxgboost.utils import single_tree_preds_from_each_client

    n_est, client_num = 4, 3
    rng = np.random.default_rng(2)
    X = rng.normal(size=(120, 8)).astype(np.float32)
    y = (rng.random(120) < 0.4).astype(np.int32)

    # Stage 1: per-client boosters (as (estimator, client_id) tuples).
    trees = []
    for cid in range(client_num):
        sl = slice(cid * 30, (cid + 1) * 30)
        est = XGBClassifier(n_estimators=n_est, max_depth=3, tree_method="hist",
                            device="cpu", eval_metric="logloss", use_label_encoder=False)
        est.fit(X[sl], y[sl])
        trees.append((est, cid))

    cfg = _tiny_cfg(n_est, client_num)
    torch.manual_seed(0)
    cnn = CNN(cfg, n_channel=8)
    cnn.eval()

    # Fixed sample → real margins → CNN → composed prediction.
    loader = DataLoader(TensorDataset(torch.from_numpy(X[:32]),
                                      torch.from_numpy(y[:32].astype(np.float32))),
                        batch_size=32)

    def compose(tree_list, cnn_module):
        # single_tree_preds_from_each_client returns a DataLoader yielding the
        # margin tensor (B,1,client_num*n_est) already shaped for the CNN.
        tree_loader = single_tree_preds_from_each_client(
            loader, 32, [(e, i) for i, e in enumerate(tree_list)], n_est, client_num)
        margins, _y = next(iter(tree_loader))
        with torch.no_grad():
            return cnn_module(margins).cpu().numpy(), margins.numpy()

    before_out, before_margins = compose([e for e, _ in trees], cnn)

    with tempfile.TemporaryDirectory() as d:
        mp.save_fedxgbllr([e for e, _ in trees], cnn, {"n_channel": 8}, d)
        r_trees, r_cnn = mp.load_fedxgbllr(XGBClassifier, CNN, (cfg, 8), d)

    # Stage-1 bitwise: reloaded boosters reproduce probabilities exactly.
    for i, ((e, _), re) in enumerate(zip(trees, r_trees)):
        mp.assert_bitwise(e.predict_proba(X[:32]), re.predict_proba(X[:32]), f"booster {i}")
    # Composed two-stage: margins from RELOADED boosters + RELOADED CNN.
    after_out, after_margins = compose(r_trees, r_cnn)
    mp.assert_bitwise(before_margins, after_margins, "fedxgbllr margins (stage 1 composed)")
    max_delta = mp.assert_within_tol(before_out, after_out, "fedxgbllr composed ensemble+CNN")
    print(f"  [fedxgbllr] composed ensemble+CNN max|Δ| after reload = {max_delta:.3e}")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nAll {len(fns)} persistence tests passed.")
