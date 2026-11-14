"""Every random draw comes from the seed, and FedXGBllr's trees are fixed after round 0
(Ma et al. 2023, Algorithm 1).

* Aggregation is order-invariant: a strategy given the same client results in two
  orders returns bit-identical parameters. The client values are chosen so float
  addition order provably matters (1e8 + 1 - 1e8 in float32), and each test first
  shows that unsorted averaging differs — otherwise it could not fail.
* FedXGBllr round 0 (server): every client's trees, in client-id order, and a CNN
  initialised from the seed.
* FedXGBllr rounds 1..R (client): the CNN is trained on the fixed ensemble and the
  client's own trees come back unchanged; the same (seed, round, client) gives the
  same weights.

Runnable via ``pytest`` or ``python tests/<file>.py``.
"""

import os
import sys
from types import SimpleNamespace

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "models", "fedxgbllr"))

from flwr.common import (  # noqa: E402
    Code, FitIns, FitRes, GetParametersRes, Status, ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.strategy.aggregate import aggregate  # noqa: E402

from experiments.determinism import derive_seed, in_client_order  # noqa: E402

# One float32 element whose FedAvg sum depends on order: (1e8 + 1) - 1e8 == 0, but
# (1e8 - 1e8) + 1 == 1. Equal sample counts, so the weights cannot hide it.
_LAYERS = [np.array([1e8], np.float32), np.array([1.0], np.float32), np.array([-1e8], np.float32)]
_ORDERS = ([0, 1, 2], [2, 0, 1])


def _results(order, params_of=lambda k: ndarrays_to_parameters([_LAYERS[k]]), metrics=None):
    return [(SimpleNamespace(cid=str(k)),
             FitRes(status=Status(Code.OK, ""), parameters=params_of(k), num_examples=10,
                    metrics=metrics or {}))
            for k in order]


def _assert_order_matters_unsorted():
    a, b = ([(_LAYERS[k:k + 1], 10) for k in o] for o in _ORDERS)
    assert not np.array_equal(aggregate(a)[0], aggregate(b)[0]), "test data is not order-sensitive"


def test_derive_seed_is_stable_and_distinct():
    assert derive_seed(42, 1, 0) == derive_seed(42, 1, 0)
    assert len({derive_seed(42, r, c) for r in range(3) for c in range(5)}) == 15
    assert derive_seed(42, 1, 0) != derive_seed(123, 1, 0)


def test_in_client_order_sorts_and_rejects_duplicates():
    rs = _results([3, 0, 2, 1], params_of=lambda k: ndarrays_to_parameters([np.zeros(1, np.float32)]))
    assert [p.cid for p, _ in in_client_order(rs)] == ["0", "1", "2", "3"]
    try:
        in_client_order(rs + rs[:1])
    except RuntimeError as e:
        assert "duplicate" in str(e)
    else:
        raise AssertionError("a duplicated client id was accepted")


def test_fedavg_lr_and_svm_are_order_invariant():
    _assert_order_matters_unsorted()
    from models.fedavg_lr.strategy import _AbortOnTotalFailureFedAvg as LR
    from models.fedavg_svm.strategy import _AbortOnTotalFailureFedAvg as SVM
    for cls in (LR, SVM):
        out = [parameters_to_ndarrays(cls().aggregate_fit(1, _results(o), [])[0])[0] for o in _ORDERS]
        assert np.array_equal(out[0], out[1]), cls.__module__


def test_accuracy_weighted_fedavg_is_order_invariant():
    _assert_order_matters_unsorted()
    from models.bert_fraud.strategy import AccuracyWeightedFedAvg as BERT
    from models.ffd.strategy import AccuracyWeightedFedAvg as FFD
    for cls in (FFD, BERT):
        out = [parameters_to_ndarrays(
                   cls().aggregate_fit(1, _results(o, metrics={"local_auprc": 0.5}), [])[0])[0]
               for o in _ORDERS]
        assert np.array_equal(out[0], out[1]), cls.__module__


def test_fedxgbllr_strategy_is_order_invariant_and_keeps_tree_order():
    _assert_order_matters_unsorted()
    from hfedxgboost.strategy import FedXgbNnAvg
    trees = {k: object() for k in range(3)}
    params_of = lambda k: (GetParametersRes(status=Status(Code.OK, ""),  # noqa: E731
                                            parameters=ndarrays_to_parameters([_LAYERS[k]])),
                           (trees[k], k))
    outs = [FedXgbNnAvg().aggregate_fit(1, _results(o, params_of), [])[0] for o in _ORDERS]
    cnn = [parameters_to_ndarrays(cnn_params)[0] for cnn_params, _ in outs]
    assert np.array_equal(cnn[0], cnn[1])
    for _, tree_list in outs:
        assert [cid for _, cid in tree_list] == [0, 1, 2]
        assert all(t is trees[cid] for t, cid in tree_list)


# ---------------------------------------------------------------- FedXGBllr round 0
def _fx_cfg(seed=42, client_num=3, n_est=3):
    from omegaconf import OmegaConf
    return OmegaConf.create({
        "random_seed": seed, "n_estimators_client": n_est, "client_num": client_num,
        "clients": {"client_num": client_num, "CNN": {"lr": 0.001}},
        "dataset": {"early_stop_patience_rounds": 2, "task": {
            "task_type": "BINARY",
            "criterion": {"_target_": "torch.nn.BCELoss"},
            "metric": {"name": "Accuracy",
                       "fn": {"_target_": "torchmetrics.Accuracy", "task": "binary"}}}},
    })


class _Proxy:
    def __init__(self, cid):
        self.cid, self.calls = cid, 0

    def get_parameters(self, ins, timeout):
        self.calls += 1
        assert ins.config == {"round": 0}
        return (GetParametersRes(status=Status(Code.OK, ""), parameters=ndarrays_to_parameters([])),
                (f"trees-of-{self.cid}", int(self.cid)))


class _Manager:
    def __init__(self, cids):
        self.proxies = {c: _Proxy(c) for c in cids}

    def wait_for(self, num_clients, timeout=86400):
        return True

    def all(self):
        return dict(self.proxies)


def _round0(seed, cids=("2", "0", "1")):
    from hfedxgboost.server import FlServer
    mgr = _Manager(cids)
    server = FlServer(cfg=_fx_cfg(seed), client_manager=mgr, early_stopper=None, strategy=None)
    w0, trees = server._get_initial_parameters(timeout=None)
    return parameters_to_ndarrays(w0), trees, mgr


def test_round0_collects_every_clients_trees_once_in_client_order():
    w0, trees, mgr = _round0(42)
    assert trees == [("trees-of-0", 0), ("trees-of-1", 1), ("trees-of-2", 2)]
    assert all(p.calls == 1 for p in mgr.proxies.values())


def test_round0_cnn_comes_from_the_seed():
    a, _, _ = _round0(42)
    b, _, _ = _round0(42)
    c, _, _ = _round0(123)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
    assert not all(np.array_equal(x, y) for x, y in zip(a, c))


def test_round0_fails_loudly_on_a_missing_client():
    try:
        _round0(42, cids=("0", "2", "3"))
    except RuntimeError as e:
        assert "expected trees from clients 0..2" in str(e)
    else:
        raise AssertionError("round 0 accepted a non-contiguous set of clients")


# ------------------------------------------------------------ FedXGBllr rounds 1..R
def _fx_client_fit(cid, round_, seed=42):
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from xgboost import XGBClassifier
    from hfedxgboost.client import FlClient
    from hfedxgboost.models import CNN

    rng = np.random.default_rng(0)
    x = rng.normal(size=(60, 4)).astype(np.float32)
    y = (x[:, 0] + 0.3 * rng.normal(size=60) > 0.5).astype(np.float32)
    ensemble = []
    for k in range(3):                                   # the fixed round-0 ensemble
        sl = slice(20 * k, 20 * (k + 1))
        est = XGBClassifier(n_estimators=3, max_depth=2, tree_method="hist",
                            device="cpu", random_state=0).fit(x[sl], y[sl])
        ensemble.append((est, k))
    cfg = _fx_cfg(seed)
    torch.manual_seed(0)
    w = ndarrays_to_parameters(CNN(cfg).get_weights())
    own = slice(20 * cid, 20 * (cid + 1))
    loader = DataLoader(TensorDataset(torch.from_numpy(x[own]), torch.from_numpy(y[own])),
                        batch_size=20, shuffle=False)
    client = FlClient(cfg, loader, None, str(cid))
    res = client.fit(FitIns(parameters=(w, list(ensemble)),
                            config={"num_iterations": 3, "batch_size": 8, "round": round_}))
    return res, ensemble


def test_fedxgbllr_client_returns_its_own_trees_unchanged():
    res, ensemble = _fx_client_fit(cid=1, round_=1)
    assert res.parameters[1] is ensemble[1]              # the very object: never re-trained
    assert res.num_examples == 20


def test_fedxgbllr_client_update_comes_from_seed_round_and_client():
    a, _ = _fx_client_fit(cid=1, round_=1)
    b, _ = _fx_client_fit(cid=1, round_=1)
    c, _ = _fx_client_fit(cid=1, round_=2)
    wa, wb, wc = (parameters_to_ndarrays(r.parameters[0].parameters) for r in (a, b, c))
    assert all(np.array_equal(x, y) for x, y in zip(wa, wb))
    assert not all(np.array_equal(x, y) for x, y in zip(wa, wc))


def test_fedxgbllr_client_rejects_an_incomplete_ensemble():
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from hfedxgboost.client import FlClient
    from hfedxgboost.models import CNN
    cfg = _fx_cfg()
    w = ndarrays_to_parameters(CNN(cfg).get_weights())
    loader = DataLoader(TensorDataset(torch.zeros(4, 4), torch.zeros(4)), batch_size=4)
    try:
        FlClient(cfg, loader, None, "0").fit(FitIns(
            parameters=(w, [("t", 0)]), config={"num_iterations": 1, "batch_size": 2, "round": 1}))
    except RuntimeError as e:
        assert "round-0 ensemble" in str(e)
    else:
        raise AssertionError("fit accepted an ensemble missing clients")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nAll {len(fns)} determinism tests passed.")
