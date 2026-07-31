"""Stage-0 SHAP feasibility probe — RUN ON THE GPU BOX (artifacts live there).

READ-ONLY. Loads frozen artifacts under ``results/models/`` + the data cache,
runs each candidate explainer once, and reports six measurements. Writes NOTHING
(no file under results/ is touched); everything goes to stdout.

    python experiments/shap_stage0_probe.py

Reports, in one pass:
  1. FedXGBllr reload — boosters + CNN reproduce stored predictions.
  2. TreeExplainer on a reloaded truncated GBM — accepts the sliced _predictors.
  3. LinearExplainer on a reloaded SVM — coef_/intercept_ suffice.
  4. DeepExplainer local-accuracy reconstruction error on one FFD and one BERT
     (fallback GradientExplainer -> KernelSHAP), tolerance stated.
  5. Per-model timing estimate for 500 explanation samples.
  6. feature_names.json order == cache feature_names, per artifact.

Nothing here refits, retrains, or edits any artifact.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "models" / "fedxgbllr"))  # hfedxgboost.*

# Explained-scale + probe knobs.
LOGIT_EPS = 1e-6
N_BACKGROUND = 100
N_RECON = 20          # samples for local-accuracy reconstruction
N_TIME = 20           # samples timed, then extrapolated to 500
N_EXPLAIN_TARGET = 500
DEEP_TOL = 1e-2       # abs tolerance on the log-odds scale for local accuracy
KNOWN = ["bert_fraud", "fedxgbllr", "gbm", "ffd", "svm", "lr", "xgb"]


def _logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), LOGIT_EPS, 1 - LOGIT_EPS)
    return np.log(p / (1 - p))


def detect_model(run_name: str):
    s = run_name[len("centralized_"):] if run_name.startswith("centralized_") else run_name
    for tok in KNOWN:
        if s == tok or s.startswith(tok + "_"):
            return tok
    return None


def _scheme_arm(run_name: str):
    scheme = ("centralized" if run_name.startswith("centralized_")
              else "dirichlet" if "dirichlet" in run_name
              else "iid" if "iid" in run_name else "?")
    arm = "smote" if "_smote_" in run_name else "none"
    return scheme, arm


def discover():
    root = PROJECT_ROOT / "results" / "models"
    arts = []
    for mani in root.glob("*/*/manifest.json"):
        d = mani.parent
        run_name = d.name
        model = detect_model(run_name)
        if not model:
            continue
        scheme, arm = _scheme_arm(run_name)
        arts.append({"model": model, "dataset": d.parent.name, "run_name": run_name,
                     "scheme": scheme, "arm": arm, "dir": d})
    return arts


def pick(arts, model, prefer=None):
    cand = [a for a in arts if a["model"] == model]
    if not cand:
        return None
    if prefer:
        cand.sort(key=prefer)
    return cand[0]


def manifest_hash(art_dir: Path) -> str:
    return hashlib.sha256((art_dir / "manifest.json").read_bytes()).hexdigest()[:16]


def load_cache(dataset):
    from experiments import data_cache
    d, dh = data_cache.get_preprocessed(dataset, 42)
    return d, dh


def _cache_feature_names(dataset):
    d, _ = load_cache(dataset)
    return list(d.get("feature_names", []))


# --------------------------------------------------------------------------- #
def probe_feature_names(arts):
    print("\n[6] feature_names.json order == cache feature_names")
    ok_all = True
    for a in arts:
        fj = a["dir"] / "feature_names.json"
        if not fj.is_file():
            print(f"    {a['dataset']}/{a['run_name']}: NO feature_names.json"); ok_all = False; continue
        art_fn = json.loads(fj.read_text())
        cache_fn = _cache_feature_names(a["dataset"])
        ok = list(art_fn) == list(cache_fn)
        ok_all &= ok
        print(f"    {a['dataset']:10} {a['model']:10}: match={ok} (n={len(art_fn)})"
              + ("" if ok else f"  !! first-diff at {_first_diff(art_fn, cache_fn)}"))
    print(f"  => feature-name order consistent across all artifacts: {ok_all}")


def _first_diff(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return f"idx {i}: {x!r} vs {y!r}"
    return f"len {len(a)} vs {len(b)}"


# --------------------------------------------------------------------------- #
def load_fedxgbllr_composed(art):
    """Return (predict_proba fn, n_features) for the reloaded two-stage model."""
    import torch
    from omegaconf import OmegaConf
    from xgboost import XGBClassifier
    from torch.utils.data import DataLoader, TensorDataset
    from hfedxgboost.models import CNN
    from hfedxgboost.utils import single_tree_preds_from_each_client
    from evaluation import model_persistence as mp

    arch = json.loads((art["dir"] / "cnn_arch_config.json").read_text())
    n_est, cnum = int(arch["n_estimators_client"]), int(arch["client_num"])
    cfg = OmegaConf.create({"dataset": {"task": {"task_type": "BINARY"}},
                            "n_estimators_client": n_est, "client_num": cnum,
                            "run_experiment": {"batch_size": 512}})
    trees, cnn = mp.load_fedxgbllr(XGBClassifier, CNN, (cfg,), art["dir"])
    tlist = [(t, i) for i, t in enumerate(trees)]

    def predict_proba(X):
        X = np.asarray(X, dtype=np.float32)
        ds = TensorDataset(torch.from_numpy(X), torch.zeros(len(X)))
        tl = single_tree_preds_from_each_client(
            DataLoader(ds, batch_size=len(ds), shuffle=False), 512, tlist, n_est, cnum)
        out = []
        with torch.no_grad():
            for xb, _ in tl:
                out.append(cnn(xb).numpy().reshape(-1))
        return np.concatenate(out)

    return predict_proba, None


def probe_fedxgbllr(art):
    print("\n[1] FedXGBllr reload reproduces stored predictions "
          f"({art['dataset']}/{art['run_name']})")
    from evaluation.metrics import auprc
    predict_proba, _ = load_fedxgbllr_composed(art)
    data, dh = load_cache(art["dataset"])
    xte, yte = np.asarray(data["x_test"], np.float32), np.asarray(data["y_test"]).astype(int)
    p = predict_proba(xte)
    a = auprc(yte, p)
    stored = _stored_metric(art["dataset"], "fedxgbllr", art["scheme"], art["arm"], "test_auprc")
    print(f"    recomputed test AUPRC = {a:.4f} | stored clean_summary = {stored}")
    print(f"    data_hash match: {dh[:12]} vs manifest {json.loads((art['dir']/'manifest.json').read_text()).get('data_hash','')[:12]}")
    ref = art["dir"] / "reference_pred.npy"
    if ref.is_file():
        rp = np.load(ref)
        pr = predict_proba(xte[:len(rp)])
        print(f"    reference_pred bitwise: max|Δ|={np.abs(pr - np.asarray(rp).reshape(-1)).max():.2e}")
    ok = stored is not None and abs(a - float(stored)) < 2e-3
    print(f"    => reload reproduces stored test AUPRC (tol 2e-3): {ok}")
    return predict_proba, xte, yte


def _stored_metric(dataset, model, scheme, arm, col):
    import csv
    p = PROJECT_ROOT / "results" / "clean_summary.csv"
    ov = "smote" if arm == "smote" else "none"
    for r in csv.DictReader(open(p)):
        if (r["dataset"] == dataset and r["model"] == model
                and r["scheme"] == scheme and r["oversampling"] == ov):
            return r.get(col)
    return None


# --------------------------------------------------------------------------- #
def probe_gbm(art):
    print(f"\n[2] TreeExplainer on truncated GBM ({art['dataset']}/{art['run_name']})")
    import shap
    from evaluation import model_persistence as mp
    model = mp.load_sklearn(art["dir"])
    n_iter = int(getattr(model, "n_iter_", -1))
    n_pred = len(getattr(model, "_predictors", []))
    print(f"    reloaded n_iter_={n_iter}  len(_predictors)={n_pred}  (k* selected model)")
    data, _ = load_cache(art["dataset"])
    xte = np.asarray(data["x_test"], np.float32)
    ex = shap.TreeExplainer(model)  # tree_path_dependent default for HistGBM
    Xs = xte[:N_RECON]
    sv = ex.shap_values(Xs)
    sv = sv[1] if isinstance(sv, list) else sv
    base = ex.expected_value
    base = base[1] if hasattr(base, "__len__") else base
    # local accuracy on TreeExplainer raw output (margin/log-odds)
    raw = model.decision_function(Xs) if hasattr(model, "decision_function") else _logit(model.predict_proba(Xs)[:, 1])
    recon = sv.sum(axis=1) + base
    err = float(np.abs(raw - recon).max())
    print(f"    TreeExplainer OK on sliced _predictors: True | base={float(base):.4f}")
    print(f"    local-accuracy max|Σφ+φ0 − f(x)| = {err:.2e}")
    return model


def probe_svm(art):
    print(f"\n[3] LinearExplainer on SVM ({art['dataset']}/{art['run_name']})")
    import shap
    from evaluation import model_persistence as mp
    model = mp.load_sklearn(art["dir"])
    print(f"    coef_ shape={getattr(model,'coef_',None).shape if hasattr(model,'coef_') else None}"
          f"  intercept_={getattr(model,'intercept_',None)}")
    data, _ = load_cache(art["dataset"])
    xtr = np.asarray(data["x_train"], np.float32)
    xte = np.asarray(data["x_test"], np.float32)
    bg = xtr[np.random.default_rng(0).choice(len(xtr), min(N_BACKGROUND, len(xtr)), replace=False)]
    try:
        ex = shap.LinearExplainer(model, bg)
    except Exception as exc:  # noqa: BLE001 — fall back to explicit (coef, intercept)
        print(f"    LinearExplainer(model) rejected ({type(exc).__name__}); using (coef_, intercept_) form")
        ex = shap.LinearExplainer((model.coef_[0], float(model.intercept_[0])), bg)
    Xs = xte[:N_RECON]
    sv = ex.shap_values(Xs)
    sv = sv[1] if isinstance(sv, list) else sv
    base = ex.expected_value
    base = base[1] if hasattr(base, "__len__") else base
    margin = model.decision_function(Xs)  # explained quantity = margin
    err = float(np.abs(margin - (sv.sum(axis=1) + base)).max())
    print(f"    LinearExplainer OK (coef_/intercept_ suffice): True")
    print(f"    local-accuracy on margin: max|Σφ+φ0 − margin| = {err:.2e}")
    return model


# --------------------------------------------------------------------------- #
def probe_torch(art, model_cls_name):
    print(f"\n[4] DeepExplainer local accuracy ({model_cls_name}: {art['dataset']}/{art['run_name']})")
    import torch
    import torch.nn as nn
    import shap
    from evaluation import model_persistence as mp
    if model_cls_name == "FFDModel":
        from models.ffd.model import FFDModel as Cls
    else:
        from models.bert_fraud.model import BertFraudModel as Cls
    model = mp.load_torch(Cls, art["dir"], device="cpu")
    model.eval()

    class LogOdds(nn.Module):
        def __init__(self, m): super().__init__(); self.m = m
        def forward(self, x):
            z = self.m(x)                      # (B,2) logits
            return (z[:, 1] - z[:, 0]).reshape(-1, 1)   # log-odds(fraud)

    wrap = LogOdds(model)
    data, _ = load_cache(art["dataset"])
    xtr = np.asarray(data["x_train"], np.float32)
    xte = np.asarray(data["x_test"], np.float32)
    rng = np.random.default_rng(0)
    bg = torch.from_numpy(xtr[rng.choice(len(xtr), min(N_BACKGROUND, len(xtr)), replace=False)])
    Xs = torch.from_numpy(xte[:N_RECON])
    fx = wrap(Xs).detach().numpy().reshape(-1)

    used, err = None, None
    for name, ctor in [("DeepExplainer", lambda: shap.DeepExplainer(wrap, bg)),
                       ("GradientExplainer", lambda: shap.GradientExplainer(wrap, bg))]:
        try:
            ex = ctor()
            sv = ex.shap_values(Xs)
            sv = sv[0] if isinstance(sv, list) else sv
            sv = np.asarray(sv).reshape(N_RECON, -1)
            ev = getattr(ex, "expected_value", 0.0)
            ev = float(np.asarray(ev).reshape(-1)[0]) if np.asarray(ev).size else 0.0
            recon = sv.sum(axis=1) + ev
            err = float(np.abs(fx - recon).max())
            used = name
            break
        except Exception as exc:  # noqa: BLE001
            print(f"    {name} failed: {type(exc).__name__}: {str(exc)[:120]}")
    if used is None:
        print("    both Deep/Gradient failed -> KernelSHAP fallback required (not timed here)")
        return
    verdict = "PASS" if (used == "DeepExplainer" and err <= DEEP_TOL) else \
              ("APPROX (Gradient — local accuracy not guaranteed)" if used == "GradientExplainer" else "FAIL")
    print(f"    explainer used: {used} | tolerance (abs, log-odds) = {DEEP_TOL}")
    print(f"    local-accuracy max|Σφ+φ0 − f(x)| = {err:.2e}  => {verdict}")


# --------------------------------------------------------------------------- #
def probe_timing(gbm_model, svm_model, fx_predict, ffd_bert_arts):
    print(f"\n[5] Timing estimate for {N_EXPLAIN_TARGET} explanation samples")
    import shap, torch
    import torch.nn as nn
    from evaluation import model_persistence as mp

    def scale(t_small):
        return t_small / N_TIME * N_EXPLAIN_TARGET

    # GBM TreeExplainer
    if gbm_model is not None:
        d, _ = load_cache("creditcard")
        X = np.asarray(d["x_test"], np.float32)[:N_TIME]
        ex = shap.TreeExplainer(gbm_model)
        t = time.time(); ex.shap_values(X); dt = time.time() - t
        print(f"    GBM   (TreeExplainer)   : ~{scale(dt):6.1f}s / {N_EXPLAIN_TARGET}")
    # SVM LinearExplainer
    if svm_model is not None:
        d, _ = load_cache("creditcard")
        X = np.asarray(d["x_test"], np.float32)
        ex = shap.LinearExplainer(svm_model, X[:N_BACKGROUND])
        t = time.time(); ex.shap_values(X[:N_TIME]); dt = time.time() - t
        print(f"    SVM   (LinearExplainer) : ~{scale(dt):6.1f}s / {N_EXPLAIN_TARGET}")
    # FedXGBllr KernelSHAP on composed feature->log-odds (the chosen black box)
    if fx_predict is not None:
        fx_ds, fx_xte, _ = fx_predict
        f = lambda X: _logit(fx_ds(np.asarray(X, np.float32)))  # noqa: E731
        bg = shap.kmeans(fx_xte[:200], 10)
        ex = shap.KernelExplainer(f, bg)
        n_small = 2
        t = time.time(); ex.shap_values(fx_xte[:n_small], nsamples=100, silent=True); dt = time.time() - t
        print(f"    FedXGBllr (KernelSHAP,nsamples=100): ~{dt/n_small*N_EXPLAIN_TARGET:6.1f}s / {N_EXPLAIN_TARGET}"
              f"  (per-sample {dt/n_small:.2f}s; scales with nsamples)")


# --------------------------------------------------------------------------- #
def main():
    print("=" * 72)
    print("SHAP Stage-0 feasibility probe — READ-ONLY, writes nothing")
    print("=" * 72)
    try:
        import shap
        print(f"shap {shap.__version__}")
    except Exception as exc:  # noqa: BLE001
        print(f"FATAL: shap import failed: {exc}"); return 1

    arts = discover()
    if not arts:
        print("\nNO artifacts under results/models/ — run this ON THE BOX where the "
              "frozen models live. Nothing to probe.")
        return 2
    print(f"\nDiscovered {len(arts)} artifacts. Provenance example (manifest hash to "
          "record in SHAP outputs, joined on dataset/model/condition/arm):")
    for a in arts[:3]:
        print(f"    {a['dataset']}/{a['model']}/{a['scheme']}_{a['arm']}: "
              f"manifest_sha256[:16]={manifest_hash(a['dir'])}")

    fx = pick(arts, "fedxgbllr", prefer=lambda a: (a["dataset"] != "baf", a["scheme"] != "dirichlet"))
    gbm = pick(arts, "gbm", prefer=lambda a: (int(_nk(a)),))  # smallest k* first (tests slicing)
    svm = pick(arts, "svm")
    ffd = pick(arts, "ffd", prefer=lambda a: (a["dataset"] != "creditcard",))
    bert = pick(arts, "bert_fraud", prefer=lambda a: (a["dataset"] != "creditcard",))

    fx_bundle = None
    gbm_model = svm_model = None
    for name, fn in [
        ("fedxgbllr", lambda: probe_fedxgbllr(fx) if fx else print("\n[1] no fedxgbllr artifact")),
        ("gbm", lambda: probe_gbm(gbm) if gbm else print("\n[2] no gbm artifact")),
        ("svm", lambda: probe_svm(svm) if svm else print("\n[3] no svm artifact")),
        ("ffd", lambda: probe_torch(ffd, "FFDModel") if ffd else print("\n[4] no ffd artifact")),
        ("bert", lambda: probe_torch(bert, "BertFraudModel") if bert else print("\n[4] no bert artifact")),
    ]:
        try:
            r = fn()
            if name == "fedxgbllr" and r:
                fx_bundle = r
            elif name == "gbm":
                gbm_model = r
            elif name == "svm":
                svm_model = r
        except Exception:  # noqa: BLE001
            print(f"    [{name}] PROBE ERROR:\n" + textwrap_indent(traceback.format_exc()))

    try:
        probe_timing(gbm_model, svm_model, fx_bundle, [ffd, bert])
    except Exception:  # noqa: BLE001
        print("    [timing] ERROR:\n" + textwrap_indent(traceback.format_exc()))

    try:
        probe_feature_names(arts)
    except Exception:  # noqa: BLE001
        print("    [feature_names] ERROR:\n" + textwrap_indent(traceback.format_exc()))

    print("\n" + "=" * 72)
    print("Probe complete. No artifact or results file was modified.")
    print("=" * 72)
    return 0


def _nk(art):
    """Best-effort k* for a gbm artifact (smallest first). Reads n_boosters or loads."""
    try:
        from evaluation import model_persistence as mp
        m = mp.load_sklearn(art["dir"])
        return int(getattr(m, "n_iter_", 999))
    except Exception:  # noqa: BLE001
        return 999


def textwrap_indent(s):
    return "\n".join("      " + ln for ln in s.splitlines())


if __name__ == "__main__":
    raise SystemExit(main())
