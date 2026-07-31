"""Stage-1 production SHAP — RUN ON THE BOX (frozen artifacts live there).

READ-ONLY over results/models/ + the data cache. Writes ONLY under results/shap/.
Never touches results/logs/, results/clean_summary.csv, or results/sweep/.

    python experiments/shap_analysis.py                 # full approved scope
    python experiments/shap_analysis.py --datasets baf  # subset
    python experiments/shap_analysis.py --max-hours 8   # wall-clock guard

Per cell (dataset, model, condition, arm) with a persisted artifact:
  * Explainer mapping (measured, Stage 0): LR/SVM -> LinearSHAP (SVM on the margin);
    GBM/XGB -> TreeSHAP (tree_path_dependent); FFD/BERT/FedXGBllr -> KernelSHAP
    (nsamples=500, the noise-floor value).
  * Each client explains the SHARED 500-sample central-test subset using its OWN
    background (100 local post-SMOTE samples -> 10 k-means centroids — the exact
    treatment the noise floor was measured under, so the floor transfers).
  * Aggregate: mean importance; cross-client Spearman, Jaccard@5, and Kuncheva's
    chance-corrected index (for cross-dataset claims). All on log-odds (SVM margin).
  * Provenance: artifact manifest sha256[:16], data_hash, partition_hash — joined
    on (dataset, model, condition, arm); nothing added to the frozen summary.

Scope order (approved): cheap models (LR/SVM/GBM/XGB) on all cells/datasets first,
then expensive models (FFD/BERT/FedXGBllr) on BAF, then ULB, then PaySim.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "models" / "fedxgbllr"))

from evaluation import shap_stability as ST  # noqa: E402

SEED = 42
NSAMPLES = 500
N_BG = 100
KMEANS_K = 10
N_EXPLAIN = 500
NUM_CLIENTS = 5
LOGIT_EPS = 1e-6
OUT = PROJECT_ROOT / "results" / "shap"
EXPLAINER = {"lr": "linear", "svm": "linear", "gbm": "tree", "xgb": "tree",
             "ffd": "kernel", "bert": "kernel", "fedxgbllr": "kernel"}
CHEAP = {"lr", "svm", "gbm", "xgb"}
EXPENSIVE_DATASET_ORDER = ["baf", "creditcard", "paysim"]
KNOWN = ["bert_fraud", "fedxgbllr", "gbm", "ffd", "svm", "lr", "xgb"]


def _logit(p):
    p = np.clip(np.asarray(p, np.float64), LOGIT_EPS, 1 - LOGIT_EPS)
    return np.log(p / (1 - p))


def _sv2d(sv, n):
    if isinstance(sv, list):
        sv = sv[-1]
    sv = np.asarray(sv)
    if sv.ndim == 3:
        sv = sv[..., -1]
    return sv.reshape(n, -1)


def detect_model(run_name):
    s = run_name[len("centralized_"):] if run_name.startswith("centralized_") else run_name
    for tok in KNOWN:
        if s == tok or s.startswith(tok + "_"):
            return tok
    return None


def discover():
    root = PROJECT_ROOT / "results" / "models"
    arts = []
    for mani in root.glob("*/*/manifest.json"):
        d = mani.parent
        model = detect_model(d.name)
        if not model:
            continue
        rn = d.name
        cond = ("centralized" if rn.startswith("centralized_")
                else "dirichlet" if "dirichlet" in rn else "iid" if "iid" in rn else "?")
        arm = "smote" if "_smote_" in rn else "none"
        alpha = None
        if "dirichlet" in rn:
            for tok in rn.split("_"):
                if tok.startswith("alpha") and tok[5:] not in ("", "-"):
                    try:
                        alpha = float(tok[5:])
                    except ValueError:
                        pass
        arts.append({"model": model, "dataset": d.parent.name, "condition": cond,
                     "arm": arm, "alpha": alpha, "run_name": rn, "dir": d})
    return arts


def scope_order(arts, datasets, models):
    arts = [a for a in arts if a["dataset"] in datasets and a["model"] in models]
    cheap = [a for a in arts if a["model"] in CHEAP]
    exp = [a for a in arts if a["model"] not in CHEAP]
    exp.sort(key=lambda a: (EXPENSIVE_DATASET_ORDER.index(a["dataset"])
                            if a["dataset"] in EXPENSIVE_DATASET_ORDER else 9))
    return cheap + exp


# --------------------------------------------------------------------------- #
def manifest_hash(d):
    return hashlib.sha256((d / "manifest.json").read_bytes()).hexdigest()[:16]


def feature_names(art):
    fj = art["dir"] / "feature_names.json"
    if fj.is_file():
        return list(json.loads(fj.read_text()))
    from experiments import data_cache
    return list(data_cache.get_preprocessed(art["dataset"], SEED)[0].get("feature_names", []))


def explanation_set(dataset, rng):
    from experiments import data_cache
    d, dh = data_cache.get_preprocessed(dataset, SEED)
    xte = np.asarray(d["x_test"], np.float32); yte = np.asarray(d["y_test"]).astype(int)
    n_pos = max(1, int(round(N_EXPLAIN * yte.mean())))
    pi = rng.choice(np.where(yte == 1)[0], min(n_pos, int((yte == 1).sum())), replace=False)
    ni = rng.choice(np.where(yte == 0)[0], N_EXPLAIN - len(pi), replace=False)
    idx = np.concatenate([pi, ni]); rng.shuffle(idx)
    return xte[idx], dh


def client_backgrounds(art, rng):
    """List of per-client background arrays (100 local post-SMOTE samples each).
    Centralized -> single background from central train."""
    from experiments import data_cache
    from imblearn.over_sampling import SMOTE

    def sample_bg(x):
        return x[rng.choice(len(x), min(N_BG, len(x)), replace=False)]

    def oversample(x, y):
        if art["arm"] != "smote":
            return x, y
        try:
            return SMOTE(sampling_strategy=0.01, k_neighbors=5, random_state=SEED).fit_resample(x, y)
        except ValueError:
            return x, y  # no-op

    if art["condition"] == "centralized":
        d, _ = data_cache.get_preprocessed(art["dataset"], SEED)
        x, y = oversample(np.asarray(d["x_train"], np.float32), np.asarray(d["y_train"]).astype(int))
        return [sample_bg(x)], "n/a (centralized)"

    clients, phash = data_cache.get_partition_clients(
        art["dataset"], SEED, art["condition"], art["alpha"], NUM_CLIENTS)
    bgs = []
    for c in clients:
        x, y = oversample(np.asarray(c["x"], np.float32), np.asarray(c["y"]).astype(int))
        bgs.append(sample_bg(x))
    return bgs, phash


# --------------------------------------------------------------------------- #
def load_predictor(art):
    """Return (obj, kind) where obj is a sklearn model (linear/tree) or a numpy
    log-odds function (kernel)."""
    kind = EXPLAINER[art["model"]]
    from evaluation import model_persistence as mp
    if kind in ("linear", "tree"):
        return mp.load_sklearn(art["dir"]), kind
    # kernel: build a numpy log-odds function
    if art["model"] == "fedxgbllr":
        return _fedxgbllr_fn(art), kind
    return _torch_logodds_fn(art), kind


def _fedxgbllr_fn(art):
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
    cnn = cnn.cpu().eval(); tl = [(t, i) for i, t in enumerate(trees)]

    def f(X):
        X = np.asarray(X, np.float32)
        ds = TensorDataset(torch.from_numpy(X), torch.zeros(len(X)))
        loader = single_tree_preds_from_each_client(
            DataLoader(ds, batch_size=len(ds), shuffle=False), 512, tl, n_est, cnum)
        out = []
        with torch.no_grad():
            for xb, _ in loader:
                out.append(cnn(xb).numpy().reshape(-1))
        return _logit(np.concatenate(out))
    return f


def _torch_logodds_fn(art):
    import torch
    import torch.nn as nn
    from evaluation import model_persistence as mp
    if art["model"] == "bert":
        from models.bert_fraud.model import BertFraudModel as Cls
    else:
        from models.ffd.model import FFDModel as Cls
    model = mp.load_torch(Cls, art["dir"], device="cpu").cpu().eval()

    class LogOdds(nn.Module):
        def __init__(self, m): super().__init__(); self.m = m
        def forward(self, x):
            z = self.m(x); return (z[:, 1] - z[:, 0]).reshape(-1, 1)
    wrap = LogOdds(model).cpu().eval()

    def f(X):
        with torch.no_grad():
            return wrap(torch.from_numpy(np.asarray(X, np.float32))).numpy().ravel()
    return f


def client_importance(obj, kind, bg, X):
    import shap
    if kind == "linear":
        ex = shap.LinearExplainer(obj, bg)
        sv = _sv2d(ex.shap_values(X), len(X))
    elif kind == "tree":
        # tree_path_dependent is background-free: for a shared global model the
        # per-client importance is identical, so tree-model cross-client stability
        # is trivially 1.0 by construction (recorded as a caveat in §4.5).
        ex = shap.TreeExplainer(obj, feature_perturbation="tree_path_dependent")
        sv = _sv2d(ex.shap_values(X), len(X))
    else:  # kernel
        np.random.seed(SEED)
        ex = shap.KernelExplainer(obj, shap.kmeans(bg, KMEANS_K))
        sv = _sv2d(ex.shap_values(X, nsamples=NSAMPLES, silent=True), len(X))
    return np.abs(sv).mean(axis=0)


# --------------------------------------------------------------------------- #
def process_cell(art, rng, summary_rows):
    kind = EXPLAINER[art["model"]]
    fnames = feature_names(art)
    d = len(fnames)
    X, dh = explanation_set(art["dataset"], np.random.default_rng(SEED))
    bgs, phash = client_backgrounds(art, np.random.default_rng(SEED + 1))
    obj, kind = load_predictor(art)

    imps = []
    for bi, bg in enumerate(bgs):
        t = time.time()
        imps.append(client_importance(obj, kind, bg, X))
        if kind == "kernel":
            print(f"      client {bi}: {time.time()-t:.0f}s")
        if kind == "tree":  # identical across clients — compute once, replicate
            imps = [imps[0]] * len(bgs); break

    imps = [np.asarray(v, float) for v in imps]
    mean_imp = ST.mean_importance(imps)
    n_cli = len(imps)
    if n_cli >= 2:
        spear = ST.spearman_stability(imps)
        jac = ST.jaccard_at_k(imps, k=5)
        kun = ST.kuncheva_index(imps, d=d, k=5)
    else:
        spear = jac = kun = float("nan")

    cell_dir = OUT / art["dataset"] / art["model"] / f"{art['condition']}_{art['arm']}"
    cell_dir.mkdir(parents=True, exist_ok=True)
    prov = {"dataset": art["dataset"], "model": art["model"], "condition": art["condition"],
            "arm": art["arm"], "explainer": kind, "nsamples": NSAMPLES if kind == "kernel" else "n/a",
            "n_clients": n_cli, "manifest_sha256": manifest_hash(art["dir"]),
            "data_hash": dh[:16], "partition_hash": (phash[:16] if isinstance(phash, str) else phash),
            "feature_names_source": "artifact" if (art["dir"] / "feature_names.json").is_file() else "cache"}
    # per-client importance matrix (features x clients)
    with open(cell_dir / "importance_per_client.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["feature"] + [f"client_{i}" for i in range(n_cli)] + ["mean"])
        for j, name in enumerate(fnames):
            w.writerow([name] + [f"{imps[i][j]:.6g}" for i in range(n_cli)] + [f"{mean_imp[j]:.6g}"])
    order = np.argsort(mean_imp)[::-1]
    (cell_dir / "aggregated.json").write_text(json.dumps(
        {**prov, "top10_features": [[fnames[i], float(mean_imp[i])] for i in order[:10]]}, indent=2))
    (cell_dir / "stability.json").write_text(json.dumps(
        {**prov, "spearman": spear, "jaccard_at5": jac, "kuncheva": kun}, indent=2))

    summary_rows.append({**{k: prov[k] for k in ("dataset", "model", "condition", "arm",
                          "explainer", "n_clients", "manifest_sha256", "data_hash", "partition_hash")},
                         "spearman": round(spear, 4) if spear == spear else "n/a",
                         "jaccard_at5": round(jac, 4) if jac == jac else "n/a",
                         "kuncheva": round(kun, 4) if kun == kun else "n/a",
                         "top_feature": fnames[order[0]]})
    print(f"    [{art['dataset']}/{art['model']}/{art['condition']}_{art['arm']}] "
          f"clients={n_cli} spearman={spear:.4f} kuncheva={kun:.4f} jac5={jac:.2f} top={fnames[order[0]]}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", default="baf,creditcard,paysim")
    ap.add_argument("--models", default=",".join(KNOWN))
    ap.add_argument("--max-hours", type=float, default=None)
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args(argv)
    datasets = set(args.datasets.split(","))
    models = set(args.models.split(","))

    try:
        import shap
        print(f"shap {shap.__version__} | nsamples={NSAMPLES} bg={N_BG}->kmeans{KMEANS_K} "
              f"explain={N_EXPLAIN}")
    except Exception as e:  # noqa: BLE001
        print(f"FATAL shap import: {e}"); return 1

    arts = discover()
    if not arts:
        print("NO artifacts under results/models/ — run on the box. Nothing to do.")
        return 2
    cells = scope_order(arts, datasets, models)
    print(f"{len(cells)} cells in scope (cheap first, then expensive BAF->ULB->PaySim).")
    OUT.mkdir(parents=True, exist_ok=True)

    summary_rows, t0, done = [], time.time(), 0
    for art in cells:
        cd = OUT / art["dataset"] / art["model"] / f"{art['condition']}_{art['arm']}"
        if args.skip_existing and (cd / "stability.json").is_file():
            print(f"    skip (exists) {cd}"); continue
        if args.max_hours and (time.time() - t0) / 3600 > args.max_hours:
            print(f"\n[wall guard] {args.max_hours}h reached — stopping after {done} cells.")
            break
        try:
            process_cell(art, None, summary_rows)
            done += 1
        except Exception:  # noqa: BLE001
            import traceback
            print(f"    [ERROR] {art['dataset']}/{art['model']}/{art['condition']}_{art['arm']}:")
            print("\n".join("        " + ln for ln in traceback.format_exc().splitlines()))

    if summary_rows:
        cols = list(summary_rows[0].keys())
        p = OUT / "shap_summary.csv"
        # append-safe: merge with any existing rows (idempotent by cell key)
        existing = {}
        if p.is_file():
            for r in csv.DictReader(open(p)):
                existing[(r["dataset"], r["model"], r["condition"], r["arm"])] = r
        for r in summary_rows:
            existing[(r["dataset"], r["model"], r["condition"], r["arm"])] = r
        with open(p, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            for r in existing.values():
                w.writerow(r)
        print(f"\nWrote {len(summary_rows)} cells -> results/shap/ (+ shap_summary.csv, "
              f"{len(existing)} total rows).")
    print(f"Elapsed {(time.time()-t0)/60:.1f} min. Nothing outside results/shap/ was written.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
