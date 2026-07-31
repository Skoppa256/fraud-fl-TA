"""KernelSHAP noise-floor measurement — RUN ON THE BOX (artifacts live there).

READ-ONLY. Measures how well KernelSHAP agrees with ITSELF across random seeds,
so cross-client stability below that floor can be recognised as sampling noise
rather than model behaviour. Writes nothing under results/; prints a report.

    python experiments/shap_noise_floor.py

For the two models where KernelSHAP has no exact reference (FedXGBllr and BERT),
on one BAF-dirichlet client each: run KernelSHAP TWICE with different random seeds
at nsamples ∈ {100, 500, 1000}; report Spearman and Jaccard@5 between the two
feature-importance vectors. Decision rule: adopt the smallest nsamples where the
two runs agree at Spearman > 0.95.

Also runs the FFD cross-check: DeepSHAP (which passed local accuracy 1.37e-06) vs
KernelSHAP on the same client — Spearman + mean|Δ| — as evidence KernelSHAP is
converging where no exact reference exists.

Background = 100 local post-SMOTE samples from one client (summarised to 10 kmeans
centroids for the explainer, matching the probe's cost model). Explanation data =
a fixed class-proportional central-test subset, identical across the two seeds.
Log-odds scale throughout.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "models" / "fedxgbllr"))

# ---- knobs (adjust here) --------------------------------------------------- #
DATASET = "baf"
CONDITION = "dirichlet"      # BAF dirichlet: the cells §4.3 rests on
ALPHA = 0.5
ARM = "none"                 # background client uses the no-SMOTE arm's model
NS_LIST = [100, 500, 1000]
N_EXPLAIN = 250              # explanation samples (raise to 500 for the exact production floor)
N_BG = 100                   # local background pool
KMEANS_K = 10                # background summary size (cost model)
BG_CLIENT = 0                # which client's local data seeds the background
SEEDS = (11, 22)             # two KernelSHAP random seeds
LOGIT_EPS = 1e-6


def _logit(p):
    p = np.clip(np.asarray(p, np.float64), LOGIT_EPS, 1 - LOGIT_EPS)
    return np.log(p / (1 - p))


def _sv_2d(sv, n):
    if isinstance(sv, list):
        sv = sv[-1]
    sv = np.asarray(sv)
    if sv.ndim == 3:
        sv = sv[..., -1]
    return sv.reshape(n, -1)


def _child_run_name(model, condition, alpha, arm):
    scheme = "iid" if condition == "iid" else "dirichlet"
    a = "-" if condition == "iid" else f"{alpha:g}"
    ov = "smote" if arm == "smote" else "none"
    return f"{model}_{scheme}_alpha{a}_{ov}_seed42"


def _art_dir(model, condition, alpha, arm):
    sub = "fedxgbllr" if model == "fedxgbllr" else model
    return PROJECT_ROOT / "results" / "models" / DATASET / _child_run_name(model, condition, alpha, arm)


# ---- model loaders (composed log-odds functions) --------------------------- #
def load_fedxgbllr_f(art_dir):
    import torch
    from omegaconf import OmegaConf
    from xgboost import XGBClassifier
    from torch.utils.data import DataLoader, TensorDataset
    from hfedxgboost.models import CNN
    from hfedxgboost.utils import single_tree_preds_from_each_client
    from evaluation import model_persistence as mp

    arch = json.loads((art_dir / "cnn_arch_config.json").read_text())
    n_est, cnum = int(arch["n_estimators_client"]), int(arch["client_num"])
    cfg = OmegaConf.create({"dataset": {"task": {"task_type": "BINARY"}},
                            "n_estimators_client": n_est, "client_num": cnum,
                            "run_experiment": {"batch_size": 512}})
    trees, cnn = mp.load_fedxgbllr(XGBClassifier, CNN, (cfg,), art_dir)
    cnn = cnn.cpu().eval()
    tlist = [(t, i) for i, t in enumerate(trees)]

    def f(X):
        X = np.asarray(X, np.float32)
        ds = TensorDataset(torch.from_numpy(X), torch.zeros(len(X)))
        tl = single_tree_preds_from_each_client(
            DataLoader(ds, batch_size=len(ds), shuffle=False), 512, tlist, n_est, cnum)
        out = []
        with torch.no_grad():
            for xb, _ in tl:
                out.append(cnn(xb).numpy().reshape(-1))
        return _logit(np.concatenate(out))
    return f


def load_torch_f(art_dir, which):
    import torch
    import torch.nn as nn
    from evaluation import model_persistence as mp
    if which == "bert":
        from models.bert_fraud.model import BertFraudModel as Cls
    else:
        from models.ffd.model import FFDModel as Cls
    model = mp.load_torch(Cls, art_dir, device="cpu").cpu().eval()

    class LogOdds(nn.Module):
        def __init__(self, m): super().__init__(); self.m = m
        def forward(self, x):
            z = self.m(x); return (z[:, 1] - z[:, 0]).reshape(-1, 1)
    wrap = LogOdds(model).cpu().eval()

    def f(X):
        with torch.no_grad():
            return wrap(torch.from_numpy(np.asarray(X, np.float32))).numpy().ravel()
    return f, wrap


# ---- data ------------------------------------------------------------------ #
def client_background(rng):
    from experiments import data_cache
    from imblearn.over_sampling import SMOTE
    clients, _ = data_cache.get_partition_clients(DATASET, 42, CONDITION, ALPHA, 5)
    c = clients[BG_CLIENT]
    xl = np.asarray(c["x"], np.float32); yl = np.asarray(c["y"]).astype(int)
    try:
        xl, yl = SMOTE(sampling_strategy=0.01, k_neighbors=5, random_state=42).fit_resample(xl, yl)
    except ValueError:
        pass  # SMOTE no-op (target met) — use raw local data
    idx = rng.choice(len(xl), min(N_BG, len(xl)), replace=False)
    return xl[idx]


def explanation_subset(rng):
    from experiments import data_cache
    d, _ = data_cache.get_preprocessed(DATASET, 42)
    xte = np.asarray(d["x_test"], np.float32); yte = np.asarray(d["y_test"]).astype(int)
    n_pos = max(1, int(round(N_EXPLAIN * yte.mean())))
    n_neg = N_EXPLAIN - n_pos
    pi = rng.choice(np.where(yte == 1)[0], min(n_pos, int((yte == 1).sum())), replace=False)
    ni = rng.choice(np.where(yte == 0)[0], n_neg, replace=False)
    idx = np.concatenate([pi, ni]); rng.shuffle(idx)
    return xte[idx]


# ---- SHAP + agreement ------------------------------------------------------ #
def kernel_importance(f, bg_km, X, nsamples, seed):
    import shap
    np.random.seed(seed)
    ex = shap.KernelExplainer(f, bg_km)
    sv = _sv_2d(ex.shap_values(X, nsamples=nsamples, silent=True), len(X))
    return np.abs(sv).mean(axis=0)


def deep_importance(wrap, bg_tensor, X_tensor):
    import shap
    ex = shap.DeepExplainer(wrap, bg_tensor)
    sv = _sv_2d(ex.shap_values(X_tensor), len(X_tensor))
    return np.abs(sv).mean(axis=0)


def agreement(a, b, k=5):
    from scipy.stats import spearmanr
    rho = float(spearmanr(a, b).correlation)
    ta = set(np.argsort(a)[::-1][:k]); tb = set(np.argsort(b)[::-1][:k])
    jac = len(ta & tb) / len(ta | tb)
    return rho, jac


# ---- driver ---------------------------------------------------------------- #
def main():
    print("=" * 72)
    print("KernelSHAP NOISE FLOOR — READ-ONLY, writes nothing")
    print(f"dataset={DATASET} {CONDITION} a={ALPHA} | N_explain={N_EXPLAIN} "
          f"bg={N_BG}->kmeans{KMEANS_K} | seeds={SEEDS}")
    print("=" * 72)
    try:
        import shap
        print(f"shap {shap.__version__}")
    except Exception as e:  # noqa: BLE001
        print(f"FATAL shap import: {e}"); return 1

    rng = np.random.default_rng(42)
    fx_dir = _art_dir("fedxgbllr", CONDITION, ALPHA, ARM)
    bert_dir = _art_dir("bert_fraud", CONDITION, ALPHA, ARM)
    ffd_dir = _art_dir("ffd", CONDITION, ALPHA, ARM)
    missing = [d for d in (fx_dir, bert_dir, ffd_dir) if not (d / "manifest.json").is_file()]
    if missing:
        print("\nMissing artifacts (run on the box):")
        for d in missing:
            print("   ", d)
        if len(missing) == 3:
            return 2

    bg = client_background(rng)
    X = explanation_subset(rng)
    bg_km = shap.kmeans(bg, KMEANS_K)
    print(f"background client={BG_CLIENT} n={len(bg)} | explanation n={len(X)} "
          f"({int(0)} shape {X.shape})\n")

    def manifest(d):
        import hashlib
        return hashlib.sha256((d / "manifest.json").read_bytes()).hexdigest()[:16]

    for name, dir_, loader in [("FedXGBllr", fx_dir, lambda: load_fedxgbllr_f(fx_dir)),
                               ("BERT", bert_dir, lambda: load_torch_f(bert_dir, "bert")[0])]:
        if not (dir_ / "manifest.json").is_file():
            print(f"[{name}] no artifact — skipped"); continue
        print(f"### {name} noise floor (manifest {manifest(dir_)})")
        f = loader()
        for ns in NS_LIST:
            t = time.time()
            i1 = kernel_importance(f, bg_km, X, ns, SEEDS[0])
            i2 = kernel_importance(f, bg_km, X, ns, SEEDS[1])
            rho, jac = agreement(i1, i2)
            dt = time.time() - t
            flag = "  <= adopt" if rho > 0.95 else ""
            print(f"    nsamples={ns:>4}: Spearman={rho:.4f}  Jaccard@5={jac:.2f}  "
                  f"[{dt:.0f}s for both runs]{flag}")
        print()

    # FFD cross-check: DeepSHAP (exact-ish) vs KernelSHAP
    if (ffd_dir / "manifest.json").is_file():
        print(f"### FFD DeepSHAP↔KernelSHAP cross-check (manifest {manifest(ffd_dir)})")
        import torch
        f_ffd, wrap = load_torch_f(ffd_dir, "ffd")
        ns = 500
        ik = kernel_importance(f_ffd, bg_km, X, ns, SEEDS[0])
        try:
            idp = deep_importance(wrap, torch.from_numpy(bg.astype(np.float32)),
                                  torch.from_numpy(X.astype(np.float32)))
            rho, jac = agreement(idp, ik)
            mad = float(np.abs(idp - ik).mean())
            print(f"    DeepSHAP vs KernelSHAP(nsamples={ns}): Spearman={rho:.4f} "
                  f"Jaccard@5={jac:.2f} mean|Δ importance|={mad:.2e}")
        except Exception as e:  # noqa: BLE001
            print(f"    DeepSHAP cross-check failed: {type(e).__name__}: {str(e)[:100]}")

    print("\n" + "=" * 72)
    print("Decision rule: adopt smallest nsamples with Spearman > 0.95 on BOTH "
          "FedXGBllr and BERT. If 1000 fails, report floor as a stability limitation.")
    print("No artifact or results file was modified.")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
