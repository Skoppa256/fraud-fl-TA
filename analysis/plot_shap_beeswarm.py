"""Per-client SHAP beeswarms — RUN ON THE BOX (frozen model artifacts live there).

Makes client-to-client feature-importance differences visible, beyond the Spearman/
Kuncheva summaries. Deliberately narrow scope:

  * BAF only (the sole dataset with real named features).
  * Deterministic explainers only — LR/SVM (LinearSHAP), GBM (interventional TreeSHAP);
    these are the only cells whose cross-client stability is resolvable (every
    KernelSHAP cell sits at or below its own noise floor).
  * Dirichlet alpha = 0.5, both arms, all five clients.

It REUSES the exact experiments/shap_analysis.py pipeline (same explanation set, same
per-client backgrounds, same explainer construction) so the extracted raw SHAP values
reproduce the frozen mean|SHAP| aggregates. Two guarantees the caller asked for:

  1. Raw matrices are persisted under results/shap/raw/ (regenerate/re-style later
     without another extraction run).
  2. Mean|SHAP| recomputed from the fresh raw values is VERIFIED against the committed
     importance_per_client.csv for each cell. A mismatch aborts with a report — it
     means the re-run is not reproducing the original explanation.

READ-ONLY except results/shap/raw/ and the figures directory.

    python analysis/plot_shap_beeswarm.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")  # headless over SSH
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "models" / "fedxgbllr"))

from experiments import shap_analysis as SA  # noqa: E402

DATASET = "baf"
MODELS = ["lr", "svm", "gbm"]
CONDITION = "dirichlet"
ALPHA = 0.5
ARMS = ["none", "smote"]
TOPK = 12          # features shown per beeswarm (shared consensus ordering)
TOPK_BARS = 10     # features in the grouped-bar comparison
VERIFY_ATOL = 2e-4  # importance_per_client.csv stores ~6 sig figs

OUT = PROJECT_ROOT / "results" / "shap" / "figures" / "beeswarm"
RAW = PROJECT_ROOT / "results" / "shap" / "raw"
plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "figure.dpi": 150})
MODEL_LABEL = {"lr": "LR", "svm": "SVM", "gbm": "GBM"}
ARM_LABEL = {"none": "tanpa-SMOTE", "smote": "dengan-SMOTE"}


def client_shap_matrix(obj, kind, bg, X):
    """Full (n_samples x n_features) SHAP matrix — mirrors SA.client_importance
    exactly but keeps the matrix instead of collapsing to mean|SHAP|."""
    import shap
    if kind == "linear":
        ex = shap.LinearExplainer(obj, bg)
        return SA._sv2d(ex.shap_values(X), len(X))
    if kind == "tree":
        ex = shap.TreeExplainer(obj, data=bg, feature_perturbation="interventional")
        return SA._sv2d(ex.shap_values(X, check_additivity=False), len(X))
    raise ValueError(f"beeswarm scope is deterministic only; got kind={kind}")


def committed_importance(art):
    """mean|SHAP| per (feature, client) from the frozen importance_per_client.csv."""
    import csv
    p = SA.OUT / art["dataset"] / art["model"] / f"{art['condition']}_{art['arm']}" / "importance_per_client.csv"
    if not p.is_file():
        return None
    rows = list(csv.DictReader(open(p)))
    feats = [r["feature"] for r in rows]
    cols = [c for c in rows[0].keys() if c.startswith("client_")]
    mat = np.array([[float(r[c]) for c in cols] for r in rows])  # (d, n_clients)
    return feats, mat


def extract_cell(art):
    """Return (features, sv[n_clients,N,d], X[N,d]) and verify against frozen aggregate."""
    fnames = SA.feature_names(art)
    X, _ = SA.explanation_set(art["dataset"], np.random.default_rng(SA.SEED))
    bgs, _ = SA.client_backgrounds(art, np.random.default_rng(SA.SEED + 1))
    obj, kind = SA.load_predictor(art)
    svs = [client_shap_matrix(obj, kind, bg, X) for bg in bgs]
    sv = np.stack(svs)                       # (n_clients, N, d)

    # verify mean|SHAP| against committed importance_per_client.csv
    fresh = np.abs(sv).mean(axis=1).T        # (d, n_clients)
    ref = committed_importance(art)
    if ref is None:
        print(f"    [warn] no committed importance_per_client.csv to verify against")
    else:
        ref_feats, ref_mat = ref
        assert ref_feats == list(fnames), "feature order differs from committed CSV"
        md = float(np.abs(fresh - ref_mat).max())
        status = "OK" if md <= VERIFY_ATOL else "MISMATCH"
        print(f"    verify mean|SHAP| vs frozen: max|Δ|={md:.2e} [{status}]")
        if md > VERIFY_ATOL:
            raise SystemExit(
                f"ABORT: {art['model']}/{art['arm']} raw extraction does not reproduce "
                f"the frozen aggregate (max|Δ|={md:.2e} > {VERIFY_ATOL}). "
                f"The re-run is not the original explanation; do not use these figures.")
    return list(fnames), sv, X


def consensus_order(fnames, sv, k):
    """Feature indices sorted by mean|SHAP| pooled across clients (shared ordering)."""
    imp = np.abs(sv).mean(axis=(0, 1))       # (d,)
    return list(np.argsort(imp)[::-1][:k])


def beeswarm_panel(ax, sv_c, Xc, order, fnames, rng, vmin, vmax):
    """One client's beeswarm on shared feature ordering (top rows = most important)."""
    for row, j in enumerate(order):
        y0 = len(order) - 1 - row
        vals = sv_c[:, j]
        fv = Xc[:, j].astype(float)
        denom = (vmax[j] - vmin[j]) or 1.0
        col = np.clip((fv - vmin[j]) / denom, 0, 1)
        jit = rng.uniform(-0.32, 0.32, size=len(vals))
        ax.scatter(vals, np.full(len(vals), y0) + jit, c=col, cmap="coolwarm",
                   s=5, alpha=0.6, linewidths=0, vmin=0, vmax=1, rasterized=True)
    ax.axvline(0, color="grey", lw=0.6, zorder=0)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([fnames[order[len(order) - 1 - r]] for r in range(len(order))])
    ax.tick_params(axis="y", labelsize=6)


def fig_beeswarm(art, fnames, sv, X):
    order = consensus_order(fnames, sv, TOPK)
    vmin = X.min(axis=0); vmax = X.max(axis=0)
    xall = sv[:, :, order]
    xlo, xhi = float(np.percentile(xall, 0.5)), float(np.percentile(xall, 99.5))
    n = sv.shape[0]
    fig, axes = plt.subplots(1, n, figsize=(3.1 * n, 4.2), sharey=True)
    rng = np.random.default_rng(0)
    for ci, ax in enumerate(axes):
        beeswarm_panel(ax, sv[ci], X, order, fnames, rng, vmin, vmax)
        ax.set_title(f"client {ci}")
        ax.set_xlim(xlo, xhi)
        if ci == 0:
            ax.set_ylabel("fitur (urutan konsensus)")
        ax.set_xlabel("SHAP (log-odds)")
    sm = plt.cm.ScalarMappable(cmap="coolwarm", norm=plt.Normalize(0, 1))
    cb = fig.colorbar(sm, ax=axes, fraction=0.015, pad=0.01)
    cb.set_ticks([0, 1]); cb.set_ticklabels(["rendah", "tinggi"]); cb.set_label("nilai fitur")
    fig.suptitle(f"BAF {MODEL_LABEL[art['model']]} — Dirichlet α=0.5, {ARM_LABEL[art['arm']]}",
                 fontsize=10)
    p = OUT / f"fig-shap-beeswarm-baf-{art['model']}-{art['arm']}.png"
    fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    return p


def fig_clientbars(model, cells):
    """Grouped bars: mean|SHAP| of top-10 consensus features, 5 bars/feature, per arm."""
    fig, axes = plt.subplots(1, len(cells), figsize=(6.6 * len(cells), 4.0), sharey=False)
    if len(cells) == 1:
        axes = [axes]
    for ax, (art, fnames, sv, X) in zip(axes, cells):
        order = consensus_order(fnames, sv, TOPK_BARS)
        imp = np.abs(sv).mean(axis=1)         # (n_clients, d)
        n = sv.shape[0]
        w = 0.8 / n
        xs = np.arange(len(order))
        for ci in range(n):
            ax.bar(xs + (ci - (n - 1) / 2) * w, imp[ci, order], width=w, label=f"client {ci}")
        ax.set_xticks(xs)
        ax.set_xticklabels([fnames[j] for j in order], rotation=45, ha="right", fontsize=6)
        ax.set_title(f"{MODEL_LABEL[model]} — {ARM_LABEL[art['arm']]}")
        ax.set_ylabel("mean |SHAP|")
        ax.grid(axis="y", alpha=0.25)
    axes[-1].legend(fontsize=7, ncol=1)
    p = OUT / f"fig-shap-baf-{model}-clientbars.png"
    fig.tight_layout(); fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    return p


def main():
    try:
        import shap  # noqa: F401
    except Exception as e:  # noqa: BLE001
        print(f"FATAL shap import: {e}"); return 1
    arts = SA.discover()
    if not arts:
        print("NO artifacts under results/models/ — run on the box."); return 2
    sel = [a for a in arts if a["dataset"] == DATASET and a["model"] in MODELS
           and a["condition"] == CONDITION and a["arm"] in ARMS
           and (a["alpha"] == ALPHA)]
    if not sel:
        print(f"No BAF {CONDITION} α={ALPHA} deterministic cells found."); return 2
    OUT.mkdir(parents=True, exist_ok=True)
    RAW.mkdir(parents=True, exist_ok=True)
    print(f"{len(sel)} cells in scope (BAF {CONDITION} α={ALPHA}; {MODELS}; {ARMS}).")

    by_model = {m: [] for m in MODELS}
    for art in sorted(sel, key=lambda a: (MODELS.index(a["model"]), a["arm"])):
        tag = f"{art['model']}/{art['arm']}"
        print(f"  [{tag}] extracting raw SHAP ...")
        fnames, sv, X = extract_cell(art)
        np.savez_compressed(RAW / f"{DATASET}__{art['model']}__{CONDITION}_{art['arm']}.npz",
                            sv=sv, X=X, features=np.array(fnames))
        p = fig_beeswarm(art, fnames, sv, X)
        print(f"    wrote {p.name}")
        by_model[art["model"]].append((art, fnames, sv, X))

    for m, cells in by_model.items():
        if cells:
            cells = sorted(cells, key=lambda c: ARMS.index(c[0]["arm"]))
            p = fig_clientbars(m, cells)
            print(f"  wrote {p.name}")

    print(f"\nRaw matrices -> {RAW.relative_to(PROJECT_ROOT)}/ ; figures -> "
          f"{OUT.relative_to(PROJECT_ROOT)}/")
    print("Nothing outside results/shap/raw/ and the figures dir was written.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
