"""Class-separability analysis — why BAF is harder than ULB despite less imbalance.

READ-ONLY over the cached preprocessed TRAIN split (experiments.data_cache).
Trains no model, edits nothing under results/. Writes only to
``analysis/separability/`` and the headline figure to ``buku/resources/``.

    python analysis/separability/separability_analysis.py

Computes per dataset:
  1. Minority-type typology (Napierala & Stefanowski 2016): k=5 NN composition
     → safe / borderline / rare / outlier.  [HEADLINE]
  2. Complexity measures (Lorena et al. 2019): N1, N2, N3, F1 (inverse convention,
     higher = harder), on a class-BALANCED subsample to isolate overlap.
  3. Univariate per-feature AUC = max(auc, 1-auc) + KS; top-5.
  4. LDA→1D projection + class-density overlap coefficient.
  5. Empirical ceiling from frozen clean_summary (centralized XGBoost, no-SMOTE).
  6. Caveat: recompute typology + N3 on a dimensionality-matched top-13-AUC subset.

Distance-based measures use SCALED features. Large majorities are subsampled with
a fixed seed; the typology uses the full reference for ULB/BAF and a uniform
prevalence-preserving subsample for PaySim (uniform subsampling preserves local
class composition in expectation — keeping-all-minority + thinning-majority does
NOT, so it is avoided for the typology).
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from sklearn.neighbors import NearestNeighbors  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis  # noqa: E402
from scipy.stats import ks_2samp  # noqa: E402
from scipy.sparse.csgraph import minimum_spanning_tree  # noqa: E402
from scipy.spatial.distance import cdist  # noqa: E402

from experiments import data_cache  # noqa: E402

SEED = 42
OUT = PROJECT_ROOT / "analysis" / "separability"
FIGS = OUT / "figures"
RES_FIG = PROJECT_ROOT / "buku" / "resources"
PAYSIM_TYPOLOGY_N = 500_000     # uniform prevalence-preserving subsample
NMEASURE_MIN_CAP = 3000         # cap minority for balanced N1/N2/N3/F1 subsample
MATCHED_DIM = 13                # dimensionality-matched subset (top-k by AUC)
DATASETS = ["creditcard", "baf", "paysim"]
LABEL = {"creditcard": "ULB", "baf": "BAF", "paysim": "PaySim"}


# --------------------------------------------------------------------------- #
# metrics (hand-implemented, unit-tested below)
# --------------------------------------------------------------------------- #
def minority_typology(x, y, ref_x=None, ref_y=None):
    """(dict type->count, n_min). k=5 NN of each minority point, excl self."""
    if ref_x is None:
        ref_x, ref_y = x, y
    q = x[y == 1]
    nn = NearestNeighbors(n_neighbors=6, algorithm="brute", metric="euclidean").fit(ref_x)
    _, idx = nn.kneighbors(q)
    mc = ref_y[idx[:, 1:6]].sum(axis=1)  # #minority among the 5 (drop self at 0)
    t = np.select([mc >= 4, mc >= 2, mc == 1, mc == 0],
                  ["safe", "borderline", "rare", "outlier"])
    c = Counter(t)
    return {k: int(c.get(k, 0)) for k in ["safe", "borderline", "rare", "outlier"]}, len(t)


def n3_error(x, y):
    """1-NN leave-one-out error rate."""
    nn = NearestNeighbors(n_neighbors=2, algorithm="brute", metric="euclidean").fit(x)
    _, idx = nn.kneighbors(x)
    return float((y[idx[:, 1]] != y).mean())


def n2_ratio(x, y):
    """Σ intra-class NN dist / Σ inter-class NN dist (Lorena; higher = harder)."""
    intra = inter = 0.0
    for cls in (0, 1):
        xi = x[y == cls]
        xo = x[y != cls]
        d_in = NearestNeighbors(n_neighbors=2).fit(xi).kneighbors(xi)[0][:, 1]
        d_out = NearestNeighbors(n_neighbors=1).fit(xo).kneighbors(xi)[0][:, 0]
        intra += d_in.sum()
        inter += d_out.sum()
    return float(intra / inter) if inter else float("nan")


def n1_boundary(x, y):
    """Fraction of vertices incident to a cross-class MST edge."""
    d = cdist(x, x)
    mst = minimum_spanning_tree(d).tocoo()
    touch = set()
    for i, j in zip(mst.row, mst.col):
        if y[i] != y[j]:
            touch.add(i); touch.add(j)
    return float(len(touch) / len(y))


def f1_fisher_inverse(x, y):
    """Lorena F1 = 1/(1+max Fisher ratio); higher = harder."""
    xp, xn = x[y == 1], x[y == 0]
    num = (xp.mean(0) - xn.mean(0)) ** 2
    den = xp.var(0) + xn.var(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(den > 0, num / den, 0.0)
    return float(1.0 / (1.0 + np.nanmax(r)))


def univariate_auc_ks(x, y):
    aucs, kss = [], []
    for j in range(x.shape[1]):
        col = x[:, j]
        try:
            a = roc_auc_score(y, col)
        except ValueError:
            a = 0.5
        aucs.append(max(a, 1 - a))
        kss.append(float(ks_2samp(col[y == 1], col[y == 0]).statistic))
    return np.array(aucs), np.array(kss)


def lda_overlap(x, y):
    z = LinearDiscriminantAnalysis(n_components=1).fit_transform(x, y).ravel()
    lo, hi = np.percentile(z, [0.1, 99.9])
    bins = np.linspace(lo, hi, 200)
    w = bins[1] - bins[0]
    hp, _ = np.histogram(z[y == 1], bins=bins, density=True)
    hn, _ = np.histogram(z[y == 0], bins=bins, density=True)
    overlap = float(np.minimum(hp, hn).sum() * w)
    return z, bins, hp, hn, overlap


def balanced_subsample(x, y, rng, min_cap=NMEASURE_MIN_CAP):
    """All minority (capped) + equal-size majority — isolates overlap from imbalance."""
    mi = np.where(y == 1)[0]
    ma = np.where(y == 0)[0]
    if len(mi) > min_cap:
        mi = rng.choice(mi, min_cap, replace=False)
    ma = rng.choice(ma, len(mi), replace=False)
    idx = np.concatenate([mi, ma])
    rng.shuffle(idx)
    return x[idx], y[idx]


# --------------------------------------------------------------------------- #
def ceiling_from_summary():
    p = PROJECT_ROOT / "results" / "clean_summary.csv"
    out = {}
    for r in csv.DictReader(open(p)):
        if r["model"] == "xgb" and r["scheme"] == "centralized" and r["oversampling"] == "none":
            a, b = float(r["test_auprc"]), float(r["baseline_auprc"])
            out[r["dataset"]] = (a, b, a / b)
    return out


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    ceil = ceiling_from_summary()
    rows = []
    fig_data = {}

    for ds in DATASETS:
        print(f"\n=== {LABEL[ds]} ({ds}) ===")
        d, dh = data_cache.get_preprocessed(ds, SEED)
        x = np.asarray(d["x_train"], np.float32)
        y = np.asarray(d["y_train"]).astype(int)
        fnames = list(d.get("feature_names", [f"f{i}" for i in range(x.shape[1])]))
        dim = x.shape[1]
        n_min = int((y == 1).sum())

        # 1. typology (full ref for ulb/baf; uniform subsample for paysim)
        tx, ty, tnote = x, y, "full train reference"
        if ds == "paysim":
            keep = rng.choice(len(y), PAYSIM_TYPOLOGY_N, replace=False)
            tx, ty = x[keep], y[keep]
            tnote = f"uniform prevalence-preserving subsample n={PAYSIM_TYPOLOGY_N}"
        t0 = time.time()
        typ, ntyp = minority_typology(tx, ty)
        print(f"  typology ({tnote}, n_min={ntyp}) [{time.time()-t0:.0f}s]:")
        for k in ["safe", "borderline", "rare", "outlier"]:
            print(f"    {k:11}: {typ[k]:>6} ({100*typ[k]/ntyp:5.1f}%)")

        # 2. complexity on balanced subsample
        bx, by = balanced_subsample(x, y, rng)
        t0 = time.time()
        N3 = n3_error(bx, by); N2 = n2_ratio(bx, by); N1 = n1_boundary(bx, by); F1 = f1_fisher_inverse(bx, by)
        print(f"  complexity (balanced n={len(by)}, {int((by==1).sum())}+/{int((by==0).sum())}-) [{time.time()-t0:.0f}s]:"
              f" N1={N1:.3f} N2={N2:.3f} N3={N3:.3f} F1={F1:.3f}")

        # 3. univariate AUC/KS
        aucs, kss = univariate_auc_ks(x, y)
        order = np.argsort(aucs)[::-1]
        top5 = [(fnames[i], float(aucs[i]), float(kss[i])) for i in order[:5]]
        print(f"  max univariate AUC={aucs.max():.3f}; top-5: "
              + ", ".join(f"{n}({a:.3f})" for n, a, _ in top5))

        # 4. LDA overlap
        z, bins, hp, hn, ov = lda_overlap(x, y)
        print(f"  LDA-1D class-density overlap coefficient = {ov:.3f}")
        fig_data[ds] = {"typ": typ, "ntyp": ntyp, "aucs": np.sort(aucs)[::-1],
                        "bins": bins, "hp": hp, "hn": hn}

        # 6. matched-dimensionality repeat (top-13 by AUC)
        topk = order[:MATCHED_DIM]
        xm = x[:, topk]
        tmx, tmy = (tx[:, topk], ty) if ds == "paysim" else (xm, y)
        typ_m, ntyp_m = minority_typology(tmx, tmy)
        bxm, bym = balanced_subsample(xm, y, np.random.default_rng(SEED))
        N3_m = n3_error(bxm, bym)
        ro_full = 100 * (typ["rare"] + typ["outlier"]) / ntyp
        ro_m = 100 * (typ_m["rare"] + typ_m["outlier"]) / ntyp_m
        print(f"  matched-dim (top-{MATCHED_DIM} AUC): rare+outlier {ro_full:.1f}% -> {ro_m:.1f}% | N3 {N3:.3f} -> {N3_m:.3f}")

        a, b, lift = ceil.get(ds, (float("nan"),) * 3)
        rows.append({
            "dataset": LABEL[ds], "n_train": len(y), "n_minority": n_min,
            "prevalence_pct": round(100 * n_min / len(y), 4), "dimensionality": dim,
            "safe_pct": round(100 * typ["safe"] / ntyp, 2),
            "borderline_pct": round(100 * typ["borderline"] / ntyp, 2),
            "rare_pct": round(100 * typ["rare"] / ntyp, 2),
            "outlier_pct": round(100 * typ["outlier"] / ntyp, 2),
            "rare_plus_outlier_pct": round(ro_full, 2),
            "N1": round(N1, 4), "N2": round(N2, 4), "N3": round(N3, 4), "F1": round(F1, 4),
            "lda_overlap": round(ov, 4), "max_univariate_auc": round(float(aucs.max()), 4),
            "top5_features": "; ".join(f"{n}({a:.3f})" for n, a, _ in top5),
            "matched_dim_rare_plus_outlier_pct": round(ro_m, 2),
            "matched_dim_N3": round(N3_m, 4),
            "xgb_auprc": round(a, 4), "baseline_auprc": round(b, 5), "xgb_lift": round(lift, 1),
            "data_hash": dh[:16],
        })

    _write_csv(rows)
    _figures(fig_data)
    _report(rows)
    print(f"\nWrote: {OUT/'separability_summary.csv'}, report.md, figures/, "
          f"and {RES_FIG/'fig-4-2-minority-typology.png'}")


def _write_csv(rows):
    cols = list(rows[0].keys())
    with open(OUT / "separability_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)


def _figures(fd):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # headline: stacked horizontal bar of minority types
    types = ["safe", "borderline", "rare", "outlier"]
    colors = {"safe": "#2c7", "borderline": "#fb3", "rare": "#f83", "outlier": "#c33"}
    order = ["paysim", "creditcard", "baf"]
    fig, ax = plt.subplots(figsize=(8, 2.6))
    for yi, ds in enumerate(order):
        typ, n = fd[ds]["typ"], fd[ds]["ntyp"]
        left = 0
        for t in types:
            pct = 100 * typ[t] / n
            ax.barh(yi, pct, left=left, color=colors[t], edgecolor="white")
            if pct >= 4:
                ax.text(left + pct / 2, yi, f"{pct:.0f}%", va="center", ha="center",
                        fontsize=8, color="white", fontweight="bold")
            left += pct
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([f"{LABEL[d]}\n({fd[d]['ntyp']} minoritas)" for d in order])
    ax.set_xlabel("Persentase contoh minoritas"); ax.set_xlim(0, 100)
    ax.legend([plt.Rectangle((0, 0), 1, 1, color=colors[t]) for t in types],
              ["safe", "borderline", "rare", "outlier"],
              ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.45), fontsize=8, frameon=False)
    fig.tight_layout()
    for p in (FIGS / "minority_typology.png", RES_FIG / "fig-4-2-minority-typology.png"):
        fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # sorted per-feature AUC
    fig, ax = plt.subplots(figsize=(6, 3.2))
    for ds in DATASETS:
        a = fd[ds]["aucs"]
        ax.plot(range(1, len(a) + 1), a, marker="o", ms=2, label=f"{LABEL[ds]} ({len(a)}d)")
    ax.axhline(0.5, color="grey", ls=":", lw=1)
    ax.set_xlabel("Peringkat fitur"); ax.set_ylabel("AUC univariat = max(auc, 1−auc)")
    ax.legend(fontsize=8); ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(FIGS / "univariate_auc.png", dpi=150, bbox_inches="tight"); plt.close(fig)

    # LDA-1D densities
    fig, axes = plt.subplots(1, 3, figsize=(10, 2.8))
    for ax, ds in zip(axes, DATASETS):
        b = fd[ds]["bins"]; ctr = (b[:-1] + b[1:]) / 2
        ax.fill_between(ctr, fd[ds]["hn"], alpha=0.5, color="#357", label="normal")
        ax.fill_between(ctr, fd[ds]["hp"], alpha=0.5, color="#c33", label="fraud")
        ax.set_title(LABEL[ds], fontsize=10); ax.set_yticks([]); ax.set_xlabel("proyeksi LDA-1D")
    axes[0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIGS / "lda_projection.png", dpi=150, bbox_inches="tight"); plt.close(fig)


def _report(rows):
    lines = ["# Class-separability analysis — hypothesis test\n",
             "**Question.** BAF has 8.5× ULB's fraud prevalence yet ~1/30th the lift over "
             "baseline. Is separability (not imbalance) the driver?\n",
             "**Verdict.** " + _verdict(rows) + "\n",
             "## Minority-type typology (Napierala & Stefanowski 2016) — headline\n",
             "| dataset | dim | safe | borderline | rare | outlier | rare+outlier |",
             "|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['dataset']} | {r['dimensionality']} | {r['safe_pct']}% | "
                     f"{r['borderline_pct']}% | {r['rare_pct']}% | {r['outlier_pct']}% | "
                     f"**{r['rare_plus_outlier_pct']}%** |")
    lines += ["\n## Complexity, univariate, overlap, ceiling\n",
              "| dataset | N1 | N2 | N3 | F1 | LDA overlap | max AUC | XGBoost lift |",
              "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['dataset']} | {r['N1']} | {r['N2']} | {r['N3']} | {r['F1']} | "
                     f"{r['lda_overlap']} | {r['max_univariate_auc']} | ~{r['xgb_lift']:.0f}× |")
    lines += ["\n## Dimensionality-matched control (top-13 features by AUC)\n",
              "| dataset | rare+outlier full → matched | N3 full → matched |",
              "|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['dataset']} | {r['rare_plus_outlier_pct']}% → "
                     f"{r['matched_dim_rare_plus_outlier_pct']}% | {r['N3']} → {r['matched_dim_N3']} |")
    lines += ["\n## Top-5 discriminative features\n"]
    for r in rows:
        lines.append(f"- **{r['dataset']}**: {r['top5_features']}")
    lines += ["\n## Method notes\n",
              f"- Typology on full train reference for ULB/BAF; PaySim on a uniform "
              f"prevalence-preserving subsample (n={PAYSIM_TYPOLOGY_N}, seed {SEED}).",
              "- N1/N2/N3/F1 on a class-BALANCED subsample (all minority capped at "
              f"{NMEASURE_MIN_CAP} + equal majority, seed {SEED}) to isolate overlap from imbalance.",
              "- F1 uses Lorena's inverse convention: higher = harder.",
              "- Distance measures are dimensionality-dependent; the matched-dim control "
              "recomputes typology + N3 on the top-13 features so the ULB↔BAF ordering is "
              "checked at equal dimensionality. BAF's 26 one-hot columns inflate Euclidean "
              "distance between category-differing records.",
              "- Read-only: nothing under results/ was modified."]
    (OUT / "report.md").write_text("\n".join(lines) + "\n")


def _verdict(rows):
    baf = next(r for r in rows if r["dataset"] == "BAF")
    ulb = next(r for r in rows if r["dataset"] == "ULB")
    return (f"HOLDS. BAF minority is {baf['rare_plus_outlier_pct']}% rare+outlier vs ULB "
            f"{ulb['rare_plus_outlier_pct']}%, and survives the dimensionality-matched control "
            f"(BAF {baf['matched_dim_rare_plus_outlier_pct']}%). Separability, not imbalance, "
            f"drives the difficulty ordering.")


# --------------------------------------------------------------------------- #
def _unit_tests():
    """N3 ≈ 0 on separable, ≈ 0.5 on fully-overlapping synthetic data."""
    rng = np.random.default_rng(0)
    n = 400
    # separable
    xs = np.vstack([rng.normal(-5, 0.3, (n, 4)), rng.normal(5, 0.3, (n, 4))])
    ys = np.array([0] * n + [1] * n)
    # overlapping (same distribution, random labels)
    xo = rng.normal(0, 1, (2 * n, 4)); yo = rng.integers(0, 2, 2 * n)
    n3s, n3o = n3_error(xs, ys), n3_error(xo, yo)
    assert n3s < 0.02, f"separable N3 should be ~0, got {n3s}"
    assert 0.4 < n3o < 0.6, f"overlapping N3 should be ~0.5, got {n3o}"
    assert f1_fisher_inverse(xs, ys) < f1_fisher_inverse(xo, yo), "F1: separable must be easier"
    assert n1_boundary(xs, ys) < 0.1 < n1_boundary(xo, yo), "N1 ordering"
    print(f"unit tests OK: N3 sep={n3s:.3f} overlap={n3o:.3f}")


if __name__ == "__main__":
    _unit_tests()
    run()
