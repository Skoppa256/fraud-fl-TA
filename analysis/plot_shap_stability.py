"""RQ3 SHAP-stability figures — READ-ONLY over results/shap/shap_summary.csv.

Generates the four §4.5 figures plus a text summary. Reads ONLY the summary CSV
and writes ONLY into results/shap/figures/. Never touches results/logs/,
results/models/, results/clean_summary.csv, or results/sweep/.

    python analysis/plot_shap_stability.py

Optional (for local smoke-testing against a synthetic CSV without touching the
canonical outputs):

    python analysis/plot_shap_stability.py --summary /tmp/fake.csv --outdir /tmp/figs

The stability columns are dtype object: degenerate cells write the literal string
"undefined", so every metric is coerced with pd.to_numeric(errors="coerce").
Centralized cells (n_clients == 1) have nothing to compare across and are dropped.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # headless over SSH
import matplotlib.pyplot as plt  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SUMMARY = PROJECT_ROOT / "results" / "shap" / "shap_summary.csv"
DEFAULT_OUTDIR = PROJECT_ROOT / "results" / "shap" / "figures"

# Seeded self-agreement (two seeds, nsamples=500, BAF dirichlet, N_explain=250).
# Deterministic explainers (linear, interventional tree) have no sampling
# variance, so no floor applies to them.
FLOOR = {"fedxgbllr": 0.9730, "bert_fraud": 0.9972, "ffd": 0.9966}

DET_EXPLAINERS = {"linear", "tree"}
METRICS = ["spearman", "kuncheva", "jaccard_at5"]

MODEL_ORDER = ["lr", "svm", "gbm", "xgb", "ffd", "bert_fraud", "fedxgbllr"]
DET_MODELS = ["lr", "svm", "gbm", "xgb"]
MODEL_LABEL = {"lr": "LR", "svm": "SVM", "gbm": "GBM", "xgb": "XGB",
               "ffd": "FFD", "bert_fraud": "BERT", "fedxgbllr": "FedXGBllr"}
METRIC_LABEL = {"spearman": "Spearman", "kuncheva": "Kuncheva", "jaccard_at5": "Jaccard@5"}

DATASET_LABEL = {"paysim": "PaySim", "creditcard": "ULB", "baf": "BAF"}
DATASET_DIM = {"paysim": 13, "creditcard": 30, "baf": 55}
DATASET_MARKER = {"paysim": "o", "creditcard": "s", "baf": "^"}
DATASET_ORDER = ["baf", "creditcard", "paysim"]

COLOR_DET = "#2b6cb0"     # blue: deterministic — stability carries signal
COLOR_KERNEL = "#c0392b"  # red:  KernelSHAP — below floor, no signal
plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
                     "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
                     "figure.dpi": 150})


def model_color(explainer: str) -> str:
    return COLOR_DET if explainer in DET_EXPLAINERS else COLOR_KERNEL


def cell_id(row) -> str:
    return f"{MODEL_LABEL.get(row['model'], row['model'])} · {DATASET_LABEL.get(row['dataset'], row['dataset'])} / {row['condition']} / {row['arm']}"


def load(summary_path: Path) -> pd.DataFrame:
    df = pd.read_csv(summary_path)
    for c in METRICS:
        df[c] = pd.to_numeric(df.get(c), errors="coerce")   # "undefined" -> NaN
    df["n_clients"] = pd.to_numeric(df.get("n_clients"), errors="coerce")
    df = df[df["n_clients"] > 1].copy()                       # drop centralized (1 client)
    df["floor"] = df["model"].map(FLOOR)                      # NaN for deterministic
    return df


# --------------------------------------------------------------------------- #
def fig_vs_floor(df: pd.DataFrame, out: Path):
    k = df[(df["explainer"] == "kernel") & df["spearman"].notna() & df["floor"].notna()].copy()
    k["margin"] = k["spearman"] - k["floor"]
    k = k.sort_values("margin", ascending=True).reset_index(drop=True)
    y = np.arange(len(k))
    fig, ax = plt.subplots(figsize=(8, max(4, 0.30 * len(k) + 1)))
    ax.barh(y, k["spearman"], color=COLOR_KERNEL, alpha=0.85, height=0.62,
            label="observed cross-client Spearman")
    for yi, fl in zip(y, k["floor"]):
        ax.vlines(fl, yi - 0.38, yi + 0.38, color="black", lw=1.6)
    ax.vlines([], [], [], color="black", lw=1.6, label="model noise floor")  # legend proxy
    ax.set_yticks(y)
    ax.set_yticklabels([cell_id(r) for _, r in k.iterrows()])
    ax.set_xlabel("Cross-client Spearman")
    below = int((k["spearman"] <= k["floor"]).sum())
    ax.set_title(f"KernelSHAP cross-client stability vs. own noise floor "
                 f"({below}/{len(k)} at or below floor)")
    lo = float(min(k["spearman"].min(), k["floor"].min()))
    ax.set_xlim(max(0.0, lo - 0.05), 1.005)
    ax.margins(y=0.01)
    ax.legend(loc="lower right", frameon=True)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig_by_model(df: pd.DataFrame, out: Path):
    models = [m for m in MODEL_ORDER if m in set(df["model"])]
    expl_of = {m: df[df["model"] == m]["explainer"].iloc[0] for m in models}
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    for ax, metric in zip(axes, METRICS):
        data = [df[(df["model"] == m)][metric].dropna().values for m in models]
        positions = np.arange(len(models))
        bp = ax.boxplot(data, positions=positions, widths=0.6, patch_artist=True,
                        showfliers=False, medianprops=dict(color="black"))
        for patch, m in zip(bp["boxes"], models):
            patch.set_facecolor(model_color(expl_of[m]))
            patch.set_alpha(0.35)
        for xi, (m, vals) in enumerate(zip(models, data)):
            if len(vals):
                jit = rng.uniform(-0.16, 0.16, size=len(vals))
                ax.scatter(np.full(len(vals), xi) + jit, vals, s=16,
                           color=model_color(expl_of[m]), edgecolor="white", lw=0.4, zorder=3)
        if metric == "spearman":  # floor line per KernelSHAP model
            for xi, m in enumerate(models):
                if m in FLOOR:
                    ax.hlines(FLOOR[m], xi - 0.34, xi + 0.34, color=COLOR_KERNEL,
                              ls="--", lw=1.3, zorder=4)
        ax.set_xticks(positions)
        ax.set_xticklabels([MODEL_LABEL[m] for m in models], rotation=45, ha="right")
        ax.set_title(METRIC_LABEL[metric])
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("stability")
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLOR_DET, alpha=0.55),
               plt.Rectangle((0, 0), 1, 1, color=COLOR_KERNEL, alpha=0.55),
               plt.Line2D([0], [0], color=COLOR_KERNEL, ls="--", lw=1.3)]
    fig.legend(handles, ["deterministic (signal)", "KernelSHAP (no signal — below floor)",
                         "KernelSHAP noise floor"],
               loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig_heatmap_det(df: pd.DataFrame, out: Path):
    det = df[df["explainer"].isin(DET_EXPLAINERS) & df["kuncheva"].notna()].copy()
    det["col"] = (det["dataset"].map(lambda d: DATASET_LABEL.get(d, d))
                  + "/" + det["condition"] + "/" + det["arm"])
    rows = [m for m in DET_MODELS if m in set(det["model"])]
    piv = det.pivot_table(index="model", columns="col", values="kuncheva", aggfunc="mean")
    piv = piv.reindex(rows)
    # order columns by dataset then condition/arm for readability
    col_order = sorted(piv.columns, key=lambda c: (DATASET_ORDER.index(c.split("/")[0].lower())
                                                   if c.split("/")[0].lower() in DATASET_ORDER
                                                   else {"BAF": 0, "ULB": 1, "PaySim": 2}.get(c.split("/")[0], 9), c))
    piv = piv[col_order]
    M = piv.values.astype(float)
    fig, ax = plt.subplots(figsize=(max(7, 0.55 * M.shape[1] + 2), 0.6 * M.shape[0] + 2.2))
    im = ax.imshow(M, cmap="RdYlGn", vmin=0.4, vmax=1.0, aspect="auto")
    ax.set_xticks(np.arange(M.shape[1]))
    ax.set_xticklabels(piv.columns, rotation=45, ha="right")
    ax.set_yticks(np.arange(M.shape[0]))
    ax.set_yticklabels([MODEL_LABEL[m] for m in piv.index])
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if not np.isnan(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=7,
                        color="black")
    ax.set_title("Kuncheva index — deterministic explainers (the measurable RQ3 signal)")
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cb.set_label("Kuncheva")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig_jaccard_vs_kuncheva(df: pd.DataFrame, out: Path):
    d = df[df["jaccard_at5"].notna() & df["kuncheva"].notna()]
    fig, ax = plt.subplots(figsize=(5.4, 5.0))
    ax.plot([0, 1], [0, 1], color="grey", ls=":", lw=1, zorder=1, label="identity")
    for ds in DATASET_ORDER:
        sub = d[d["dataset"] == ds]
        if not len(sub):
            continue
        ax.scatter(sub["kuncheva"], sub["jaccard_at5"], marker=DATASET_MARKER[ds], s=34,
                   alpha=0.75, edgecolor="white", lw=0.4,
                   label=f"{DATASET_LABEL[ds]} (d={DATASET_DIM[ds]})")
    ax.set_xlabel("Kuncheva (chance-corrected)")
    ax.set_ylabel("Jaccard@5 (not chance-corrected)")
    ax.set_title("Jaccard@5 vs. Kuncheva — divergence grows with dimensionality")
    ax.set_xlim(0.3, 1.02)
    ax.set_ylim(0.3, 1.02)
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
def text_summary(df: pd.DataFrame) -> str:
    L = []
    L.append("SHAP stability summary (federated cells only, n_clients > 1)")
    L.append("=" * 60)
    L.append(f"total federated cells: {len(df)}")
    L.append("")
    L.append("cell counts by explainer:")
    for expl, n in df["explainer"].value_counts().items():
        L.append(f"  {expl:<8} {n}")
    L.append("")
    k = df[(df["explainer"] == "kernel") & df["spearman"].notna() & df["floor"].notna()]
    below = int((k["spearman"] <= k["floor"]).sum())
    frac = f"{below}/{len(k)}" + (f" ({below/len(k):.0%})" if len(k) else "")
    L.append(f"KernelSHAP cells at or below their own floor: {frac}")
    L.append("")
    det = df[df["explainer"].isin(DET_EXPLAINERS) & df["kuncheva"].notna()]
    L.append("mean Kuncheva per (dataset, model) — deterministic models only:")
    piv = det.pivot_table(index="dataset", columns="model", values="kuncheva", aggfunc="mean")
    piv = piv.reindex([d for d in DATASET_ORDER if d in piv.index])
    piv = piv[[m for m in DET_MODELS if m in piv.columns]]
    hdr = "  " + f"{'dataset':<10}" + "".join(f"{MODEL_LABEL[m]:>9}" for m in piv.columns)
    L.append(hdr)
    for ds, r in piv.iterrows():
        L.append("  " + f"{DATASET_LABEL.get(ds, ds):<10}"
                 + "".join(("        -" if pd.isna(r[m]) else f"{r[m]:>9.3f}") for m in piv.columns))
    L.append("")
    L.append("five least stable deterministic cells by Kuncheva:")
    worst = det.sort_values("kuncheva", ascending=True).head(5)
    for _, r in worst.iterrows():
        L.append(f"  {r['kuncheva']:.4f}  {cell_id(r)}  "
                 f"(Jaccard@5={r['jaccard_at5']:.4f}, Spearman={r['spearman']:.4f})")
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    ap.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    args = ap.parse_args(argv)

    if not args.summary.is_file():
        print(f"NO summary at {args.summary} — run the SHAP grid on the box first.")
        return 2
    df = load(args.summary)
    if df.empty:
        print("No federated cells (n_clients > 1) in summary — nothing to plot.")
        return 2

    args.outdir.mkdir(parents=True, exist_ok=True)
    fig_vs_floor(df, args.outdir / "fig-shap-vs-floor.png")
    fig_by_model(df, args.outdir / "fig-shap-by-model.png")
    fig_heatmap_det(df, args.outdir / "fig-shap-heatmap-deterministic.png")
    fig_jaccard_vs_kuncheva(df, args.outdir / "fig-shap-jaccard-vs-kuncheva.png")

    summary = text_summary(df)
    (args.outdir / "stability_summary.txt").write_text(summary + "\n")
    print(summary)
    print(f"\nWrote 4 figures + stability_summary.txt -> {args.outdir}")
    print("Nothing outside the figures directory was written.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
