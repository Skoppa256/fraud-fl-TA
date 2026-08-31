"""RQ3 v2 stability-profile figures — READ-ONLY over results/shap_v2/.

Per cell: the chance-corrected Kuncheva consistency index at every top-k
(between-client curve) against the per-client two-seed floor band (within-client
min..max) from the same cell. Where the between curve leaves the band is where
the ranking stops being readable beyond estimator noise. Reads ONLY the
profile.csv / stability.json artifacts written by experiments/shap_rq3.py and
writes ONLY into results/shap_v2/figures/profiles/.

    python analysis/plot_shap_profiles.py
    python analysis/plot_shap_profiles.py --root /path/to/synced/shap_v2
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")  # headless over SSH
import matplotlib.pyplot as plt  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ROOT = PROJECT_ROOT / "results" / "shap_v2"

MODEL_LABEL = {"lr": "LR", "svm": "SVM", "gbm": "GBM", "xgb": "XGB",
               "ffd": "FFD", "bert_fraud": "BERT", "fedxgbllr": "FedXGBllr"}
DATASET_LABEL = {"paysim": "PaySim", "creditcard": "ULB", "baf": "BAF"}
DATASET_COLOR = {"paysim": "#2b6cb0", "creditcard": "#2f855a", "baf": "#c0392b"}
plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
                     "xtick.labelsize": 8, "ytick.labelsize": 8,
                     "legend.fontsize": 8, "figure.dpi": 150})


def read_profile(p: Path):
    rows = list(csv.DictReader(open(p)))

    def col(name):
        return np.array([float(r[name]) if r[name] not in ("", "undefined") else np.nan
                         for r in rows])
    return {"k": col("k"), "between": col("between_mean"),
            "wmean": col("within_mean"), "wmin": col("within_min"),
            "wmax": col("within_max")}


def discover_cells(root: Path):
    for prof in sorted(root.glob("*/*/*/profile.csv")) + \
            sorted(root.glob("*/*/*/bg_shared/profile.csv")):
        cdir = prof.parent
        shared = cdir.name == "bg_shared"
        base = cdir.parent if shared else cdir
        ds, model, cond_arm = base.parts[-3], base.parts[-2], base.parts[-1]
        yield {"dataset": ds, "model": model, "cond_arm": cond_arm,
               "shared": shared, "profile": prof,
               "stability": cdir / "stability.json"}


def crossing_k(prof):
    """First k where the between curve drops below the floor band's minimum."""
    below = prof["between"] < prof["wmin"]
    idx = np.flatnonzero(below & np.isfinite(prof["between"]) & np.isfinite(prof["wmin"]))
    return int(prof["k"][idx[0]]) if idx.size else None


def plot_cell(cell, outdir: Path):
    prof = read_profile(cell["profile"])
    meta = json.loads(cell["stability"].read_text()) if cell["stability"].is_file() else {}
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    ax.fill_between(prof["k"], prof["wmin"], prof["wmax"], alpha=0.25,
                    color="#718096", label="pita floor per-client (min–maks)")
    ax.plot(prof["k"], prof["wmean"], ls="--", lw=1, color="#4a5568",
            label="floor (rerata)")
    ax.plot(prof["k"], prof["between"], lw=1.6,
            color=DATASET_COLOR.get(cell["dataset"], "#333"),
            label="antar-client (CRN)")
    ck = crossing_k(prof)
    if ck is not None:
        ax.axvline(ck, color="#c0392b", lw=0.8, ls=":")
        ax.annotate(f"k = {ck}", (ck, ax.get_ylim()[0]), xytext=(3, 4),
                    textcoords="offset points", fontsize=7, color="#c0392b")
    tag = " · bg bersama" if cell["shared"] else ""
    p = meta.get("p_value")
    sub = f"  (p = {p:.4g})" if isinstance(p, (int, float)) else ""
    ax.set_title(f"{MODEL_LABEL.get(cell['model'], cell['model'])} · "
                 f"{DATASET_LABEL.get(cell['dataset'], cell['dataset'])} / "
                 f"{cell['cond_arm']}{tag}{sub}")
    ax.set_xlabel("k (ukuran himpunan fitur teratas)")
    ax.set_ylabel("indeks Kuncheva")
    ax.set_ylim(min(-0.05, np.nanmin(prof["between"]) - 0.05), 1.02)
    ax.legend(loc="lower left", framealpha=0.7)
    fig.tight_layout()
    name = (f"fig-shap-profile-{cell['dataset']}-{cell['model']}-"
            f"{cell['cond_arm']}" + ("-shared" if cell["shared"] else "") + ".png")
    fig.savefig(outdir / name)
    plt.close(fig)
    return name


def plot_overview(cells, outdir: Path):
    """One panel per kernel model: every cell's between-curve, colored by dataset."""
    models = ["ffd", "bert_fraud", "fedxgbllr"]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2), sharey=True)
    for ax, model in zip(axes, models):
        for cell in cells:
            if cell["model"] != model or cell["shared"]:
                continue
            prof = read_profile(cell["profile"])
            ax.plot(prof["k"], prof["between"], lw=1,
                    color=DATASET_COLOR[cell["dataset"]], alpha=0.7)
        ax.set_title(MODEL_LABEL[model])
        ax.set_xlabel("k")
    axes[0].set_ylabel("indeks Kuncheva antar-client")
    handles = [plt.Line2D([], [], color=DATASET_COLOR[d], label=DATASET_LABEL[d])
               for d in ("paysim", "creditcard", "baf")]
    axes[-1].legend(handles=handles, loc="lower right", framealpha=0.7)
    fig.tight_layout()
    fig.savefig(outdir / "fig-shap-profile-overview.png")
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--outdir", type=Path, default=None)
    args = ap.parse_args(argv)
    outdir = args.outdir or args.root / "figures" / "profiles"
    cells = list(discover_cells(args.root))
    if not cells:
        print(f"no profile.csv under {args.root} — run experiments/shap_rq3.py first.")
        return 2
    outdir.mkdir(parents=True, exist_ok=True)
    for cell in cells:
        print("  ", plot_cell(cell, outdir))
    plot_overview(cells, outdir)
    print(f"{len(cells)} profiles + overview -> {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
