#!/usr/bin/env python3
"""Boxplot korelasi peringkat antar client per model: tanpa bobot
(<fig-4-rq3-spearman>) and berbobot (<fig-4-rq3-weighted>), BAB 4 §4.3.1. The
two figures replace the former Tabel «Korelasi peringkat berbobot dan tanpa
bobot antar client pada pengukuran utama».

    python analysis/figures/fig_4_rq3_korelasi_boxplot.py

Writes buku/resources/fig-4-rq3-spearman-boxplot.png and
buku/resources/fig-4-rq3-weighted-boxplot.png.

Source: results/shap_v2/shap_summary_v2.csv (frozen), read only. The cells are
the main measurement of §4.3: local background, the explainer of record per
model (LinearSHAP, TreeSHAP, sampled KernelSHAP), federated conditions only
(terpusat has one model and no clients to compare). That is 11 cells per model
(ULB 4, PaySim 4, BAF 3), 66 in all.

Layout: one figure per column, `spearman` (tanpa bobot) and
`weighted_between` (berbobot), each with one box per model plus the 11 cells
as points. There is no inset, and each figure has its own Y scale (the
figures are not meant to be compared by eye): it starts at that series' lowest
cell floored to 0,05 and runs to 1,00, so every cell is drawn and the boxes
above 0,90 keep as much height as possible. For the current data that is 0,75
tanpa bobot and 0,65 berbobot (FedXGBllr PaySim SMOTE Non-IID, 0,665).

The figures print no numbers, so the prose next to them is the only place the
exact values survive. The paragraph after <fig-4-rq3-weighted> in content.typ
(read only) must therefore quote exactly what this data gives: every weighted
median and FedXGBllr's weighted range. A mismatch stops the build.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

import _common as C

FIG_LABEL = "<fig-4-rq3-weighted>"  # the paragraph after it quotes the values
MAIN_EXPLAINERS = {"linear", "tree", "kernel"}
CELLS_PER_MODEL = 11

# (CSV column, output PNG, tint index in C.TINTS)
SERIES = [("spearman", "fig-4-rq3-spearman-boxplot.png", 5),
          ("weighted_between", "fig-4-rq3-weighted-boxplot.png", 0)]

Y_TOP = 1.005
Y_STEP = 0.05


def f4(x: float) -> str:
    return C.idn(x, 4)


def prose_after_figure() -> str:
    """The paragraph that directly follows FIG_LABEL in content.typ."""
    content = C.PROJECT_ROOT / "buku" / "content.typ"
    if not content.is_file():
        C.fail(f"thesis source not found.\n  looked in: {content}")
    text = content.read_text(encoding="utf-8")
    at = text.find(FIG_LABEL)
    if at < 0:
        C.fail(f"{FIG_LABEL} not found in {content}")
    paras = [p for p in re.split(r"\n\s*\n", text[at + len(FIG_LABEL):]) if p.strip()]
    if not paras:
        C.fail(f"no paragraph after {FIG_LABEL} in {content}")
    return " ".join(paras[0].split())


def draw(ax, vals, pos, jitter, tint, s):
    fill, edge, dark = C.TINTS[tint]
    ax.boxplot(vals, positions=pos, widths=0.5, showfliers=False,
               patch_artist=True,
               boxprops=dict(facecolor=fill, edgecolor=edge, linewidth=1.0),
               whiskerprops=dict(color=edge, linewidth=1.0),
               capprops=dict(color=edge, linewidth=1.0),
               medianprops=dict(color=dark, linewidth=1.6))
    for x, v, j in zip(pos, vals, jitter):
        ax.scatter(x + j, v, s=s, color=dark, alpha=0.6, linewidths=0, zorder=3)


def main() -> None:
    C.apply_style()
    import matplotlib.pyplot as plt

    df = C.read_csv(C.SHAP_SUMMARY, ["dataset", "model", "condition", "arm",
                                     "bg", "explainer", "spearman",
                                     "weighted_between"])
    main_ = df[(df["bg"] == "local") & df["explainer"].isin(MAIN_EXPLAINERS)
               & (df["condition"] != "centralized")].copy()
    # terpusat rows hold the text "undefined" (one client, nothing to correlate),
    # so the columns load as text; every federated value must be a number.
    for col, _t, _k in SERIES:
        num = pd.to_numeric(main_[col], errors="coerce")
        if num.isna().any():
            C.fail(f"non-numeric {col} in federated cells of {C.SHAP_SUMMARY}: "
                   f"{sorted(set(main_.loc[num.isna(), col]))}")
        main_[col] = num
    unknown = set(main_["model"]) - set(C.MODEL_ORDER)
    if unknown:
        C.fail(f"unknown model code(s) {sorted(unknown)} in {C.SHAP_SUMMARY}")
    models = list(C.MODEL_ORDER)
    for m in models:
        n = int((main_["model"] == m).sum())
        if n != CELLS_PER_MODEL:
            C.fail(f"{m}: {n} main federated cells in {C.SHAP_SUMMARY}, "
                   f"expected {CELLS_PER_MODEL}")

    # --- the prose must quote what the data gives ------------------------- #
    stats = {m: {c: main_.loc[main_["model"] == m, c].to_numpy(float)
                 for c, _t, _k in SERIES} for m in models}
    expected = [f4(np.median(stats[m]["weighted_between"])) for m in models]
    fx = stats["fedxgbllr"]["weighted_between"]
    expected += [f4(fx.min()), f4(fx.max())]
    prose = prose_after_figure()
    missing = [v for v in expected if v not in prose]
    if missing:
        C.fail(f"the paragraph after {FIG_LABEL} does not quote {missing}, which "
               f"the data in {C.SHAP_SUMMARY} gives; fix the prose or the data.")

    pos = np.arange(1, len(models) + 1)
    rng = np.random.default_rng(C.SEED)  # fixed jitter: rebuilds are byte-identical
    sources = dict(C._SOURCES)  # save() clears them; both figures read the same CSV
    for col, out, tint in SERIES:
        fig, ax = plt.subplots(figsize=(5.0, 3.6))
        vals = [stats[m][col] for m in models]
        # own Y scale per figure: lowest cell floored to Y_STEP, up to 1,00
        t0 = np.floor(min(v.min() for v in vals) / Y_STEP) * Y_STEP
        ticks = np.arange(t0, 1.0001, Y_STEP)
        jitter = [rng.uniform(-0.13, 0.13, len(v)) for v in vals]
        draw(ax, vals, pos, jitter, tint, s=9)
        ax.set_xticks(pos)
        ax.set_xticklabels([C.label(C.MODEL_LABEL, m, "model") for m in models])
        ax.tick_params(axis="x", length=0)
        ax.set_xlim(0.4, len(models) + 0.6)
        ax.set_ylim(t0 - 0.01, Y_TOP)
        ax.set_yticks(ticks)
        ax.spines["left"].set_bounds(ticks[0], ticks[-1])
        ax.yaxis.set_major_formatter(C.idn_formatter(2))
        ax.grid(axis="y", color=C.RULE, linewidth=0.5)
        ax.set_axisbelow(True)
        ax.set_ylabel("Korelasi peringkat antar client")
        ax.yaxis.set_label_coords(-0.09, 0.5)
        C._SOURCES.update(sources)
        C.save(fig, out)


if __name__ == "__main__":
    main()
