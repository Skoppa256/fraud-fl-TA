#!/usr/bin/env python3
"""Boxplot AUPRC test seluruh sel per dataset (BAB 4, paragraf setelah
<tab-4-auprc-paysim>).

    python analysis/figures/fig_4_auprc_boxplot.py

Writes buku/resources/fig-4-auprc-boxplot.png.

Source: results/clean_summary.csv (frozen, 96 rows), read only. One box per
dataset over every executed cell: all models (XGBoost included), all three
conditions (Terpusat, IID, Non-IID) and both arms (SMOTE, tanpa SMOTE). The
individual cells are drawn over the box, because PaySim is bimodal (GBM and
FedXGBllr near 0,996, the rest near 0,6) and a box alone hides that.

The prevalence in each x label is the dataset's `baseline_auprc` (the AUPRC of
a random classifier, i.e. the test-set fraud rate), the same "batas bawah" the
surrounding prose divides by.
"""

from __future__ import annotations

import numpy as np

import _common as C

N_CELLS = 96
OUT = "fig-4-auprc-boxplot.png"


def pct_label(p: float) -> str:
    """Fraud rate as a percentage with at most two decimals and no trailing
    zero, Indonesian style: 0.00129 -> '0,13%', 0.01103 -> '1,1%'."""
    s = C.idn(100 * p, 2).rstrip("0").rstrip(",")
    return f"{s}%"


def main() -> None:
    C.apply_style()
    import matplotlib.pyplot as plt

    df = C.read_csv(C.CLEAN_SUMMARY, ["dataset", "model", "scheme",
                                      "oversampling", "test_auprc",
                                      "baseline_auprc"])
    if len(df) != N_CELLS:
        C.fail(f"expected {N_CELLS} cells in {C.CLEAN_SUMMARY}, found {len(df)}")
    unknown = set(df["dataset"]) - set(C.DATASET_ORDER)
    if unknown:
        C.fail(f"unknown dataset code(s) {sorted(unknown)} in {C.CLEAN_SUMMARY}")
    if df["test_auprc"].isna().any() or not df["test_auprc"].between(0, 1).all():
        C.fail(f"test_auprc outside [0, 1] or missing in {C.CLEAN_SUMMARY}")

    datasets = [d for d in C.DATASET_ORDER if d in set(df["dataset"])]
    values, labels = [], []
    for d in datasets:
        g = df[df["dataset"] == d]
        # baseline_auprc differs between rows only in float noise (~1e-10)
        base = g["baseline_auprc"].round(8)
        prev = float(C.one(base, f"baseline_auprc for {d}"))
        values.append(g["test_auprc"].to_numpy(float))
        labels.append(f"{C.label(C.DATASET_LABEL, d, 'dataset')}\n({pct_label(prev)})")

    _fill, edge, dark = C.TINTS[0]
    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    pos = np.arange(1, len(datasets) + 1)
    ax.boxplot(values, positions=pos, widths=0.5, showfliers=False,
               patch_artist=True,
               boxprops=dict(facecolor=_fill, edgecolor=edge, linewidth=1.0),
               whiskerprops=dict(color=edge, linewidth=1.0),
               capprops=dict(color=edge, linewidth=1.0),
               medianprops=dict(color=dark, linewidth=1.6))

    rng = np.random.default_rng(C.SEED)  # fixed jitter: rebuilds are byte-identical
    for x, v in zip(pos, values):
        ax.scatter(x + rng.uniform(-0.14, 0.14, len(v)), v, s=9, color=C.INK2,
                   alpha=0.65, linewidths=0, zorder=3)

    ax.set_xticks(pos)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Dataset (rasio fraud)")
    ax.set_ylabel("AUPRC test")
    ax.set_ylim(0, 1.02)
    ax.yaxis.set_major_formatter(C.idn_formatter(1))
    ax.grid(axis="y", color=C.RULE, linewidth=0.5)
    ax.set_axisbelow(True)

    C.save(fig, OUT)


if __name__ == "__main__":
    main()
