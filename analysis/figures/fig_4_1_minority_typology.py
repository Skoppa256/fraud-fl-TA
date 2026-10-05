#!/usr/bin/env python3
"""Gambar 4.1 <fig-typology> — Distribusi tipe contoh minoritas per dataset.

    python analysis/figures/fig_4_1_minority_typology.py

Writes buku/resources/fig-4-2-minority-typology.png (filename kept so BAB 4's
Typst source stays byte-identical).

Source: analysis/separability/separability_summary.csv, written by
analysis/separability/separability_analysis.py (Napierala & Stefanowski typology,
k = 5). The four shares per dataset must sum to 100% (to the CSV's 2-dp rounding)
and must agree with rare_plus_outlier_pct, or the script aborts. The segment labels
(one decimal, round-half-even) must also equal the values printed in BAB 4's
<tab-typology>, which this figure sits next to; content.typ is only read.

The bar labels carry the dataset's dimensionality (`dimensionality` column) rather
than a minority count: the count the typology was computed on is not in the CSV
for PaySim (a 500k prevalence-preserving subsample), and `n_minority` there is the
full training count, so printing it would misstate the base of the percentages.
"""

from __future__ import annotations

import numpy as np

import _common as C

TYPES = ["safe", "borderline", "rare", "outlier"]
# bottom-to-top order of the existing figure: PaySim, ULB, BAF (BAF on top)
ORDER = ["PaySim", "ULB", "BAF"]
TABLE_LABEL = "<tab-typology>"


def pct1(x: float) -> str:
    """One decimal, round-half-even on the CSV's decimal text (6.65 -> 6,6), the
    rule that reproduces every value typed into <tab-typology>."""
    from decimal import ROUND_HALF_EVEN, Decimal
    return C.idn(float(Decimal(repr(float(x))).quantize(Decimal("0.1"),
                                                         ROUND_HALF_EVEN)), 1)


def table_values() -> dict:
    """The four shares per dataset as printed in BAB 4's <tab-typology>
    (read-only), keyed by dataset -> [safe, borderline, rare, outlier]."""
    import re
    content = C.PROJECT_ROOT / "buku" / "content.typ"
    if not content.is_file():
        C.fail(f"thesis source not found.\n  looked in: {content}")
    lines = content.read_text(encoding="utf-8").splitlines()
    end = next((i for i, l in enumerate(lines) if TABLE_LABEL in l), None)
    if end is None:
        C.fail(f"{TABLE_LABEL} not found in {content}")
    start = max(i for i in range(end) if lines[i].lstrip().startswith("#figure("))
    out = {}
    for l in lines[start:end]:
        m = re.match(r"\s*\[(\w+)\],\s*\[\d+\],(.*)", l)
        if m and m.group(1) in ORDER:
            cells = re.findall(r"\[\*?([0-9]+,[0-9])%\*?\]", m.group(2))
            if len(cells) != 5:
                C.fail(f"{content}: cannot parse {TABLE_LABEL} row: {l.strip()}")
            out[m.group(1)] = cells[:4]
    if set(out) != set(ORDER):
        C.fail(f"{content}: {TABLE_LABEL} rows found for {sorted(out)}, "
               f"expected {ORDER}")
    return out


def main() -> None:
    C.apply_style()
    import matplotlib.pyplot as plt

    cols = [f"{t}_pct" for t in TYPES]
    df = C.read_csv(C.SEPARABILITY, ["dataset", "dimensionality",
                                     "rare_plus_outlier_pct", *cols])
    missing = [d for d in ORDER if d not in set(df["dataset"])]
    if missing:
        C.fail(f"dataset(s) {missing} missing from {C.SEPARABILITY}")
    df = df.set_index("dataset").loc[ORDER]
    for d, r in df.iterrows():
        s = float(r[cols].sum())
        if abs(s - 100) > 0.05:
            C.fail(f"{d}: typology shares sum to {s:.2f}%, not 100%")
        if abs(r["rare_pct"] + r["outlier_pct"] - r["rare_plus_outlier_pct"]) > 0.02:
            C.fail(f"{d}: rare+outlier ({r['rare_pct'] + r['outlier_pct']:.2f}) "
                   f"!= rare_plus_outlier_pct ({r['rare_plus_outlier_pct']})")

    # The figure sits next to <tab-typology> in BAB 4; the two must print the
    # same numbers. A mismatch means the CSV or the table moved: stop and say so.
    tab = table_values()
    for d in ORDER:
        mine = [pct1(df.loc[d, c]) for c in cols]
        if mine != tab[d]:
            C.fail(f"{d}: figure labels {mine} disagree with {TABLE_LABEL} {tab[d]} "
                   f"(source {C.SEPARABILITY}); fix the table or the data, not this check.")

    fig, ax = plt.subplots(figsize=(C.TEXT_WIDTH_IN, 2.05))
    y = np.arange(len(ORDER))
    left = np.zeros(len(ORDER))
    for t, c in zip(TYPES, cols):
        v = df[c].to_numpy(float)
        ax.barh(y, v, left=left, height=0.72, color=C.TYPOLOGY_COLORS[t],
                edgecolor="white", linewidth=0.8, label=t)
        for yi, (l, w) in enumerate(zip(left, v)):
            # one decimal, same rounding as <tab-typology> (checked above);
            # segments under 5% are too narrow to hold a label
            if w >= 5:
                ax.text(l + w / 2, yi, f"{pct1(w)}%", ha="center", va="center",
                        fontsize=7.8, color="white", fontweight="bold")
        left += v
    ax.set_yticks(y)
    ax.set_yticklabels([f"{d}\n(d = {int(df.loc[d, 'dimensionality'])})"
                        for d in ORDER])
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, 100)
    ax.xaxis.set_major_formatter(C.idn_formatter(0))
    ax.set_xlabel("Persentase contoh minoritas (%)")
    ax.spines["left"].set_visible(False)
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.30),
              frameon=False, handlelength=1.2, columnspacing=1.6)

    C.save(fig, "fig-4-2-minority-typology.png")


if __name__ == "__main__":
    main()
