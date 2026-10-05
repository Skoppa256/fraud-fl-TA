#!/usr/bin/env python3
"""Gambar 3.1 <fig-3-1> — Arsitektur umum sistem penelitian (six-stage pipeline).

    python analysis/figures/fig_3_1_pipeline.py

Writes buku/resources/fig-3-1-system-architecture.png.

Data-driven content (each item read from the experiment CSVs, not typed in):
  datasets + feature counts ....... results/analysis/minority_census.csv
  K, Dirichlet alpha, SMOTE arms .. results/clean_summary.csv
  model -> aggregation paradigm ... results/sweep/sweep_master.csv (`aggregation`)
  number of executed runs, seed ... results/clean_summary.csv (row count)
  reported metrics ................ checked present as clean_summary columns
  SHAP explainers ................. results/shap_v2/shap_summary_v2.csv
  train/val/test split ............ test_size literals in preprocessing/*.py
The stage names and the wording around those values are authored here and
follow the stage list in BAB 3 §3.1.
"""

from __future__ import annotations

import _common as C
from _diagram import Canvas, Line, block_height, bullets, title

EXPLAINER_LABEL = {"linear": "LinearSHAP", "tree": "TreeSHAP",
                   "kernel": "KernelSHAP", "kernel-exact-grouped": "KernelSHAP"}


def stages(f: dict):
    ds = ", ".join(f"{C.DATASET_LABEL[d]} ({f['data'][d]['n_features']} fitur)"
                   for d in f["datasets"])
    tr, va, te = f["split"]
    para = [f"{C.PARADIGM_LABEL[p]}: " + ", ".join(C.MODEL_LABEL[m] for m in ms)
            for p, ms in f["paradigms"]]
    found = {C.label(EXPLAINER_LABEL, e, "SHAP explainer") for e in f["explainers"]}
    expl = [e for e in ("LinearSHAP", "TreeSHAP", "KernelSHAP") if e in found]
    scheme_txt = {"centralized": "Terpusat (pembanding)",
                  "iid": "Federated IID",
                  "dirichlet": f"Federated Non-IID (α = {C.idn(f['alpha'], 1)})"}
    arm_txt = {"none": "tanpa SMOTE", "smote": "SMOTE lokal"}
    return [
        ("Akuisisi data dan preprocessing",
         [ds, "Encoding kategorikal dan StandardScaler (fit pada training set)",
          f"Split stratified {tr} : {va} : {te}"]),
        ("Partisi client",
         [f"Training set dibagi ke K = {f['K']} client",
          "IID: pembagian seragam",
          f"Non-IID: Dirichlet α = {C.idn(f['alpha'], 1)}"]),
        ("Pelatihan lokal dan class imbalance",
         ["Pelatihan model pada data lokal setiap client",
          "Dua lengan: " + " vs ".join(arm_txt[a] for a in f["arms"]),
          "Data mentah tidak dibagikan"]),
        ("Agregasi federated", para),
        ("Skenario eksperimen",
         [scheme_txt[s] for s in f["schemes"]]
         + [f"{f['n_runs']} run dieksekusi (seed {f['seed']})"]),
        ("Evaluasi dan explainability",
         ["AUPRC (utama) dan Recall@5%FPR",
          "F1, Precision, Recall pada ambang dari validation set",
          "Brier score dan kalibrasi",
          "SHAP per client (" + ", ".join(expl) + "): konsistensi antar client"]),
    ]


def main() -> None:
    C.apply_style()
    f = C.design_facts()
    st = stages(f)

    W, GAP, ROWGAP = C.TEXT_WIDTH_IN, 0.30, 0.42
    cw = (W - 2 * GAP) / 3
    head_size, body_size, pad = 8.6, 7.8, 0.08
    inner = cw - 2 * pad

    def content(i, name):
        return ([Line(f"TAHAP {i + 1}", size=6.8, weight="bold",
                      color=C.TINTS[i][2])]
                + title(name, inner, size=head_size))

    bodies = [bullets(items, size=body_size, max_w=inner) for _, items in st]
    heads = [content(i, t) for i, (t, _) in enumerate(st)]
    hh = [block_height(h, pad) for h in heads]
    hh = [max(hh[r * 3:r * 3 + 3]) for r in (0, 1) for _ in range(3)]  # align bands
    bh = [block_height(b, pad) for b in bodies]
    row_h = [max(hh[i] + bh[i] for i in range(r * 3, r * 3 + 3)) for r in (0, 1)]
    H = row_h[0] + ROWGAP + row_h[1] + 0.02

    cv = Canvas(W, H)
    pos = []
    for i in range(6):
        r, c = divmod(i, 3)
        x = c * (cw + GAP)
        y = 0.01 if r == 0 else row_h[0] + ROWGAP
        fill, edge, ink = C.TINTS[i]
        cv.rect(x, y, cw, row_h[r], fill="white", edge=edge, lw=0.9)
        cv.rect(x, y, cw, hh[i], fill=fill, edge=edge, lw=0.9)
        cv.lines(x + pad, y + pad, heads[i])
        cv.lines(x + pad, y + hh[i] + pad, bodies[i])
        pos.append((x, y, row_h[r]))

    for i in (0, 1, 3, 4):                       # within-row arrows
        x, y, h = pos[i]
        ym = y + hh[i] / 2
        cv.arrow([(x + cw + 0.04, ym), (x + cw + GAP - 0.04, ym)])
    x2, y2, h2 = pos[2]                          # wrap 3 -> 4
    x3, y3, _ = pos[3]
    ymid = y2 + h2 + ROWGAP / 2
    cv.arrow([(x2 + cw / 2, y2 + h2 + 0.03), (x2 + cw / 2, ymid),
              (x3 + cw / 2, ymid), (x3 + cw / 2, y3 - 0.03)])

    C.save(cv.fig, "fig-3-1-system-architecture.png")


if __name__ == "__main__":
    main()
