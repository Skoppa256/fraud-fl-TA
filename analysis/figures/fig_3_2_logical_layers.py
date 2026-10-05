#!/usr/bin/env python3
"""Gambar 3.2 <fig-3-2> — Lapisan logis sistem penelitian (four logical layers).

    python analysis/figures/fig_3_2_logical_layers.py

Writes buku/resources/fig-3-2-logical-layers.png.

Data-driven content:
  dataset cards (training rows, features, fraud share) .. results/analysis/minority_census.csv
                                                          (seed-42 IID clients, summed)
  K, Dirichlet alpha .................................... results/clean_summary.csv
  paradigm cards and their models ....................... results/sweep/sweep_master.csv
  split ratio ........................................... preprocessing/*.py test_size
Layer names follow BAB 3 §3.3 ("empat lapisan logis").
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import _common as C
from _diagram import Canvas, Line, block_height, line_h, wrap

NUM_WORD = {1: "satu", 2: "dua", 3: "tiga", 4: "empat", 5: "lima", 6: "enam"}

# model-family note under each paradigm card (authored; keyed by sweep code)
PARADIGM_NOTE = {
    "FedAvg": "model parametrik",
    "best-model-selection": "tree ensemble",
    "accuracy-weighted-FedAvg": "deep learning",
    "tree-ensemble+CNN-FedAvg": "tree + 1D CNN",
}

Card = Tuple[str, Sequence[str]]

PAD_X, PAD_Y = 0.07, 0.065
GAP = 0.09
T_SIZE, D_SIZE = 8.2, 7.4


def card_lines(title: str, desc: Sequence[str], w: float) -> List[Line]:
    inner = w - 2 * PAD_X
    out = [Line(t, size=T_SIZE, weight="bold", color=C.INK)
           for t in wrap(title, T_SIZE, inner, weight="bold")]
    for d in desc:
        out += [Line(t, size=D_SIZE, color=C.INK2) for t in wrap(d, D_SIZE, inner)]
    return out


def main() -> None:
    C.apply_style()
    f = C.design_facts()
    tr, va, te = f["split"]
    a = C.idn(f["alpha"], 1)
    K = f["K"]

    ds_cards: List[Card] = []
    for d in f["datasets"]:
        x = f["data"][d]
        ds_cards.append((C.DATASET_LONG[d], [
            f"{C.idn(x['n_train'])} baris training",
            f"{x['n_features']} fitur · fraud {C.idn(100 * x['prevalence'], 2)}%"]))

    para_cards: List[Card] = []
    for p, ms in f["paradigms"]:
        para_cards.append((C.PARADIGM_LABEL[p],
                           [", ".join(C.MODEL_LABEL[m] for m in ms),
                            C.label(PARADIGM_NOTE, p, "aggregation")]))

    layers = [
        ("Lapisan 1 — Data dan preprocessing",
         f"Persiapan dan transformasi {NUM_WORD[len(f['datasets'])]} dataset",
         [ds_cards,
          [("Pembersihan data", ["hapus kolom identifier"]),
           ("Encoding dan StandardScaler", ["fit pada training set"]),
           ("Split stratified", [f"{tr} : {va} : {te}"])]]),
        ("Lapisan 2 — Partisi dan client orchestration",
         "Pembagian training set ke client federated",
         [[("Skema IID", ["distribusi seragam antar client"]),
           ("Skema Dirichlet Non-IID", [f"α = {a}"]),
           (f"Distribusi ke client (K = {K})",
            [f"Client 1 – Client {K}"])],
          [("SMOTE lokal per client",
            ["lengan SMOTE; penanganan class imbalance tanpa berbagi data mentah"])]]),
        ("Lapisan 3 — Pelatihan federated learning",
         f"{NUM_WORD[len(para_cards)].capitalize()} paradigma agregasi, "
         f"{NUM_WORD[len(f['fed_models'])]} model",
         [para_cards]),
        ("Lapisan 4 — Evaluasi",
         "Pengukuran performa dan analisis explainability",
         [[("Performa deteksi",
            ["AUPRC (utama), Recall@5%FPR",
             "F1, Precision, Recall; Brier dan kalibrasi"]),
           ("Explainability",
            ["SHAP per client",
             "Spearman, Kuncheva, uji exchangeability eksak"])]]),
    ]

    W = C.TEXT_WIDTH_IN
    LPAD, ARROW = 0.12, 0.30
    head = [Line("x", size=8.8, weight="bold"), Line("x", size=7.6)]
    head_h = sum(line_h(l.size) for l in head) + 0.10

    # measure every layer first, then draw
    plan = []
    for i, (title, sub, rows) in enumerate(layers):
        inner_w = W - 2 * LPAD
        laid = []
        for row in rows:
            cw = (inner_w - GAP * (len(row) - 1)) / len(row)
            ls = [card_lines(t, d, cw) for t, d in row]
            rh = max(block_height(l, PAD_Y) for l in ls)
            laid.append((cw, ls, rh))
        h = LPAD + head_h + sum(r[2] for r in laid) + GAP * (len(laid) - 1) + LPAD
        plan.append((title, sub, laid, h))
    H = sum(p[3] for p in plan) + ARROW * (len(plan) - 1) + 0.02

    cv = Canvas(W, H)
    y = 0.01
    for i, (title, sub, laid, h) in enumerate(plan):
        fill, edge, ink = C.TINTS[i]
        cv.rect(0.005, y, W - 0.01, h, fill=fill, edge=edge, lw=0.9, radius=0.08)
        cv.text(LPAD, y + LPAD, title, size=8.8, weight="bold", color=ink)
        cv.text(LPAD, y + LPAD + line_h(8.8), sub, size=7.6, color=ink)
        yy = y + LPAD + head_h
        for cw, ls, rh in laid:
            for j, l in enumerate(ls):
                x = LPAD + j * (cw + GAP)
                cv.rect(x, yy, cw, rh, fill="white", edge=edge, lw=0.7, radius=0.05)
                content = sum(line_h(q.size) for q in l)
                cv.lines(x + cw / 2, yy + (rh - content) / 2, l, ha="center")
            yy += rh + GAP
        y += h
        if i < len(plan) - 1:
            cv.arrow([(W / 2, y + 0.05), (W / 2, y + ARROW - 0.05)])
            y += ARROW

    C.save(cv.fig, "fig-3-2-logical-layers.png")


if __name__ == "__main__":
    main()
