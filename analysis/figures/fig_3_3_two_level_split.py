#!/usr/bin/env python3
"""Gambar 3.3 <fig-3-3> — Skema pembagian data dua tingkat.

    python analysis/figures/fig_3_3_two_level_split.py

Writes buku/resources/fig-3-3-two-level-split.png.

Data-driven content:
  datasets, features, fraud share, training rows .. results/analysis/minority_census.csv
                                                    (seed-42 IID clients, summed)
  per-client fraud counts under Non-IID ........... results/analysis/minority_census.csv
                                                    (seed 42, Dirichlet alpha of record)
  K, alpha, seed, threshold policy ................ results/clean_summary.csv
  split percentages ............................... preprocessing/*.py test_size
The roles of the validation and test sets are worded after BAB 3 §3.4.4/§3.4.5.
"""

from __future__ import annotations

from typing import List

import _common as C
from _diagram import Canvas, Line, block_height, line_h, wrap

PAD = 0.07


def lines_for(title: str, body: List[str], w: float, tsize=8.0, bsize=7.3,
              bullet=False) -> List[Line]:
    inner = w - 2 * PAD
    out = [Line(t, size=tsize, weight="bold", color=C.INK)
           for t in wrap(title, tsize, inner, weight="bold")]
    for b in body:
        parts = (wrap(b, bsize, inner, first="• ", cont="   ") if bullet
                 else wrap(b, bsize, inner))
        out += [Line(p, size=bsize, color=C.INK2) for p in parts]
    return out


def main() -> None:
    C.apply_style()
    f = C.design_facts()
    tr, va, te = f["split"]
    K, a, seed = f["K"], C.idn(f["alpha"], 1), f["seed"]
    lab = C.DATASET_LABEL
    if f["threshold_policy"] != "val_f1_tuned":
        C.fail(f"threshold policy is {f['threshold_policy']!r}; the validation-set "
               "card describes F1-tuned thresholds — update its wording first.")

    W = C.TEXT_WIDTH_IN
    cv_items = []  # (x, y, w, lines, fill, edge) collected, drawn after sizing

    # ---------------------------------------------------------------- level 1
    y = 0.01
    lv1 = lines_for(f"TINGKAT 1 — PEMBAGIAN GLOBAL",
                    [f"stratified, seed {seed}"], 2.6, tsize=8.2)
    h_lv1 = block_height(lv1, PAD)
    ds_body = [f"{lab[d]}: {f['data'][d]['n_features']} fitur, fraud "
               f"{C.idn(100 * f['data'][d]['prevalence'], 2)}%" for d in f["datasets"]]
    ds = lines_for("Dataset", ds_body, 3.0)
    h_ds = block_height(ds, PAD)

    colw, colgap = 1.98, (W - 3 * 1.98) / 2
    xs = [0.0, colw + colgap, 2 * (colw + colgap)]
    val = lines_for(f"Validation set ({va}%)",
                    ["server simulasi", "pemilihan model dan putaran terbaik (AUPRC)",
                     "seleksi prefix dan kandidat GBM",
                     "ambang F1 tertinggi untuk F1, Precision, Recall"],
                    colw, bullet=True)
    train = lines_for(f"Training set ({tr}%)",
                      [f"{lab[d]}: {C.idn(f['data'][d]['n_train'])} baris"
                       for d in f["datasets"]], colw, bullet=True)
    test = lines_for(f"Test set ({te}%)",
                     ["server simulasi", "evaluasi akhir seluruh model",
                      "AUPRC, Recall@5%FPR, F1, Precision, Recall",
                      "tidak diakses selama pelatihan"], colw, bullet=True)
    h_row = max(block_height(x, PAD) for x in (val, train, test))

    lv2 = lines_for("TINGKAT 2 — PARTISI CLIENT",
                    [f"IID atau Dirichlet α = {a}, K = {K}"], 2.6, tsize=8.2)
    h_lv2 = block_height(lv2, PAD)

    cgap = 0.10
    cw = (W - cgap * (K - 1)) / K
    clients = []
    for k in range(K):
        body = [f"{lab[d]}: {C.idn(f['clients'][d]['n_minority'][k])}"
                for d in f["datasets"]]
        clients.append(lines_for(f"D{k + 1}", body, cw, tsize=8.0, bsize=7.3))
    h_cl = max(block_height(c, PAD) for c in clients)
    note = (f"Angka pada D1–D{K}: jumlah sampel fraud per client pada partisi "
            f"Non-IID (α = {a}, seed {seed}); skema IID membagi setiap kelas merata. "
            f"SMOTE lokal diterapkan per client pada lengan SMOTE.")
    note_lines = [Line(t, size=6.9, color=C.MUTED, style="italic")
                  for t in wrap(note, 6.9, W - 0.04)]
    local = lines_for("Pelatihan model lokal di setiap client", [], 3.0, tsize=8.0)
    h_local = block_height(local, PAD)

    GV = 0.30
    Y_LV1 = y
    Y_DS = Y_LV1 + h_lv1 + 0.18
    Y_ROW = Y_DS + h_ds + 0.42
    Y_LV2 = Y_ROW + h_row + GV
    Y_CL = Y_LV2 + h_lv2 + 0.40
    h_note = sum(line_h(l.size) for l in note_lines)
    Y_LOC = Y_CL + h_cl + 0.36
    Y_NOTE = Y_LOC + h_local + 0.10
    H = Y_NOTE + h_note + 0.02

    cv = Canvas(W, H)
    blue, green, slate = C.TINTS[0], C.TINTS[1], C.TINTS[5]

    def box(x, yy, w, ls, tint, h=None, ha="center"):
        hh = h if h is not None else block_height(ls, PAD)
        cv.rect(x, yy, w, hh, fill=tint[0], edge=tint[1], lw=0.8, radius=0.05)
        content = sum(line_h(l.size) for l in ls)
        if ha == "center":      # centred cards: centre the block vertically
            cv.lines(x + w / 2, yy + (hh - content) / 2, ls, ha=ha)
        else:                   # bulleted cards in a row: top-align
            cv.lines(x + PAD, yy + PAD, ls, ha=ha)
        return hh

    cx = W / 2
    box(cx - 1.3, Y_LV1, 2.6, lv1, slate)
    cv.arrow([(cx, Y_LV1 + h_lv1), (cx, Y_DS)])
    box(cx - 1.5, Y_DS, 3.0, ds, ("white", C.RULE, C.INK))

    yj = Y_DS + h_ds + 0.20
    cv.ax.plot([cx, cx], [Y_DS + h_ds, yj], color=C.INK2, lw=0.9, zorder=2)
    cv.ax.plot([xs[0] + colw / 2, xs[2] + colw / 2], [yj, yj], color=C.INK2,
               lw=0.9, zorder=2)
    for x in xs:
        cv.arrow([(x + colw / 2, yj), (x + colw / 2, Y_ROW)])
    cv.text(cx + 0.06, Y_DS + h_ds + 0.035, f"split stratified {tr} : {va} : {te}",
            size=7.0, color=C.MUTED, style="italic")

    box(xs[0], Y_ROW, colw, val, ("white", C.RULE, C.INK), h=h_row, ha="left")
    box(xs[1], Y_ROW, colw, train, blue, h=h_row, ha="left")
    box(xs[2], Y_ROW, colw, test, ("white", C.RULE, C.INK), h=h_row, ha="left")

    cv.arrow([(cx, Y_ROW + h_row), (cx, Y_LV2)])
    box(cx - 1.3, Y_LV2, 2.6, lv2, slate)

    yj2 = Y_LV2 + h_lv2 + 0.18
    cv.ax.plot([cx, cx], [Y_LV2 + h_lv2, yj2], color=C.INK2, lw=0.9, zorder=2)
    cv.ax.plot([cw / 2, W - cw / 2], [yj2, yj2], color=C.INK2, lw=0.9, zorder=2)
    for k in range(K):
        x = k * (cw + cgap)
        cv.arrow([(x + cw / 2, yj2), (x + cw / 2, Y_CL)])
        box(x, Y_CL, cw, clients[k], green, h=h_cl)
    cv.lines(0.02, Y_NOTE, note_lines)

    yj3 = Y_LOC - 0.16
    for k in range(K):
        x = k * (cw + cgap) + cw / 2
        cv.ax.plot([x, x], [Y_CL + h_cl, yj3], color=C.INK2, lw=0.9, zorder=2)
    cv.ax.plot([cw / 2, W - cw / 2], [yj3, yj3], color=C.INK2, lw=0.9, zorder=2)
    cv.arrow([(cx, yj3), (cx, Y_LOC)])
    box(cx - 1.5, Y_LOC, 3.0, local, slate)

    C.save(cv.fig, "fig-3-3-two-level-split.png")


if __name__ == "__main__":
    main()
