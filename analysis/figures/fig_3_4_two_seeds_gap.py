#!/usr/bin/env python3
"""Gambar 3.4 <fig-3-two-seeds-gap> — Skema pengukuran dua-seed.

    python analysis/figures/fig_3_4_two_seeds_gap.py [dataset/model/condition_arm]

Writes buku/resources/fig-3-two-seeds-gap.png. Default cell:
baf/bert_fraud/dirichlet_none (the one the BAB 3 caption names).

Source: results/shap_v2/<cell>/importance_per_client_seed.csv — the K x 2 x M
importance matrix the RQ3 runner persisted (frozen tree, read-only). Everything
on the figure is recomputed from it here with the SAME functions the runner used
(evaluation/shap_inference.py):

  floor   = Spearman(client c @ seed 11, client c @ seed 22)          K values
  between = Spearman(client i @ seed s, client j @ seed s), i<j, s    K(K-1) values
  p       = exact exchangeability test over all (2K-1)!! perfect matchings

and then cross-checked, value by value, against the stored stability.json and
the cell's row in shap_summary_v2.csv. Any disagreement aborts: the figure can
only show numbers the frozen record also holds.

Panel A draws the real per-client importances of the cell's five highest-mean
features (fixed order), so the glyphs are data, not decoration.
"""

from __future__ import annotations

import sys

import numpy as np

import _common as C

sys.path.insert(0, str(C.PROJECT_ROOT))
from evaluation import shap_inference as SI  # noqa: E402

CELL = sys.argv[1] if len(sys.argv) > 1 else "baf/bert_fraud/dirichlet_none"
TOL = 1e-9


def load_cell():
    ds, model, cond_arm = CELL.split("/")
    cond, arm = cond_arm.rsplit("_", 1)
    cdir = C.SHAP_V2.joinpath(*CELL.split("/"))
    stab = C.read_json(cdir / "stability.json")
    seeds = [int(s) for s in stab.get("shap_seeds", [])]
    K = int(stab.get("n_clients", 0))
    if len(seeds) != 2 or K < 2:
        C.fail(f"{cdir}/stability.json: need 2 SHAP seeds and K >= 2, got "
               f"seeds={seeds}, K={K}")
    cols = [f"client_{c}_s{s}" for c in range(K) for s in seeds]
    imp = C.read_csv(cdir / "importance_per_client_seed.csv", ["feature", *cols])
    G = np.stack([np.stack([imp[f"client_{c}_s{s}"].to_numpy(float) for s in seeds])
                  for c in range(K)])                       # (K, 2, M)
    summ = C.read_csv(C.SHAP_SUMMARY, ["dataset", "model", "condition", "arm", "bg",
                                       "floor_mean", "between_mean", "p_value",
                                       "n_matchings"])
    row = summ[(summ.dataset == ds) & (summ.model == model) & (summ.condition == cond)
               & (summ.arm == arm) & (summ.bg == "local")]
    if len(row) != 1:
        C.fail(f"expected one local-background row for {CELL} in {C.SHAP_SUMMARY}, "
               f"found {len(row)}")
    return imp["feature"].tolist(), G, seeds, stab, row.iloc[0]


def check(name, ours, stored, tol=TOL):
    ours, stored = np.asarray(ours, float), np.asarray(stored, float)
    if ours.shape != stored.shape or not np.allclose(ours, stored, rtol=0, atol=tol):
        C.fail(f"{CELL}: recomputed {name} does not match the stored record\n"
               f"  recomputed: {ours}\n  stored:     {stored}")


def main() -> None:
    C.apply_style()
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    feats, G, seeds, stab, row = load_cell()
    K = G.shape[0]
    stats = SI.cell_inference(G)
    if stats.get("status") != "ok":
        C.fail(f"{CELL}: cell is {stats.get('status')} ({stats.get('reason')}); "
               "no stability numbers to draw.")
    F, B = np.array(stats["floor"]), np.array(stats["between"])
    p, nm = float(stats["p_value"]), int(stats["n_matchings"])

    # -- cross-checks against the frozen record ---------------------------------
    check("floor", F, stab["floor"])
    check("between", B, stab["between"])
    check("p_value", p, stab["p_value"])
    check("floor_mean (summary csv, 4 dp)", round(F.mean(), 4), float(row.floor_mean), 5e-5)
    check("between_mean (summary csv, 4 dp)", round(B.mean(), 4), float(row.between_mean), 5e-5)
    check("p_value (summary csv, 4 dp)", round(p, 4), float(row.p_value), 5e-5)
    if nm != int(stab["n_matchings"]) or nm != int(row.n_matchings):
        C.fail(f"{CELL}: n_matchings {nm} vs stored {stab['n_matchings']}/{row.n_matchings}")
    if len(B) != K * (K - 1):
        C.fail(f"{CELL}: expected {K * (K - 1)} between values, got {len(B)}")

    # top-5 features by overall mean |importance| (the CSV's `mean` column rule)
    mean_imp = np.abs(G).mean(axis=(0, 1))
    top = np.argsort(-mean_imp, kind="stable")[:5]
    gmax = G[:, :, top].max()

    W = C.TEXT_WIDTH_IN
    HF = 5.4
    fig = plt.figure(figsize=(W, HF))
    axA = fig.add_axes([0.0, 0.48, 1.0, 0.52])
    axB = fig.add_axes([0.26, 0.165, 0.70, 0.20])
    axA.set_xlim(0, W)
    axA.set_ylim(HF * 0.52, 0)
    axA.axis("off")

    # ---------------------------------------------------------------- panel A
    axA.text(0.02, 0.10, "A  Apa yang dibandingkan", fontsize=9.5,
             fontweight="bold", va="top")
    axA.text(0.02, 0.30, f"{K} client × 2 seed koalisi = {2 * K} vektor importance "
             f"({C.DATASET_LABEL[CELL.split('/')[0]]}, lima fitur teratas)",
             fontsize=7.8, color=C.INK2, va="top")
    gx0, gx1 = 2.15, 3.95                  # left / right glyph x
    bw, gap_b, gh = 0.085, 0.022, 0.26
    row_h, y0 = 0.34, 0.80
    glyph_w = 5 * bw + 4 * gap_b
    mid = (gx0 + glyph_w + gx1) / 2
    for sd, gx in enumerate((gx0, gx1)):
        axA.text(gx + glyph_w / 2, y0 - 0.12, f"SEED {seeds[sd]}", ha="center",
                 fontsize=7.2, color=C.MUTED, fontweight="bold")
    axA.text(mid, y0 - 0.12, "floor (ρ)", ha="center", fontsize=7.4,
             color=C.FLOOR_COLOR, fontweight="bold")
    for c in range(K):
        yb = y0 + c * row_h + gh                  # bar baseline
        axA.text(gx0 - 0.15, yb - gh / 2, f"client {c}", ha="right", va="center",
                 fontsize=8, color=C.INK2)
        for sd, gx in enumerate((gx0, gx1)):
            for b, j in enumerate(top):
                h = max(0.012, G[c, sd, j] / gmax * gh)
                axA.add_patch(plt.Rectangle((gx + b * (bw + gap_b), yb - h), bw, h,
                                            color=C.BETWEEN_COLOR,
                                            alpha=1.0 if sd == 0 else 0.55, lw=0))
        yl = yb - gh / 2
        axA.plot([gx0 + glyph_w + 0.05, gx1 - 0.05], [yl, yl], ls=(0, (3, 3)),
                 color=C.FLOOR_COLOR, lw=1)
        axA.add_patch(Circle((mid, yl), 0.075, facecolor="white",
                             edgecolor=C.FLOOR_COLOR, lw=1.2, zorder=3))
        axA.text(gx1 + glyph_w + 0.12, yl, f"ρ = {C.idn(F[c], 4)}", va="center",
                 fontsize=7.4, color=C.FLOOR_COLOR)
    # bracket for the between-client comparison
    bx = 0.95
    ytop, ybot = y0 + 0.02, y0 + (K - 1) * row_h + gh - 0.02
    axA.plot([bx + 0.08, bx, bx, bx + 0.08], [ytop, ytop, ybot, ybot],
             color=C.BETWEEN_COLOR, lw=1.2)
    axA.text(bx - 0.08, (ytop + ybot) / 2 - 0.07, "antar-client", ha="right",
             va="center", fontsize=8, color=C.INK2)
    axA.text(bx - 0.08, (ytop + ybot) / 2 + 0.10, "seed sama (CRN)", ha="right",
             va="center", fontsize=7.2, color=C.MUTED)
    fn = ", ".join(feats[j] for j in top)
    from _diagram import wrap
    note = (f"Batang, kiri ke kanan: {fn}. Tinggi batang = rerata |SHAP| client "
            "pada seed tersebut, relatif terhadap batang tertinggi.")
    axA.text(0.02, y0 + K * row_h - 0.02, "\n".join(wrap(note, 6.8, W - 0.1)),
             fontsize=6.8, color=C.MUTED, style="italic", va="top", linespacing=1.3)

    # ---------------------------------------------------------------- panel B
    fig.text(0.004, 0.455, "B  Sebaran kesepakatan pada satu sel", fontsize=9.5,
             fontweight="bold", va="top")
    fig.text(0.004, 0.418, CELL, fontsize=7.8, color=C.INK2, va="top",
             family="monospace")
    rng_lo = min(B.min(), F.min())
    lo = np.floor((rng_lo - 0.005) * 100) / 100
    axB.set_xlim(lo, 1.004)
    axB.set_ylim(-0.6, 1.6)
    axB.axvspan(F.min(), F.max() if F.max() > F.min() else F.min() + 1e-4,
                color=C.FLOOR_COLOR, alpha=0.10, lw=0)
    axB.scatter(F, np.ones_like(F), s=34, color=C.FLOOR_COLOR, zorder=3, lw=0)
    axB.scatter(B, np.zeros_like(B), s=30, color=C.BETWEEN_COLOR, alpha=0.85,
                zorder=3, lw=0)
    mf, mb = F.mean(), B.mean()
    axB.annotate("", xy=(mf, 0.5), xytext=(mb, 0.5),
                 arrowprops=dict(arrowstyle="-|>", color=C.INK2, lw=1,
                                 mutation_scale=9, shrinkA=0, shrinkB=0))
    axB.text((mf + mb) / 2, 0.62, f"Δ = {C.idn(mf - mb, 3)}", ha="center",
             va="bottom", fontsize=8.2, fontweight="bold", color=C.INK2)
    axB.text(F.min() - 0.002, 1.35, f"{K} titik, {C.idn(F.min(), 4)}–{C.idn(F.max(), 4)}",
             ha="right", va="center", fontsize=7, color=C.MUTED)
    axB.set_yticks([1, 0])
    axB.set_yticklabels(["floor\nclient sama, 2 seed",
                         "antar-client\nseed sama (CRN)"], fontsize=7.8)
    axB.tick_params(axis="y", length=0)
    axB.spines["left"].set_visible(False)
    axB.xaxis.set_major_formatter(C.idn_formatter(2))
    axB.set_xlabel("Spearman ρ", fontsize=8)

    fig.text(0.004, 0.035,
             "Di bawah $H_0$ kedua sebaran berimpit, sehingga “pada atau di bawah "
             "floor” bukan kriteria deteksi.\nUji eksak menyusun ulang "
             f"ke-{2 * K} vektor atas seluruh {nm} perfect matching: "
             f"p = {C.idn(p, 6)}.", fontsize=7.6, color=C.INK2, va="bottom",
             linespacing=1.4)

    C.save(fig, "fig-3-two-seeds-gap.png")


if __name__ == "__main__":
    main()
