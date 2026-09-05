#!/usr/bin/env python3
"""Method figure for the two-seed floor -> self-contained HTML.

    python analysis/two_seeds_gap.py [out.html] [dataset/model/condition_arm]

Shows what the RQ3 measurement actually compares. KernelSHAP draws coalitions, so
a client re-explained under a second coalition seed does not reproduce itself
exactly; that same-client, cross-seed agreement is the floor. Under H0 ("all
clients share one true importance vector") the 2K vectors are exchangeable, so
the expected between-client agreement EQUALS the floor -- which is why "at or
below the floor" cannot be a detection rule, and why the null is enumerated over
all (2K-1)!! perfect matchings instead.

Panels are stacked, not side by side: the figure is printed one text-column wide
in the thesis, so a wide layout would shrink the labels below legibility.

Reads one cell's stability.json (floor[] and between[] are already stored there);
falls back to the BAF/BERT/dirichlet_none values if the tree is absent.
"""
import json, os, sys

OUT = sys.argv[1] if len(sys.argv) > 1 else "two_seeds_gap.html"
CELL = sys.argv[2] if len(sys.argv) > 2 else "baf/bert_fraud/dirichlet_none"
SEEDS = (11, 22)

FALLBACK = {
    "floor": [0.9996, 0.9998, 0.9997, 0.9999, 0.9997],
    "between": [0.9517, 0.9646, 0.9707, 0.9625, 0.9274, 0.9790, 0.9545, 0.9383,
                0.9427, 0.9574, 0.9555, 0.9650, 0.9725, 0.9627, 0.9289, 0.9793,
                0.9563, 0.9384, 0.9412, 0.9562],
    "p_value": 0.0010582010582010583, "n_matchings": 945, "n_clients": 5,
}


def load():
    p = os.path.join("results", "shap_v2", *CELL.split("/"), "stability.json")
    if os.path.exists(p):
        s = json.load(open(p))
        if s.get("floor") and s.get("between"):
            return s, p
    return FALLBACK, "embedded snapshot"


s, src = load()
floor, between = list(s["floor"]), list(s["between"])
K, p, nm = int(s["n_clients"]), float(s["p_value"]), int(s["n_matchings"])

W = 820
# ---------------------------------------------------------------- panel A
# K clients x 2 seeds; each importance vector drawn as a 5-bar glyph. The bar
# heights are decorative -- the panel is about WHICH pairs are compared.
BARS = [[.95, .62, .41, .28, .17], [.93, .65, .39, .30, .15]]
JIT = [0, .02, -.03, .04, -.02]
gx, gy, gh, rowh, bw, gap2 = 300, 92, 40, 52, 11, 168
glyphs, links = [], []
for c in range(K):
    y = gy + c * rowh
    for sd in (0, 1):
        x = gx + sd * gap2
        for b in range(5):
            h = max(3.0, (BARS[sd][b] + JIT[c]) * gh)
            glyphs.append(f'<rect x="{x + b * bw:.0f}" y="{y + gh - h:.1f}" width="{bw - 3}" '
                          f'height="{h:.1f}" rx="1.5" class="bar s{sd}"/>')
    links.append(f'<path d="M{gx + 5 * bw + 2} {y + gh / 2} H{gx + gap2 - 4}" class="within"/>')
    links.append(f'<circle cx="{gx + (5 * bw + gap2) / 2:.0f}" cy="{y + gh / 2}" r="11" class="wdot"/>')
    glyphs.append(f'<text x="{gx - 18}" y="{y + gh / 2 + 5}" class="rowlab">client {c}</text>')
if K:
    links.append(f'<text x="{gx + (5 * bw + gap2) / 2:.0f}" y="{gy - 14}" class="wlab">floor</text>')
bx = gx - 108
btw = (f'<path d="M{bx + 10} {gy + 4} H{bx} V{gy + (K - 1) * rowh + gh - 4} H{bx + 10}" class="btw"/>'
       f'<text x="{bx - 8}" y="{gy + (K - 1) * rowh / 2 + gh / 2 - 6}" class="brk">antar-client</text>'
       f'<text x="{bx - 8}" y="{gy + (K - 1) * rowh / 2 + gh / 2 + 12}" class="brknote">seed sama (CRN)</text>')
AY = gy + (K - 1) * rowh + gh + 40          # panel A bottom

# ---------------------------------------------------------------- panel B
AX0, AX1 = 250, W - 40
LO = min(min(between), min(floor)) - 0.012
HI = 1.0 + 0.004
BY = AY + 74                                 # panel B heading baseline
FL, BL, AXY = BY + 80, BY + 150, BY + 190   # floor lane, between lane, axis


def X(v):
    return AX0 + (v - LO) / (HI - LO) * (AX1 - AX0)


ticks = [t / 100 for t in range(int(LO * 100) + 1, 101, 2)]
tick_svg = ''.join(
    f'<line x1="{X(t):.1f}" y1="{AXY}" x2="{X(t):.1f}" y2="{AXY + 7}" class="tick"/>'
    f'<text x="{X(t):.1f}" y="{AXY + 24}" class="ticklab">{t:.2f}</text>' for t in ticks)
band = (f'<rect x="{X(min(floor)):.1f}" y="{FL - 26}" '
        f'width="{max(2.5, X(max(floor)) - X(min(floor))):.1f}" height="{BL - FL + 52}" class="band"/>')
fdots = ''.join(f'<circle cx="{X(v):.1f}" cy="{FL}" r="7" class="fdot"/>' for v in floor)
bdots = ''.join(f'<circle cx="{X(v):.1f}" cy="{BL}" r="7" class="bdot"/>' for v in between)
mb, mf = sum(between) / len(between), sum(floor) / len(floor)
MY = (FL + BL) / 2
gap = (f'<path d="M{X(mb):.1f} {MY} H{X(mf):.1f}" class="gap"/>'
       f'<path d="M{X(mf) - 9:.1f} {MY - 5} L{X(mf):.1f} {MY} L{X(mf) - 9:.1f} {MY + 5}" class="gap"/>'
       f'<text x="{(X(mb) + X(mf)) / 2:.1f}" y="{MY - 9}" class="gaplab">'
       f'&#916; = {mf - mb:.3f}</text>')
H = AXY + 100

doc = f"""<!doctype html><meta charset="utf-8">
<title>Two-seed floor vs between-client gap</title>
<style>
:root{{color-scheme:light;--surface:#fcfcfb;--plane:#f9f9f7;--ink:#0b0b0b;
 --ink2:#52514e;--muted:#898781;--rule:#e1e0d9;--ring:rgba(11,11,11,.10);
 --floor:#0d366b;--btwn:#3987e5;--band:rgba(13,54,107,.10)}}
@media (prefers-color-scheme:dark){{:root:not([data-theme=light]){{
 --surface:#1a1a19;--plane:#0d0d0d;--ink:#fff;--ink2:#c3c2b7;--muted:#898781;
 --rule:#2c2c2a;--ring:rgba(255,255,255,.10);--floor:#9ec5f4;--btwn:#3987e5;
 --band:rgba(158,197,244,.12)}}}}
[data-theme=dark]{{--surface:#1a1a19;--plane:#0d0d0d;--ink:#fff;--ink2:#c3c2b7;
 --muted:#898781;--rule:#2c2c2a;--ring:rgba(255,255,255,.10);--floor:#9ec5f4;
 --btwn:#3987e5;--band:rgba(158,197,244,.12)}}
*{{box-sizing:border-box}}
body{{margin:0;padding:26px 22px;background:var(--plane);color:var(--ink);
 font:14px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}}
.card{{max-width:760px;margin:0 auto;background:var(--surface);
 border:1px solid var(--ring);border-radius:12px;padding:14px 16px 10px}}
svg{{display:block;width:100%;height:auto;font-family:inherit}}
.bar{{fill:var(--btwn)}} .bar.s1{{fill:var(--btwn);opacity:.55}}
.within{{stroke:var(--floor);stroke-width:2;stroke-dasharray:4 4;fill:none}}
.wdot{{fill:var(--surface);stroke:var(--floor);stroke-width:2}}
.btw{{stroke:var(--btwn);stroke-width:2;fill:none;opacity:.8}}
.rowlab{{fill:var(--ink2);font-size:16px;text-anchor:end}}
.collab{{fill:var(--muted);font-size:14px;text-anchor:middle;
 letter-spacing:.05em;text-transform:uppercase}}
.h{{fill:var(--ink);font-size:19px;font-weight:600}}
.sub{{fill:var(--ink2);font-size:15px}}
.axis{{stroke:var(--rule);stroke-width:1.2}}
.tick{{stroke:var(--rule);stroke-width:1.2}}
.ticklab{{fill:var(--muted);font-size:14px;text-anchor:middle;
 font-variant-numeric:tabular-nums}}
.band{{fill:var(--band)}}
.fdot{{fill:var(--floor)}} .bdot{{fill:var(--btwn);opacity:.85}}
.gap{{stroke:var(--ink2);stroke-width:1.8;fill:none}}
.gaplab{{fill:var(--ink2);font-size:16px;text-anchor:middle;font-weight:600}}
.lane{{fill:var(--ink2);font-size:16px;text-anchor:end}}
.note{{fill:var(--muted);font-size:13.5px}}
.eq{{fill:var(--ink2);font-size:14.5px}}
.brk{{fill:var(--ink2);font-size:15px;text-anchor:end}}
.brknote{{fill:var(--muted);font-size:13px;text-anchor:end}}
.wlab{{fill:var(--floor);font-size:14px;text-anchor:middle}}
</style>
<div class="card">
<svg viewBox="0 0 {W} {H}" role="img"
     aria-label="Two coalition seeds per client: the within-client floor and the
     between-client comparison it is tested against.">
  <text x="18" y="30" class="h">A &nbsp;Apa yang dibandingkan</text>
  <text x="18" y="54" class="sub">{K} client &#215; 2 seed koalisi = {2 * K} vektor</text>
  <text x="{gx + 5 * bw / 2:.0f}" y="{gy - 14}" class="collab">seed {SEEDS[0]}</text>
  <text x="{gx + gap2 + 5 * bw / 2:.0f}" y="{gy - 14}" class="collab">seed {SEEDS[1]}</text>
  {''.join(links)}{btw}{''.join(glyphs)}
  <line x1="18" y1="{AY}" x2="{W - 18}" y2="{AY}" class="axis"/>
  <text x="18" y="{BY}" class="h">B &nbsp;Sebaran kesepakatan pada satu sel</text>
  <text x="18" y="{BY + 24}" class="sub">{CELL}</text>
  {band}{gap}{bdots}{fdots}
  <line x1="{AX0}" y1="{AXY}" x2="{AX1}" y2="{AXY}" class="axis"/>{tick_svg}
  <text x="{AX0 - 16}" y="{FL - 4}" class="lane">floor</text>
  <text x="{AX0 - 16}" y="{FL + 16}" class="note" text-anchor="end">client sama, 2 seed</text>
  <text x="{AX0 - 16}" y="{BL - 4}" class="lane">antar-client</text>
  <text x="{AX0 - 16}" y="{BL + 16}" class="note" text-anchor="end">seed sama (CRN)</text>
  <text x="{X(min(floor)) - 16:.1f}" y="{FL - 20}" class="note" text-anchor="end">{K} titik berimpit, {min(floor):.4f}&#8211;{max(floor):.4f}</text>
  <text x="18" y="{AXY + 58}" class="eq">Di bawah H&#8320; kedua sebaran berimpit, sehingga
  &#34;pada atau di bawah floor&#34; bukan kriteria deteksi.</text>
  <text x="18" y="{AXY + 80}" class="eq">Uji eksak menyusun ulang ke-{2 * K} vektor atas
  seluruh {nm} perfect matching: p = {p:.4g}.</text>
</svg>
</div>
"""
open(OUT, "w").write(doc)
print(f"wrote {OUT}  (cell {CELL}, source: {src})")
