#!/usr/bin/env python3
"""RQ3 cross-client stability heatmap -> self-contained HTML.

Run on the box:   python make_rq3_heatmap.py
Reads results/shap_v2/**/stability.json when present; otherwise falls back to the
embedded kernel-tier snapshot so the layout can be previewed without the data.
"""
import glob, json, os, sys, html

OUT = sys.argv[1] if len(sys.argv) > 1 else "rq3_stability.html"
DS = {"baf": "BAF", "creditcard": "ULB", "paysim": "PaySim"}
MODEL_ORDER = ["lr", "svm", "gbm", "ffd", "bert_fraud", "fedxgbllr"]
MODEL_LABEL = {"lr": "LR", "svm": "SVM", "gbm": "GBM", "ffd": "FFD",
               "bert_fraud": "BERT", "fedxgbllr": "FedXGBllr"}
TIER = {"lr": "exact", "svm": "exact", "gbm": "exact",
        "ffd": "sampled", "bert_fraud": "sampled", "fedxgbllr": "sampled"}
COLS = [("baf","dirichlet","none"),("baf","dirichlet","smote"),("baf","iid","none"),
        ("creditcard","dirichlet","none"),("creditcard","dirichlet","smote"),
        ("creditcard","iid","none"),("creditcard","iid","smote"),
        ("paysim","dirichlet","none"),("paysim","dirichlet","smote"),
        ("paysim","iid","none"),("paysim","iid","smote")]

# sequential blue, steps 700 -> 100 : low stability (dark) -> high stability (light)
RAMP = ["#0d366b","#104281","#184f95","#1c5cab","#256abf","#2a78d6","#3987e5",
        "#5598e7","#6da7ec","#86b6ef","#9ec5f4","#b7d3f6","#cde2fb"]
LO, HI = 0.65, 1.00

FALLBACK = [  # kernel tier, bg=local, weighted between-client + exchangeability p
 ("creditcard","fedxgbllr","iid","none",0.9524,0.8772),
 ("creditcard","fedxgbllr","dirichlet","none",0.9821,0.6974),
 ("paysim","fedxgbllr","iid","none",0.9859,0.8381),
 ("creditcard","bert_fraud","iid","none",0.9949,0.3577),
 ("creditcard","ffd","iid","smote",0.9924,0.001058),
 ("paysim","ffd","iid","none",0.9732,0.001058),
 ("creditcard","ffd","dirichlet","none",0.9877,0.001058),
 ("creditcard","ffd","iid","none",0.9825,0.001058),
 ("creditcard","ffd","dirichlet","smote",0.9846,0.001058),
 ("baf","bert_fraud","iid","none",0.9435,0.001058),
 ("paysim","ffd","iid","smote",0.9307,0.002116),
 ("baf","fedxgbllr","dirichlet","smote",0.9679,0.001058),
 ("creditcard","bert_fraud","dirichlet","none",0.9801,0.001058),
 ("baf","ffd","dirichlet","none",0.9634,0.001058),
 ("baf","ffd","dirichlet","smote",0.9481,0.001058),
 ("paysim","bert_fraud","dirichlet","smote",0.9517,0.009524),
 ("baf","ffd","iid","none",0.9633,0.001058),
 ("baf","fedxgbllr","dirichlet","none",0.9723,0.001058),
 ("creditcard","bert_fraud","iid","smote",0.9583,0.001058),
 ("baf","fedxgbllr","iid","none",0.9650,0.001058),
 ("paysim","fedxgbllr","iid","smote",0.9482,0.001058),
 ("baf","bert_fraud","dirichlet","smote",0.8894,0.001058),
 ("creditcard","bert_fraud","dirichlet","smote",0.9475,0.001058),
 ("baf","bert_fraud","dirichlet","none",0.8682,0.001058),
 ("paysim","ffd","dirichlet","smote",0.8859,0.009524),
 ("creditcard","fedxgbllr","dirichlet","smote",0.9417,0.001058),
 ("paysim","bert_fraud","iid","none",0.9825,0.001058),
 ("paysim","bert_fraud","dirichlet","none",0.9410,0.002116),
 ("paysim","ffd","dirichlet","none",0.7979,0.001058),
 ("creditcard","fedxgbllr","iid","smote",0.9576,0.001058),
 ("paysim","bert_fraud","iid","smote",0.9722,0.001058),
 ("paysim","fedxgbllr","dirichlet","none",0.8216,0.002116),
 ("paysim","fedxgbllr","dirichlet","smote",0.6650,0.001058),
]


def load():
    """One value per (model, dataset, condition, arm), exact tier wins.

    results/shap_v2/ holds two runs for the PaySim kernel cells: the sampled main
    stage (nsamples = 500) and the grouped exact stage (M_eff = 9, nsamples = 510,
    zero estimator noise). Both carry bg="local", so a plain glob collides on the
    key and the surviving value depends on filesystem order. Resolve it in favour
    of the exact one — a measured attribution beats an estimate of it, and the
    sampled-vs-exact gap reaches 0.099 — and remember which cells that was so the
    figure can mark them.
    """
    cells, src = {}, "embedded snapshot (kernel tier only)"
    files = sorted(glob.glob("results/shap_v2/**/stability.json", recursive=True))
    if files:
        src, n = f"results/shap_v2/ ({len(files)} files)", 0
        for f in files:
            s = json.load(open(f))
            if s.get("n_clients", 1) < 2 or s.get("bg", "local") != "local":
                continue
            key = (s["model"], s["dataset"], s["condition"], s["arm"])
            if s.get("status") != "ok" or s.get("weighted_between") is None:
                cells.setdefault(key, None)
                continue
            det = bool(s.get("deterministic"))
            prev = cells.get(key)
            if prev and prev[3] and not det:
                continue                      # keep the exact measurement
            # deterministic cells carry no exchangeability test: p is null there,
            # and NaN keeps them out of the "not distinguishable" hatching below.
            pv = s.get("p_value")
            cells[key] = (float(s["weighted_between"]),
                          float(pv) if pv is not None else float("nan"),
                          float(s.get("weighted_within") or 0), det)
            n += 1
        if n:
            return cells, src
    for d, m, c, a, b, p in FALLBACK:
        cells[(m, d, c, a)] = (b, p, None, False)
    return cells, src


def color(v):
    t = min(max((v - LO) / (HI - LO), 0.0), 1.0)
    return RAMP[min(int(t * len(RAMP)), len(RAMP) - 1)]


def ink(v):                      # dark fills need light text
    return "#ffffff" if (v - LO) / (HI - LO) < 0.46 else "#0b0b0b"


cells, src = load()
present = [m for m in MODEL_ORDER if any((m, *c) in cells for c in COLS)]
vals = [v[0] for v in cells.values() if v]
n_ns = sum(1 for v in cells.values() if v and v[1] >= 0.05)

rows, tbody = [], []
for m in present:
    tds = [f'<th scope="row"><span class="m">{MODEL_LABEL[m]}</span>'
           f'<span class="t {TIER[m]}">{"exact" if TIER[m]=="exact" else "sampled"}</span></th>']
    for ds, cond, arm in COLS:
        e = cells.get((m, ds, cond, arm))
        lbl = f"{DS[ds]} / {cond} / {arm}"
        if e is None:
            tds.append(f'<td class="na" title="{lbl} — not run / undefined"><span>·</span></td>')
            continue
        b, p, w, det = e
        ns = p >= 0.05
        promoted = det and TIER[m] == "sampled"   # kernel cell measured exactly
        tip = (f"{MODEL_LABEL[m]} — {lbl}&#10;"
               f"between-client {b:.3f}&#10;"
               + ("no sampling floor — exact explainer" if p != p else f"p = {p:.4g}")
               + ("&#10;exact tier (grouped, nsamples = 510)" if promoted else "")
               + ("&#10;NOT distinguishable from estimator noise" if ns else ""))
        tds.append(
            f'<td class="c{" ns" if ns else ""}" style="--f:{color(b)};--i:{ink(b)}" '
            f'title="{tip}"><span>{b:.3f}</span>'
            + ('<u></u>' if promoted else '')
            + ('<em>ns</em>' if ns else '') + '</td>')
        tbody.append(f"<tr><td>{MODEL_LABEL[m]}</td><td>{DS[ds]}</td><td>{cond}</td>"
                     f"<td>{arm}</td><td>{b:.4f}</td>"
                     f"<td>{'—' if p != p else format(p, '.4g')}</td>"
                     f"<td>{'no' if ns else 'yes'}</td></tr>")
    rows.append("<tr>" + "".join(tds) + "</tr>")

head = "".join(f'<th><span>{DS[d]}</span><b>{c}</b><i>{a}</i></th>' for d, c, a in COLS)
legend = "".join(f'<i style="background:{c}"></i>' for c in RAMP)

doc = f"""<!doctype html><meta charset="utf-8">
<title>RQ3 — cross-client SHAP stability</title>
<style>
:root{{color-scheme:light;--surface:#fcfcfb;--plane:#f9f9f7;--ink:#0b0b0b;
 --ink2:#52514e;--muted:#898781;--rule:#e1e0d9;--ring:rgba(11,11,11,.10)}}
@media (prefers-color-scheme:dark){{:root:not([data-theme=light]){{
 --surface:#1a1a19;--plane:#0d0d0d;--ink:#fff;--ink2:#c3c2b7;--muted:#898781;
 --rule:#2c2c2a;--ring:rgba(255,255,255,.10)}}}}
[data-theme=dark]{{--surface:#1a1a19;--plane:#0d0d0d;--ink:#fff;--ink2:#c3c2b7;
 --muted:#898781;--rule:#2c2c2a;--ring:rgba(255,255,255,.10)}}
*{{box-sizing:border-box}}
body{{margin:0;padding:32px 28px 44px;background:var(--plane);color:var(--ink);
 font:14px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}}
.wrap{{max-width:1120px;margin:0 auto}}
h1{{font-size:19px;margin:0 0 4px;letter-spacing:-.01em}}
p.sub{{margin:0 0 22px;color:var(--ink2);font-size:13px;max-width:74ch}}
.card{{background:var(--surface);border:1px solid var(--ring);border-radius:12px;
 padding:20px 18px 16px;overflow-x:auto}}
table.hm{{border-collapse:separate;border-spacing:2px;font-variant-numeric:tabular-nums}}
table.hm th{{font-weight:600;color:var(--ink2);font-size:12px}}
table.hm thead th{{padding:0 2px 8px;vertical-align:bottom;text-align:center;line-height:1.25}}
table.hm thead th span{{display:block;font-size:11px;color:var(--muted);
 text-transform:uppercase;letter-spacing:.05em}}
table.hm thead th b{{display:block;font-weight:600;color:var(--ink2)}}
table.hm thead th i{{display:block;font-style:normal;font-size:11px;color:var(--muted)}}
table.hm th[scope=row]{{text-align:right;padding-right:12px;white-space:nowrap}}
th[scope=row] .m{{display:block;font-size:13px;color:var(--ink)}}
th[scope=row] .t{{display:block;font-size:10px;letter-spacing:.06em;
 text-transform:uppercase;color:var(--muted)}}
th[scope=row] .t.exact{{color:#0ca30c}}
td.c{{position:relative;width:82px;height:46px;text-align:center;border-radius:4px;
 background:var(--f);color:var(--i);font-size:13px;font-weight:600;cursor:default}}
td.c span{{position:relative;z-index:1}}
td.c em{{position:absolute;right:5px;bottom:3px;z-index:1;font-style:normal;
 font-size:9px;font-weight:700;letter-spacing:.06em;opacity:.85}}
td.c u{{position:absolute;left:5px;top:4px;z-index:1;width:6px;height:6px;
 border-radius:50%;background:currentColor;opacity:.9}}
td.c.ns::after{{content:"";position:absolute;inset:0;border-radius:4px;
 background:repeating-linear-gradient(45deg,transparent 0 4px,
 color-mix(in srgb,var(--i) 42%,transparent) 4px 5px)}}
td.na{{width:82px;height:46px;text-align:center;color:var(--muted);
 border:1px dashed var(--rule);border-radius:4px}}
.key{{display:flex;flex-wrap:wrap;gap:22px;align-items:center;margin-top:16px;
 padding-top:14px;border-top:1px solid var(--rule);font-size:12px;color:var(--ink2)}}
.ramp{{display:flex;align-items:center;gap:8px}}
.ramp b{{font-weight:600;color:var(--ink2);font-variant-numeric:tabular-nums}}
.ramp .bar{{display:flex;border-radius:3px;overflow:hidden}}
.ramp i{{width:15px;height:11px;display:block}}
.dot{{display:inline-block;width:7px;height:7px;border-radius:50%;
 background:var(--ink2);vertical-align:1px;margin-right:6px}}
.hatch{{display:inline-block;width:15px;height:11px;border-radius:2px;
 vertical-align:-1px;margin-right:6px;background:#9ec5f4;
 background-image:repeating-linear-gradient(45deg,transparent 0 4px,
 rgba(11,11,11,.42) 4px 5px)}}
details{{margin-top:20px}} summary{{cursor:pointer;color:var(--ink2);font-size:13px}}
table.tv{{border-collapse:collapse;margin-top:12px;font-size:12.5px;
 font-variant-numeric:tabular-nums}}
table.tv th,table.tv td{{border-bottom:1px solid var(--rule);padding:5px 12px 5px 0;
 text-align:left}}
table.tv th{{color:var(--ink2);font-weight:600}}
footer{{margin-top:18px;color:var(--muted);font-size:11.5px}}
@media print{{body{{background:#fff}} .card{{break-inside:avoid}}}}
</style>
<div class="wrap">
<h1>Cross-client SHAP feature-importance stability</h1>
<p class="sub">Magnitude-weighted Spearman between clients (K = 5), per-client
background arm. Colour encodes divergence, so cells where clients agree recede and
disagreement is the salient signal. Hatched cells are not
distinguishable from the explainer's own noise floor by the exact exchangeability
test (945 matchings, BH-adjusted) and carry no stability claim.</p>
<div class="card">
<table class="hm"><thead><tr><th></th>{head}</tr></thead>
<tbody>{''.join(rows)}</tbody></table>
<div class="key">
 <span class="ramp"><b>0.65</b>&nbsp;clients diverge
   <span class="bar">{legend}</span> clients agree&nbsp;<b>1.00</b></span>
 <span><span class="hatch"></span>not distinguishable from noise ({n_ns})</span>
 <span><b style="color:#0ca30c">exact</b> = deterministic explainer, no sampling floor</span>
 <span><span class="dot"></span>KernelSHAP measured exactly (grouped, nsamples&nbsp;=&nbsp;510)</span>
</div>
</div>
<details><summary>Table view — {len(tbody)} cells</summary>
<table class="tv"><thead><tr><th>model</th><th>dataset</th><th>condition</th>
<th>arm</th><th>between-client</th><th>p</th><th>distinguishable</th></tr></thead>
<tbody>{''.join(tbody)}</tbody></table></details>
<footer>source: {html.escape(src)} &nbsp;·&nbsp; {len(vals)} cells plotted,
range {min(vals):.3f}–{max(vals):.3f}</footer>
</div>"""

open(OUT, "w").write(doc)
print(f"wrote {OUT}  ({len(vals)} cells, source: {src})")
