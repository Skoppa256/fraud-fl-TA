#!/usr/bin/env python3
"""Audit trail: why the three KernelSHAP models now carry a valid RQ3 claim.

Run on the box from the repo root. Read-only. Six independent legs — each one
prints PASS/FAIL on its own, so a reader can reject any single leg and see what
survives.
"""
import glob, json
import numpy as np, pandas as pd

rows = []
for f in sorted(glob.glob("results/shap_v2/**/stability.json", recursive=True)):
    s = json.load(open(f))
    s["_multi"] = s.get("n_clients", 1) > 1
    s["_bg"] = s.get("bg", "local")
    s["_dir"] = f.rsplit("/", 1)[0]
    rows.append(s)
d = pd.DataFrame(rows)
k = d[(d.explainer == "kernel") & d._multi & (d._bg == "local") & (d.status == "ok")]
# The anchor must be exact INDEPENDENTLY of the models being audited. That is
# LinearSHAP / interventional TreeSHAP on LR, SVM and GBM only. The exact-grouped
# PaySim tier is the same three kernel models re-run without sampling, so folding
# it in would compare the sampled tier against its own exact re-run and the
# comparison could not fail; it is reported below as its own tier instead.
anchor = d[d.explainer.isin(["linear", "tree"])]
exk = d[d.explainer.astype(str).str.contains("exact") & d._multi & (d.status == "ok")]
ok = lambda c: "PASS" if c else "**FAIL**"
print(f"{len(d)} cells on disk · {len(k)} multi-client kernel · "
      f"{len(anchor)} linear/tree anchor · {len(exk)} multi-client exact-grouped\n")

# 1 — the regularisation that manufactured the old floor is gone
bad = int((k.max_nonzero_per_row <= 10).sum())
print(f"[1] l1_reg disabled everywhere")
print(f"    cells still showing the LARS keep-10 signature: {bad}   {ok(bad == 0)}")
print(f"    max nonzeros per row, min across cells: {int(k.max_nonzero_per_row.min())}\n")

# 2 — the floor is measured per client per cell, not borrowed
nf = k.floor.apply(len)
print(f"[2] floor measured per client, per cell (v1 broadcast one BAF number to all)")
print(f"    cells with a {int(nf.mode()[0])}-client floor: {int((nf == nf.mode()[0]).sum())}/{len(k)}   {ok((nf >= 2).all())}")
print(f"    distinct floor values across the grid: {len({round(x,6) for r in k.floor for x in r})}\n")

# 3 — the test is exact, so its p-values have a hard lower bound it must respect
viol = int((k.p_value < 1/945 - 1e-12).sum())
nm = sorted(set(k.n_matchings.dropna().astype(int)))
print(f"[3] exchangeability test is exact, not asymptotic")
print(f"    matchings enumerated: {nm}  (9!! = 945 at K=5)")
print(f"    p < 1/945 violations: {viol}   {ok(viol == 0)}")
print(f"    cells at the 1/945 floor: {int(np.isclose(k.p_value, 1/945).sum())}\n")

# 4 — separation needs no statistics at all
sep = [min(f) > max(b) for f, b in zip(k.floor, k.between)]
print(f"[4] complete separation (every within-client value above every between-client value)")
print(f"    {sum(sep)}/{len(k)} cells separate outright — no test required to read these\n")

# 5 — the instrument says NO when the answer is no
ns = k[k.p_value >= 0.05]
print(f"[5] the test discriminates (a test that always fires proves nothing)")
print(f"    cells NOT distinguishable from noise: {len(ns)}/{len(k)}   {ok(0 < len(ns) < len(k))}")
for _, r in ns.iterrows():
    print(f"      {r.dataset}/{r.model}/{r.condition}/{r.arm}  p={r.p_value:.3f}")
sh = d[(d._bg == 'shared') & d._multi]
if len(sh):
    shns = sh[sh.p_value >= 0.05]
    print(f"    independently, shared-bg arm: {len(shns)}/{len(sh)} non-significant"
          + (f"  (e.g. {shns.iloc[0].model}/{shns.iloc[0].condition}/{shns.iloc[0].arm}"
             f" p={shns.iloc[0].p_value:.3f})" if len(shns) else ""))
print()

# 6 — anchored to explainers that are exact independently of the audited models
def _zero_noise(g, label):
    """seed_delta_max must be present AND exactly 0.0 — a missing value fails.

    dropna() here used to hide the 16 exact-grouped cells, which wrote NaN until
    the field was wired into the kernel path; a silently excluded cell is exactly
    the failure this leg exists to catch."""
    sd = pd.to_numeric(g.get("seed_delta_max"), errors="coerce")
    miss, worst = int(sd.isna().sum()), (float(sd.max()) if sd.notna().any() else float("nan"))
    good = miss == 0 and worst == 0.0
    print(f"    {label}: {len(g)} cells, {miss} missing seed_delta_max, "
          f"worst {worst}   {ok(good)}")
    if miss:
        # Do not let a missing field pass quietly, but do not leave the reader
        # with nothing either: recompute the cross-seed delta from the per-cell
        # artifact. This is over the per-feature MEANS that file stores, so it is
        # weaker than the raw-phi guard in experiments/shap_rq3.py — a cell that
        # clears it still needs the run that populates the field.
        worst_rc, bad_rc = 0.0, []
        for _, r in g[sd.isna()].iterrows():
            try:
                imp = pd.read_csv(f"{r._dir}/importance_per_client_seed.csv")
                cols = [c for c in imp.columns if c.startswith("client_")]
                a = imp[[c for c in cols if c.endswith("_s11")]].to_numpy()
                b = imp[[c for c in cols if c.endswith("_s22")]].to_numpy()
                worst_rc = max(worst_rc, float(np.abs(a - b).max()))
            except Exception as e:                      # unreadable == not verified
                bad_rc.append(f"{r.dataset}/{r.model}/{r.condition}/{r.arm}: {e}")
        print(f"      recomputed from importance_per_client_seed.csv "
              f"(per-feature means, weaker): worst {worst_rc}"
              + (f"   UNREADABLE: {bad_rc}" if bad_rc else ""))
        print( "      -> re-run these cells so the field is written at source")
    return good

am = anchor[anchor._multi]
print(f"[6] anchored by LR / SVM / GBM (LinearSHAP, interventional TreeSHAP)")
_zero_noise(anchor, "anchor")
_zero_noise(exk, "exact-grouped kernel (reported, NOT an anchor)")
if len(am):
    lo, hi = am.weighted_between.min(), am.weighted_between.max()
    klo, khi = k.weighted_between.min(), k.weighted_between.max()
    print(f"    anchor         between-client range: {lo:.4f} – {hi:.4f}  (n={len(am)})")
    print(f"    kernel sampled between-client range: {klo:.4f} – {khi:.4f}  (n={len(k)})")
    if len(exk):
        print(f"    kernel exact   between-client range: "
              f"{exk.weighted_between.min():.4f} – {exk.weighted_between.max():.4f}  "
              f"(n={len(exk)})  [same models as the sampled tier — not an anchor]")
    over = k[k.weighted_between < lo]
    print(f"    sampled-kernel cells more divergent than ANY anchor cell: {len(over)}")
    for _, r in over.iterrows():
        same = exk[(exk.dataset == r.dataset) & (exk.model == r.model)
                   & (exk.condition == r.condition) & (exk.arm == r.arm)]
        ex = f", exact {float(same.weighted_between.iloc[0]):.4f}" if len(same) else ""
        print(f"      {r.dataset}/{r.model}/{r.condition}/{r.arm}  "
              f"sampled {r.weighted_between:.4f}{ex}")
    print("    -> a FINDING about which model families diverge under Non-IID, not a")
    print("       validity check: it is confounded with model family, since the models")
    print("       that admit an exact explainer are also the structurally simplest.\n")

print("SUMMARY")
print(f"  {int((k.p_value < 0.05).sum())}/{len(k)} kernel cells carry a stability value that is")
print( "  distinguishable from the explainer's own measured noise floor.")
print(f"  median between-client stability by model:")
for m, g in k.groupby("model"):
    print(f"    {m:11s} {g.weighted_between.median():.3f}   "
          f"(floor {g.weighted_within.median():.4f}, "
          f"{int((g.p_value<0.05).sum())}/{len(g)} distinguishable)")
