"""RQ3 SHAP v2 — Phase 1.1 + Phase 2 production runner. RUN ON THE BOX (tmux TA_Deo).

Supersedes experiments/shap_analysis.py for RQ3 stability outputs. READ-ONLY over
results/models/ + the data cache. Writes ONLY under results/shap_v2/ — results/shap/
(the v1, pre-l1_reg-fix record) is deliberately left untouched so every number that
moves can be shown next to its old value.

    python experiments/shap_rq3.py --stage main                    # all 96 cells
    python experiments/shap_rq3.py --stage main --datasets paysim  # subset
    python experiments/shap_rq3.py --stage twobg                   # shared-background arm
    python experiments/shap_rq3.py --stage paysim-exact            # grouped exact (510)
    python experiments/shap_rq3.py --stage paysim-exact --include-ungrouped  # + 8190
    python experiments/shap_rq3.py --stage budget                  # nsamples-vs-N probe
    (every stage resumes: cells with a stability.json are skipped; --no-skip-existing
     forces recomputation)

What changes vs v1 (each item measured or verified before being adopted):

* ``l1_reg=False`` on every KernelSHAP call. shap >= 0.47 defaults to
  l1_reg="num_features(10)": LARS keeps 10 features and sets every other phi to
  exactly 0.0, per instance. On BAF (55 features) that zeroes >= 45/55 per row,
  makes rank statistics tie-dominated, and inflates the noise floor because the
  LARS selection is a discontinuous function of the coalition draw. A regression
  guard here fails loudly if any cell with M > 10 still shows <= 10 nonzeros in
  every explained row (the l1 signature).
* Two SHAP-only coalition seeds (SHAP_SEEDS) per client, identical explained
  instances and identical background: per-cell, per-client noise floors replace
  the single BAF-measured floor that was broadcast to all datasets. Training
  seeds are untouched; nothing is retrained.
* Common random numbers: every between-client comparison pairs vectors computed
  under the SAME coalition seed (both clients share the draw), which cuts the
  run-to-run sd of the between-client Spearman at identical compute.
* The floor-as-threshold reading is replaced by an exact exchangeability test
  (evaluation/shap_inference.py): under H0 "all clients share one true importance
  vector" the 2K vectors are exchangeable, the null is all 945 perfect matchings
  at K=5, and Benjamini-Hochberg is applied across the multi-client kernel cells.
* Degenerate guard (unchanged from v1): any all-zero/constant vector -> the cell
  is "undefined" with NO stability numbers, never a fabricated 1.0.
* Deterministic explainers (Linear/interventional Tree) are re-run at two global
  seeds and their outputs verified bit-identical — the "exact tier" whose
  between-client spread carries zero estimator noise and anchors the kernel tier.
* seed_delta_max is measured on EVERY cell, kernel included: max|delta phi| over
  the raw attributions of the two coalition seeds. On a sampled kernel cell it is
  the estimator error's magnitude in attribution units; on any cell the run calls
  deterministic it must be exactly 0.0, and an exactness guard aborts the cell
  otherwise. Nothing claims zero noise on an unmeasured field.
* Two-background design (--stage twobg): each selected cell re-run with a SHARED
  background (pooled local backgrounds -> k-means) next to the per-client one.
  The difference separates "the model behaves differently on this client" from
  "this client's background distribution differs". Default scope: PaySim
  dirichlet kernel cells (the largest-gap family), extend via --datasets/--conditions.
* PaySim exact tier (--stage paysim-exact): grouping the 5 type_* one-hots into
  one Shapley player gives M_eff = 9, so nsamples = 2^9 - 2 = 510 enumerates 100%
  of the kernel weight — exact, zero-noise attributions at ~today's budget.
  Ungrouped exact (2^13 - 2 = 8190, 13 players, comparable to v1/v2 rankings) is
  ~16.4x the per-instance cost and therefore gated behind --include-ungrouped;
  the runner prints a measured per-client extrapolation before committing.
  Acceptance for both is enforced by the exactness guard above, not checked by
  hand: max|delta phi| across the two seeds == 0.0 exactly, or the run aborts.
* Budget probe (--stage budget): on ONE real cell/client, measures the two-seed
  floor across (nsamples, n_explain) pairs at fixed model-eval budget, deciding
  where extra compute goes before any is committed.

Sequential like v1 (no Ray; the FedXGBllr path OOM-history is why), so the
memory profile equals v1's. Wall-clock guard via --max-hours.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "models" / "fedxgbllr"))

from evaluation import shap_inference as SI  # noqa: E402
from evaluation import shap_stability as ST  # noqa: E402
from experiments.shap_analysis import (  # noqa: E402
    CHEAP, EXPLAINER, KMEANS_K, KNOWN, N_BG, N_EXPLAIN, NSAMPLES, NUM_CLIENTS, SEED,
    _sv2d, client_backgrounds, discover, explanation_set, feature_names,
    manifest_hash, scope_order,
)
from experiments.shap_analysis import load_predictor  # noqa: E402

# Two SHAP-only coalition seeds (the same pair the noise-floor probe used —
# never the training seed; nothing is retrained).
SHAP_SEEDS = (11, 22)
BH_ALPHA = 0.05
OUT = PROJECT_ROOT / "results" / "shap_v2"
SUMMARY = OUT / "shap_summary_v2.csv"
SUMMARY_COLS = [
    "dataset", "model", "condition", "arm", "bg", "explainer", "deterministic",
    "n_clients", "manifest_sha256", "data_hash", "partition_hash",
    "spearman", "jaccard_at5", "kuncheva",
    "floor_mean", "floor_min", "between_mean", "between_sd", "delta",
    "p_value", "p_adj", "n_matchings", "disattenuated",
    "weighted_between", "weighted_within",
    "max_nonzero_per_row", "seed_delta_max", "status", "verdict",
    "top_feature", "note",
]
# PaySim's one categorical; its 5 one-hot columns become ONE Shapley player.
GROUP_PREFIXES = {"paysim": ["type"]}


def _f(x):  # json-safe float (nan -> None)
    x = float(x)
    return None if x != x else x


def _c(x):  # csv cell: "undefined" for nan/None, else rounded
    if x is None or (isinstance(x, float) and x != x):
        return "undefined"
    return round(x, 4) if isinstance(x, float) else x


# --------------------------------------------------------------------------- #
# importance computation
# --------------------------------------------------------------------------- #
def kernel_g(obj, bg_km, X, seed, nsamples):
    """One (client, seed) global importance vector on log-odds, l1_reg=False.

    Returns (mean|phi| vector, max nonzeros per explained row, raw phi matrix).
    The nonzero count is the l1_reg regression guard's input: under the
    num_features(10) default every row has <= 10 nonzeros; with l1_reg=False a
    non-degenerate model yields dense rows. The raw matrix is what the cross-seed
    delta is measured on, so the exact tier's claim is made over the ATTRIBUTIONS
    themselves and not over their per-feature means, which two opposite-signed
    sampling errors could reconcile by accident.
    """
    import shap
    np.random.seed(seed)  # the CRN lever: same seed => same coalition draw
    ex = shap.KernelExplainer(obj, bg_km)
    sv = _sv2d(ex.shap_values(X, nsamples=nsamples, l1_reg=False, silent=True), len(X))
    return np.abs(sv).mean(axis=0), int((sv != 0).sum(axis=1).max()), sv


def det_g(obj, kind, bg, X):
    """Deterministic explainer importance + bit-identity check across two seeds.

    Linear/interventional-Tree SHAP are exact given the background; we VERIFY
    (not assume) by running twice under different global seeds and recording
    max|delta phi| — expected exactly 0.0.
    """
    import shap
    svs = []
    for s in SHAP_SEEDS:
        np.random.seed(s)
        if kind == "linear":
            ex = shap.LinearExplainer(obj, bg)
            sv = _sv2d(ex.shap_values(X), len(X))
        else:  # tree, interventional with per-client background (v1 semantics)
            ex = shap.TreeExplainer(obj, data=bg, feature_perturbation="interventional")
            sv = _sv2d(ex.shap_values(X, check_additivity=False), len(X))
        svs.append(sv)
    dmax = float(np.abs(svs[0] - svs[1]).max())
    return np.abs(svs[0]).mean(axis=0), dmax


def shared_background(bgs):
    """Pooled-local shared background (the Ducange-style contrast arm).

    Pools the RAW per-client background samples (client_backgrounds returns
    <=100 local post-SMOTE rows each, not summaries), so the k-means applied
    afterwards sees ~500 real points and derives its centroid weights from them
    natively. Pooling per-client CENTROIDS instead would require propagating each
    centroid's cluster size by hand — the DenseData 4th-positional weight issue —
    and would throw away the within-client spread; pooling raw rows avoids both.
    """
    return np.concatenate(bgs, axis=0)


def client_explanation_sets(art, rng, n=N_EXPLAIN):
    """Per-client explanation sets drawn from each client's OWN partition.

    Required by the shared-background arm: with one global model, one pooled
    background and one shared explanation set, every client receives byte-identical
    inputs and the client axis collapses (all agreement 1.0 by construction). The
    local arm varies the background and holds the explained instances fixed; the
    shared arm does the converse — it holds the background fixed and lets each
    client explain the region its own data actually occupies. So the two arms
    isolate the two confounded mechanisms rather than differing in one factor.

    Sampled class-proportionally from the client's PRE-SMOTE partition (real rows
    only, never synthetic minority points), mirroring how the local arm explains
    real central-test rows. Clients smaller than ``n`` contribute all they have.
    Note these are training-split rows: the partition covers x_train only, so
    there is no per-client held-out split in this pipeline.
    """
    from experiments import data_cache
    clients, phash = data_cache.get_partition_clients(
        art["dataset"], SEED, art["condition"], art["alpha"], NUM_CLIENTS)
    out = []
    for c in clients:
        x = np.asarray(c["x"], np.float32)
        y = np.asarray(c["y"]).astype(int)
        pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
        take_pos = min(max(1, int(round(n * y.mean()))) if pos.size else 0, pos.size)
        take_neg = min(n - take_pos, neg.size)
        idx = np.concatenate([
            rng.choice(pos, take_pos, replace=False) if take_pos else np.empty(0, int),
            rng.choice(neg, take_neg, replace=False) if take_neg else np.empty(0, int),
        ]).astype(int)
        rng.shuffle(idx)
        out.append(x[idx])
    return out, phash


# --------------------------------------------------------------------------- #
# per-cell processing
# --------------------------------------------------------------------------- #
def cell_dir(art, bg_mode, root=OUT):
    d = root / art["dataset"] / art["model"] / f"{art['condition']}_{art['arm']}"
    return d / "bg_shared" if bg_mode == "shared" else d


def write_cell(cdir, fnames, prov, G, stats, extras):
    """Persist per-cell artifacts: importance matrix, stability.json, profile.csv."""
    cdir.mkdir(parents=True, exist_ok=True)
    K, S, M = G.shape
    mean_imp = np.abs(G).mean(axis=(0, 1))
    with open(cdir / "importance_per_client_seed.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["feature"]
                   + [f"client_{c}_s{SHAP_SEEDS[s]}" for c in range(K) for s in range(S)]
                   + ["mean"])
        for j, name in enumerate(fnames):
            w.writerow([name]
                       + [f"{G[c, s, j]:.6g}" for c in range(K) for s in range(S)]
                       + [f"{mean_imp[j]:.6g}"])
    payload = {**prov, **{k: (_f(v) if isinstance(v, float) else v)
                          for k, v in stats.items() if k not in ("floor", "between")},
               "floor": [_f(v) for v in stats.get("floor", [])],
               "between": [_f(v) for v in stats.get("between", [])],
               "shap_seeds": list(SHAP_SEEDS), **extras}
    (cdir / "stability.json").write_text(json.dumps(payload, indent=2))
    if stats.get("status") == "ok" and K >= 2:
        prof = SI.kuncheva_profile(G, d=M)
        with open(cdir / "profile.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["k", "between_mean", "within_mean", "within_min", "within_max"])
            for i, k in enumerate(prof["k"]):
                w.writerow([k] + [_c(prof[c][i]) for c in
                                  ("between_mean", "within_mean", "within_min", "within_max")])
    order = np.argsort(mean_imp)[::-1]
    (cdir / "aggregated.json").write_text(json.dumps(
        {**prov, "top10_features": [[fnames[i], float(mean_imp[i])] for i in order[:10]]},
        indent=2))
    return mean_imp


def legacy_metrics(G, d):
    """v1-continuity metrics from the two-seed data: per-seed cross-client
    Jaccard@5 / Kuncheva@5 averaged over seeds (CRN semantics)."""
    K, S, _ = G.shape
    if K < 2:
        return float("nan"), float("nan")
    jac = float(np.mean([ST.jaccard_at_k([G[c, s] for c in range(K)], k=5)
                         for s in range(S)]))
    kun = float(np.mean([ST.kuncheva_index([G[c, s] for c in range(K)], d=d, k=5)
                         for s in range(S)]))
    return jac, kun


def process_cell(art, bg_mode, X, dh, nsamples=NSAMPLES, groups=None, gnames=None,
                 root=OUT, explainer_tag=None):
    """Compute one cell (one bg mode). Returns a summary row dict."""
    import shap
    kind = EXPLAINER[art["model"]]
    fnames = gnames if groups is not None else feature_names(art)
    d = len(fnames)
    bgs, phash = client_backgrounds(art, np.random.default_rng(SEED + 1))
    Xs = [X] * len(bgs)
    if bg_mode == "shared":
        # Hold the background fixed (pooled) and let each client explain its own
        # data region. Sharing BOTH the background and the explanation set would
        # hand every client identical inputs and collapse the client axis.
        bgs = [shared_background(bgs)] * len(bgs)
        Xs, _ = client_explanation_sets(art, np.random.default_rng(SEED + 2))
        if len(Xs) != len(bgs):
            raise RuntimeError(f"{len(Xs)} client explanation sets vs "
                               f"{len(bgs)} backgrounds")
    obj, kind = load_predictor(art)

    K = len(bgs)
    G = np.empty((K, 2, d))
    nz_max, det_delta = -1, float("nan")
    for c, bg in enumerate(bgs):
        X = Xs[c]
        if kind == "kernel":
            bg_sum = shap.kmeans(bg, KMEANS_K)
            if groups is not None:
                from shap.utils._legacy import DenseData
                # Re-wrap the k-means summary with a grouping. DenseData's 4th
                # positional is the background weights; shap.kmeans sets them to
                # the CLUSTER SIZES and KernelExplainer uses them for E[f(x)], so
                # they must be carried over — omitting them silently reweights the
                # reference distribution to uniform. Copy: DenseData normalises
                # the array in place.
                bg_sum = DenseData(np.asarray(bg_sum.data), list(gnames),
                                   list(groups), np.array(bg_sum.weights, float))
            sv_prev, dd = None, float("nan")
            for s, seed in enumerate(SHAP_SEEDS):
                t = time.time()
                g_vec, nz, sv = kernel_g(obj, bg_sum, X, seed, nsamples)
                if g_vec.shape[0] != d:
                    raise RuntimeError(
                        f"output width {g_vec.shape[0]} != {d} expected players — "
                        "DenseData grouping unsupported by this shap version?")
                G[c, s] = g_vec
                nz_max = max(nz_max, nz)
                # cross-seed delta on the raw attributions: exactly 0.0 once
                # nsamples enumerates every coalition, and a measured magnitude
                # of the estimator's own error below that.
                if sv_prev is None:
                    sv_prev = sv
                else:
                    dd = float(np.abs(sv - sv_prev).max())
                    det_delta = dd if det_delta != det_delta else max(det_delta, dd)
                print(f"      client {c} seed {seed}: {time.time()-t:.1f}s"
                      + ("" if dd != dd else f" seed_delta={dd:.3g}"))
        else:
            t = time.time()
            g, dd = det_g(obj, kind, bg, X)
            # keep the WORST client's cross-seed delta, not the last one's —
            # one non-reproducible client must not be masked by a later clean one
            det_delta = dd if det_delta != det_delta else max(det_delta, dd)
            G[c, 0] = G[c, 1] = g
            print(f"      client {c} (deterministic x2): {time.time()-t:.1f}s "
                  f"seed_delta={dd:.3g}")

    # ---- l1_reg regression guard (kernel only): the num_features(10) default
    # leaves <= 10 nonzeros in EVERY row; one denser row disproves it.
    if kind == "kernel" and groups is None and d > 10 and 0 <= nz_max <= 10:
        raise RuntimeError(
            f"l1_reg signature detected: max nonzeros/row = {nz_max} <= 10 with "
            f"M = {d}. KernelSHAP appears to be running under the "
            "num_features(10) default — l1_reg=False did not take effect.")

    # ---- exactness guard: every cell this run calls deterministic claims zero
    # estimator noise — the kernel tier by enumerating all 2^d - 2 coalitions,
    # the linear/tree tier by construction. _det_stats hard-codes floor = 1.0 on
    # exactly that premise, so measure it rather than assume it. NaN (never
    # measured) fails the comparison too, which is the intent.
    exact_kernel = kind == "kernel" and nsamples >= 2 ** d - 2
    deterministic = kind != "kernel" or exact_kernel
    if deterministic and K >= 1 and det_delta != 0.0:
        raise RuntimeError(
            f"cell claims zero estimator noise but max|delta phi| across seeds "
            f"{SHAP_SEEDS} = {det_delta:.3g}, not 0.0"
            + (f" (kernel, nsamples = {nsamples} enumerates all 2^{d} - 2 "
               f"coalitions)" if exact_kernel else f" ({kind} explainer)"))

    stats = SI.cell_inference(G) if kind == "kernel" else _det_stats(G)
    wbw = (SI.weighted_between_within(G) if stats.get("status") == "ok" and K >= 2
           else {"weighted_between_mean": float("nan"), "weighted_within_mean": float("nan")})
    jac, kun = ((float("nan"),) * 2 if stats.get("status") != "ok"
                else legacy_metrics(G, d))

    prov = {"dataset": art["dataset"], "model": art["model"],
            "condition": art["condition"], "arm": art["arm"], "bg": bg_mode,
            "explainer": explainer_tag or kind,
            "deterministic": deterministic,
            "nsamples": nsamples if kind == "kernel" else "n/a",
            "l1_reg": False if kind == "kernel" else "n/a",
            "n_clients": K, "manifest_sha256": manifest_hash(art["dir"]),
            "data_hash": dh[:16],
            "partition_hash": phash[:16] if isinstance(phash, str) else phash,
            # which input the client axis varies over, and how many rows each
            # client actually explained (small clients contribute fewer)
            "explanation_set": ("per-client local partition" if bg_mode == "shared"
                                else "shared central test subset"),
            "n_explained_per_client": [int(len(x)) for x in Xs],
            "near_zero_fraction": [[round(v, 4) for v in ST.near_zero_fraction(G[c])]
                                   for c in range(K)]}
    mean_imp = write_cell(cell_dir(art, bg_mode, root), fnames, prov, G, stats,
                          {"max_nonzero_per_row": nz_max if kind == "kernel" else "n/a",
                           "seed_delta_max": _f(det_delta),
                           "jaccard_at5": _f(jac), "kuncheva": _f(kun),
                           "weighted_between": _f(wbw["weighted_between_mean"]),
                           "weighted_within": _f(wbw["weighted_within_mean"])})

    row = {**{k: prov[k] for k in ("dataset", "model", "condition", "arm", "bg",
                                   "explainer", "deterministic", "n_clients",
                                   "manifest_sha256", "data_hash", "partition_hash")},
           "spearman": _c(stats.get("between_mean", float("nan"))),
           "jaccard_at5": _c(jac), "kuncheva": _c(kun),
           "floor_mean": _c(stats.get("floor_mean", float("nan"))),
           "floor_min": _c(stats.get("floor_min", float("nan"))),
           "between_mean": _c(stats.get("between_mean", float("nan"))),
           "between_sd": _c(stats.get("between_sd", float("nan"))),
           "delta": _c(stats.get("delta", float("nan"))),
           "p_value": _c(stats.get("p_value", float("nan"))), "p_adj": "n/a",
           "n_matchings": stats.get("n_matchings", 0),
           "disattenuated": _c(stats.get("disattenuated", float("nan"))),
           "weighted_between": _c(wbw["weighted_between_mean"]),
           "weighted_within": _c(wbw["weighted_within_mean"]),
           "max_nonzero_per_row": nz_max if kind == "kernel" else "n/a",
           "seed_delta_max": _c(det_delta),
           "status": stats.get("status", "undefined"),
           "verdict": _verdict(stats, kind, K),
           "top_feature": fnames[int(np.argmax(mean_imp))],
           "note": stats.get("reason") or ""}
    print(f"    [{art['dataset']}/{art['model']}/{art['condition']}_{art['arm']}"
          f"{'/shared' if bg_mode == 'shared' else ''}] K={K} "
          f"floor={row['floor_mean']} between={row['between_mean']} "
          f"delta={row['delta']} p={row['p_value']} status={row['status']}"
          + (f"  [{row['note']}]" if row["note"] else ""))
    return row


def _det_stats(G):
    """Stats for the deterministic tier: no estimator noise, so no floor/test —
    the between-client spread IS real (the calibration anchor for the kernel tier)."""
    K = G.shape[0]
    if K < 2:
        return {"status": "ok", "reason": None, "n_clients": K,
                "floor_mean": float("nan"), "floor_min": float("nan"),
                "between_mean": float("nan"), "between_sd": float("nan"),
                "delta": float("nan"), "p_value": float("nan"), "n_matchings": 0,
                "disattenuated": float("nan"), "floor": [], "between": []}
    degen = ST.degenerate_clients([G[c, 0] for c in range(K)])
    if degen:
        return {"status": "undefined", "n_clients": K,
                "reason": f"degenerate attributions: clients {degen}", "floor": [],
                "between": []}
    if SI.collapsed_seeds(G):
        return {"status": "undefined", "n_clients": K, "floor": [], "between": [],
                "reason": "collapsed client axis: every client's importance vector "
                          "is identical, so cross-client agreement is 1.0 by "
                          "construction and measures nothing"}
    B = [SI.spearman(G[i, 0], G[j, 0])
         for i in range(K) for j in range(i + 1, K)]
    return {"status": "ok", "reason": None, "n_clients": K,
            "floor_mean": 1.0, "floor_min": 1.0,  # verified via seed_delta_max == 0
            "between_mean": float(np.mean(B)),
            "between_sd": float(np.std(B, ddof=1)) if len(B) > 1 else float("nan"),
            "delta": 1.0 - float(np.mean(B)), "p_value": float("nan"),
            "n_matchings": 0, "disattenuated": float("nan"),
            "floor": [1.0] * K, "between": B}


def _verdict(stats, kind, K):
    if stats.get("status") != "ok":
        return "undefined"
    if K < 2:
        return "single_client"
    if kind != "kernel":
        return "exact_reference"
    return "pending_bh"  # finalized when the summary is (re)written


# --------------------------------------------------------------------------- #
# summary with BH across the multi-client kernel cells
# --------------------------------------------------------------------------- #
def write_summary(new_rows):
    """Idempotent merge + BH pass. The BH family is the multi-client kernel cells
    with a finite p, per bg mode (local and shared are separate families)."""
    OUT.mkdir(parents=True, exist_ok=True)
    rows = {}
    if SUMMARY.is_file():
        for r in csv.DictReader(open(SUMMARY)):
            rows[(r["dataset"], r["model"], r["condition"], r["arm"],
                  r.get("bg", "local"), r["explainer"])] = r
    for r in new_rows:
        rows[(r["dataset"], r["model"], r["condition"], r["arm"], r["bg"],
              r["explainer"])] = {k: str(v) for k, v in r.items()}

    allr = list(rows.values())
    families = {(r.get("bg", "local"), r["explainer"]) for r in allr}
    for bg, exp_tag in families:
        fam = [r for r in allr
               if r.get("bg", "local") == bg and r["explainer"] == exp_tag
               and r["verdict"] in ("pending_bh", "clients_differ", "noise_limited")]
        pv = []
        for r in fam:
            try:
                pv.append(float(r["p_value"]))
            except (TypeError, ValueError):
                pv.append(float("nan"))
        adj = SI.benjamini_hochberg(pv)
        for r, a in zip(fam, adj):
            if a != a:
                r["p_adj"], r["verdict"] = "undefined", "undefined"
            else:
                r["p_adj"] = str(round(a, 4))
                r["verdict"] = "clients_differ" if a <= BH_ALPHA else "noise_limited"
    with open(SUMMARY, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_COLS, extrasaction="ignore")
        w.writeheader()
        for r in sorted(allr, key=lambda r: (r["dataset"], r["model"],
                                             r["condition"], r["arm"],
                                             r.get("bg", "local"), r["explainer"])):
            r.setdefault("bg", "local")
            w.writerow(r)
    print(f"\nSummary -> {SUMMARY} ({len(allr)} rows; BH alpha={BH_ALPHA} per "
          "(bg, explainer) family).")


# --------------------------------------------------------------------------- #
# stages
# --------------------------------------------------------------------------- #
def run_cells(cells, bg_mode, args, nsamples=NSAMPLES, groups_for=None, root=OUT,
              explainer_tag=None):
    rows, t0, done = [], time.time(), 0
    X_cache = {}
    for art in cells:
        cd = cell_dir(art, bg_mode, root)
        if args.skip_existing and (cd / "stability.json").is_file():
            print(f"    skip (exists) {cd.relative_to(PROJECT_ROOT)}")
            continue
        if args.max_hours and (time.time() - t0) / 3600 > args.max_hours:
            print(f"\n[wall guard] {args.max_hours}h reached after {done} cells.")
            break
        if art["dataset"] not in X_cache:
            X_cache[art["dataset"]] = explanation_set(art["dataset"],
                                                      np.random.default_rng(SEED))
        X, dh = X_cache[art["dataset"]]
        groups = gnames = None
        if groups_for:
            groups, gnames = groups_for(art)
        try:
            rows.append(process_cell(art, bg_mode, X, dh, nsamples=nsamples,
                                     groups=groups, gnames=gnames, root=root,
                                     explainer_tag=explainer_tag))
            done += 1
            write_summary(rows[-1:])  # checkpoint after every cell (resume-safe)
        except Exception:  # noqa: BLE001
            import traceback
            print(f"    [ERROR] {art['dataset']}/{art['model']}/"
                  f"{art['condition']}_{art['arm']} ({bg_mode}):")
            print("\n".join("        " + ln
                            for ln in traceback.format_exc().splitlines()))
    return rows


def build_groups(art):
    """(groups, group_names) for one-hot grouping, or (None, None) if no mapping."""
    prefixes = GROUP_PREFIXES.get(art["dataset"])
    if not prefixes:
        return None, None
    fnames = feature_names(art)
    groups, gnames, used = [], [], set()
    for p in prefixes:
        idx = [i for i, n in enumerate(fnames) if n.startswith(p + "_")]
        if idx:
            groups.append(idx), gnames.append(p)
            used.update(idx)
    for i, n in enumerate(fnames):
        if i not in used:
            groups.append([i]), gnames.append(n)
    # deterministic order: singletons in feature order, categorical groups appended
    order = sorted(range(len(groups)), key=lambda k: groups[k][0])
    return [groups[k] for k in order], [gnames[k] for k in order]


def stage_main(args):
    arts = discover()
    cells = scope_order(arts, set(args.datasets.split(",")), set(args.models.split(",")))
    print(f"{len(cells)} cells in scope (cheap first, then expensive BAF->ULB->PaySim).")
    run_cells(cells, "local", args)


def stage_twobg(args):
    arts = [a for a in discover()
            if EXPLAINER[a["model"]] == "kernel"
            and a["dataset"] in set(args.datasets.split(","))
            and a["condition"] in set(args.conditions.split(","))
            and a["condition"] != "centralized"]  # shared == local for 1 client
    cells = scope_order(arts, set(args.datasets.split(",")),
                        set(args.models.split(",")))
    print(f"two-background arm: {len(cells)} kernel cells "
          f"({args.datasets} x {args.conditions}).")
    run_cells(cells, "shared", args)


def stage_paysim_exact(args):
    arts = [a for a in discover()
            if a["dataset"] == "paysim" and EXPLAINER[a["model"]] == "kernel"
            and a["model"] in set(args.models.split(","))]
    if not arts:
        print("no paysim kernel cells in scope — nothing to do.")
        return
    # grouped: M_eff = 9 -> 2^9 - 2 = 510 enumerates 100% of the kernel weight
    g0, n0 = build_groups(arts[0])
    m_eff = len(g0)
    ns_grouped = 2 ** m_eff - 2
    print(f"PaySim exact — grouped: {m_eff} players, nsamples={ns_grouped} "
          f"(full enumeration).")
    run_cells(arts, "local", args, nsamples=ns_grouped, groups_for=build_groups,
              root=OUT / "paysim_exact" / "grouped",
              explainer_tag="kernel-exact-grouped")
    if args.include_ungrouped:
        m = len(feature_names(arts[0]))
        ns_full = 2 ** m - 2
        print(f"PaySim exact — ungrouped: {m} players, nsamples={ns_full} "
              f"(~{ns_full / NSAMPLES:.1f}x the ns={NSAMPLES} per-instance cost; "
              "watch the first client's timing before letting it run).")
        run_cells(arts, "local", args, nsamples=ns_full,
                  root=OUT / "paysim_exact" / "ungrouped",
                  explainer_tag="kernel-exact")
    else:
        print("(ungrouped 8190 run skipped — pass --include-ungrouped after the "
              "grouped tier confirms the ranking.)")


def _explanation_subset_n(dataset, n, rng):
    """Class-proportional test subset of size n (the budget probe needs n > N_EXPLAIN)."""
    from experiments import data_cache
    d, _ = data_cache.get_preprocessed(dataset, SEED)
    xte = np.asarray(d["x_test"], np.float32)
    yte = np.asarray(d["y_test"]).astype(int)
    n_pos = max(1, int(round(n * yte.mean())))
    pi = rng.choice(np.where(yte == 1)[0], min(n_pos, int((yte == 1).sum())),
                    replace=False)
    ni = rng.choice(np.where(yte == 0)[0], n - len(pi), replace=False)
    idx = np.concatenate([pi, ni])
    rng.shuffle(idx)
    return xte[idx]


def stage_budget(args):
    """nsamples vs N_explain at fixed model-eval budget, on one real cell/client.

    Decides where GPU time goes (Phase 3.4): if part of the estimator error is
    systematic across instances, raising nsamples beats raising N_explain."""
    import shap
    ds, model, cond, arm = args.budget_cell.split("/")
    art = next(a for a in discover()
               if (a["dataset"], a["model"], a["condition"], a["arm"])
               == (ds, model, cond, arm))
    bgs, _ = client_backgrounds(art, np.random.default_rng(SEED + 1))
    bg_km = shap.kmeans(bgs[0], KMEANS_K)
    obj, _ = load_predictor(art)
    budget = NSAMPLES * N_EXPLAIN
    pairs = [(125, 2000), (250, 1000), (500, 500), (1000, 250), (2000, 125)]
    OUT.mkdir(parents=True, exist_ok=True)
    out_rows = []
    print(f"budget probe on {args.budget_cell} client 0 — fixed budget "
          f"{budget} evals, seeds {SHAP_SEEDS}")
    for ns, ne in pairs:
        assert ns * ne == budget
        X = _explanation_subset_n(ds, ne, np.random.default_rng(SEED))
        t = time.time()
        g1, _, _ = kernel_g(obj, bg_km, X, SHAP_SEEDS[0], ns)
        g2, _, _ = kernel_g(obj, bg_km, X, SHAP_SEEDS[1], ns)
        rho = SI.spearman(g1, g2)
        jac = ST.jaccard_at_k([g1, g2], k=5)
        print(f"    nsamples={ns:>5} n_explain={ne:>5}: floor rho={rho:.4f} "
              f"jac5={jac:.2f} [{time.time()-t:.0f}s]")
        out_rows.append({"nsamples": ns, "n_explain": ne, "floor_spearman": round(rho, 4),
                         "floor_jaccard_at5": round(jac, 4)})
    with open(OUT / "budget_probe.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0]))
        w.writeheader()
        w.writerows(out_rows)
    print(f"-> {OUT / 'budget_probe.csv'}")


# --------------------------------------------------------------------------- #
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", default="main",
                    choices=["main", "twobg", "paysim-exact", "budget"])
    ap.add_argument("--datasets", default=None,
                    help="default: all for main, paysim for twobg")
    ap.add_argument("--models", default=",".join(KNOWN))
    ap.add_argument("--conditions", default="dirichlet",
                    help="twobg only (default dirichlet)")
    ap.add_argument("--max-hours", type=float, default=None)
    ap.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    ap.add_argument("--include-ungrouped", action="store_true",
                    help="paysim-exact: also run the 2^13-2 = 8190 ungrouped tier")
    ap.add_argument("--budget-cell", default="baf/ffd/dirichlet/none",
                    help="budget: dataset/model/condition/arm (client 0)")
    args = ap.parse_args(argv)
    if args.datasets is None:
        args.datasets = "paysim" if args.stage == "twobg" else "baf,creditcard,paysim"

    try:
        import shap
        print(f"shap {shap.__version__} | stage={args.stage} l1_reg=False "
              f"seeds={SHAP_SEEDS} nsamples={NSAMPLES} bg={N_BG}->kmeans{KMEANS_K} "
              f"explain={N_EXPLAIN}")
        maj, minor = (int(x) for x in shap.__version__.split(".")[:2])
        if (maj, minor) < (0, 47):
            print("WARNING: shap < 0.47 — the l1_reg default audit does not apply "
                  "to this version; results remain valid but pin ==0.49.1 to match "
                  "the recorded artifacts.")
    except Exception as e:  # noqa: BLE001
        print(f"FATAL shap import: {e}")
        return 1

    if not discover():
        print("NO artifacts under results/models/ — run on the box. Nothing to do.")
        return 2
    OUT.mkdir(parents=True, exist_ok=True)
    {"main": stage_main, "twobg": stage_twobg,
     "paysim-exact": stage_paysim_exact, "budget": stage_budget}[args.stage](args)
    print("Done. Nothing outside results/shap_v2/ was written; results/shap/ (v1) "
          "is untouched.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
