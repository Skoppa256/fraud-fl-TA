"""Per-cell statistical inference for cross-client SHAP stability (RQ3, Phase 2).

Replaces the single broadcast noise floor + "at or below" reading with a per-cell
hypothesis test. Inputs are per-client global importance vectors measured at TWO
coalition seeds under common random numbers (CRN): array ``g`` of shape
``(K, 2, M)`` — K clients x 2 SHAP-only seeds x M features.

Definitions (K clients, seeds s1/s2, all correlations Spearman):

* within/floor set  ``F = { rho(g[c,0], g[c,1]) : c }``            — K values
* between set       ``B = { rho(g[i,s], g[j,s]) : i<j, s }``       — K(K-1) values
  (same-seed pairs only: CRN makes both sides of each comparison share the
  coalition draw, which cuts the run-to-run sd of B without moving its mean)

The test (:func:`exchangeability_test`): under H0 "all clients share one true
importance vector", the 2K vectors are exchangeable draws (same signal +
estimator noise), so the observed pairing into (client, seed1)/(client, seed2)
is arbitrary and the null is EVERY perfect matching of the 2K vectors into K
pairs. For K=5 that is 9!! = 945 matchings — fully enumerated, an exact test
(minimum achievable p = 1/945). The statistic is mean(within-pair rho) minus
mean(rho over all pairs NOT in the matching). This replaces a pooled permutation
test, which is invalid here: the K(K-1) between values are computed from only K
client vectors and are heavily dependent, so pooling them with the K floor
values overstates precision.

Multiplicity: :func:`benjamini_hochberg` across the multi-client kernel cells.

Also here: the chance-corrected stability profile over k
(:func:`kuncheva_profile`) and a magnitude-weighted rank correlation
(:func:`weighted_spearman`) so the near-zero tail stops driving the statistic.

Everything in this module is pure numpy/scipy — no shap, no torch — so it is
unit-testable off the box (``tests/test_shap_inference.py``).
"""

from __future__ import annotations

import itertools
import math
import warnings
from typing import Dict, Iterator, List, Sequence, Tuple

import numpy as np

from evaluation import shap_stability as ST


# --------------------------------------------------------------------------- #
# rank correlations
# --------------------------------------------------------------------------- #
def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rho; nan (never a coerced 0.0) when undefined (constant input)."""
    from scipy.stats import spearmanr

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # ConstantInputWarning IS the nan we handle
        rho = spearmanr(np.asarray(a, float), np.asarray(b, float)).correlation
    return float("nan") if rho is None else float(rho)


def weighted_spearman(a: np.ndarray, b: np.ndarray, w: np.ndarray) -> float:
    """Magnitude-weighted rank correlation: weighted Pearson on the rank vectors.

    ``w`` must be non-negative (typically mean |phi| across the vectors being
    compared, so near-zero tail features carry ~no weight). nan when either rank
    vector is constant or the weights sum to 0.
    """
    from scipy.stats import rankdata

    a, b, w = (np.asarray(x, float) for x in (a, b, w))
    if w.sum() <= 0 or not np.all(np.isfinite(w)):
        return float("nan")
    ra, rb = rankdata(a), rankdata(b)
    wn = w / w.sum()
    ma, mb = float(wn @ ra), float(wn @ rb)
    ca, cb = ra - ma, rb - mb
    va, vb = float(wn @ ca**2), float(wn @ cb**2)
    if va <= 0 or vb <= 0:
        return float("nan")
    return float((wn @ (ca * cb)) / math.sqrt(va * vb))


# --------------------------------------------------------------------------- #
# exact exchangeability test over perfect matchings
# --------------------------------------------------------------------------- #
def perfect_matchings(items: Sequence[int]) -> Iterator[List[Tuple[int, int]]]:
    """Yield every perfect matching of an even-length sequence into pairs.

    Count is (n-1)!! — for n=10 (K=5 clients x 2 seeds) that is 945.
    """
    items = list(items)
    if not items:
        yield []
        return
    a, rest = items[0], items[1:]
    for i, b in enumerate(rest):
        for tail in perfect_matchings(rest[:i] + rest[i + 1:]):
            yield [(a, b)] + tail


def double_factorial_odd(n_items: int) -> int:
    """(n-1)!! for even n_items — the number of perfect matchings."""
    out = 1
    for k in range(n_items - 1, 0, -2):
        out *= k
    return out


def pairwise_spearman_matrix(vectors: Sequence[np.ndarray]) -> np.ndarray:
    """Symmetric matrix of pairwise Spearman rho (diagonal 0, unused)."""
    n = len(vectors)
    R = np.zeros((n, n))
    for i, j in itertools.combinations(range(n), 2):
        R[i, j] = R[j, i] = spearman(vectors[i], vectors[j])
    return R


def exchangeability_test(vectors: Sequence[np.ndarray]) -> Tuple[float, float, int]:
    """Exact matching-null test of H0: all clients share one importance vector.

    ``vectors``: 2K vectors ordered [(c1,s1),(c1,s2),(c2,s1),(c2,s2),...].
    Statistic: mean(within-pair rho) - mean(rho over pairs outside the matching).
    Null: all perfect matchings of the 2K vectors (exhaustive — exact test).
    Returns (obs, p_value, n_matchings); (nan, nan, 0) if any pairwise rho is
    undefined (degenerate vector) — callers must report the cell as undefined.
    """
    n = len(vectors)
    if n < 4 or n % 2:
        return float("nan"), float("nan"), 0
    R = pairwise_spearman_matrix(vectors)
    iu = np.triu_indices(n, k=1)
    if not np.all(np.isfinite(R[iu])):
        return float("nan"), float("nan"), 0
    all_pairs = list(itertools.combinations(range(n), 2))

    def stat(matching):
        ms = {frozenset(p) for p in matching}
        within = [R[a, b] for a, b in matching]
        between = [R[i, j] for i, j in all_pairs if frozenset((i, j)) not in ms]
        return float(np.mean(within) - np.mean(between))

    obs = stat([(2 * k, 2 * k + 1) for k in range(n // 2)])
    null = np.array([stat(m) for m in perfect_matchings(list(range(n)))])
    # the observed matching is one of the enumerated matchings, so p >= 1/len(null)
    return obs, float((null >= obs).mean()), int(len(null))


def benjamini_hochberg(pvals: Sequence[float]) -> List[float]:
    """BH step-up adjusted p-values (monotone, capped at 1). nan passes through."""
    p = np.asarray(pvals, float)
    out = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    m = int(ok.sum())
    if m == 0:
        return out.tolist()
    order = np.argsort(p[ok])
    ranked = p[ok][order] * m / (np.arange(m) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]  # enforce monotonicity
    adj = np.empty(m)
    adj[order] = np.minimum(ranked, 1.0)
    out[ok] = adj
    return out.tolist()


# --------------------------------------------------------------------------- #
# per-cell summary
# --------------------------------------------------------------------------- #
def collapsed_seeds(g: np.ndarray, atol: float = 0.0) -> List[int]:
    """Seeds at which EVERY client's importance vector is identical.

    A structural collapse, not a measurement: if the explainer received the same
    model, the same background and the same explained instances for every client,
    the client dimension carries no information and every cross-client statistic
    is 1.0 by construction. This is the third such artifact in this pipeline
    (``tree_path_dependent`` made tree stability trivially 1.0; the all-zero
    PaySim cells faked Jaccard = 1.0 off identical tie-breaking; a pooled shared
    background with a shared explanation set collapses the client axis outright),
    so the check is permanent rather than per-incident.

    Exact equality by default: identical inputs produce bitwise-identical output,
    whereas genuinely similar-but-distinct clients differ in the low bits. Cells
    flagged here must be reported undefined with NO stability metrics.
    """
    if g.shape[0] < 2:
        return []
    out = []
    for s in range(g.shape[1]):
        ref = g[0, s]
        if all(np.allclose(g[c, s], ref, rtol=0.0, atol=atol)
               for c in range(1, g.shape[0])):
            out.append(s)
    return out


def floor_and_between(g: np.ndarray) -> Tuple[List[float], List[float]]:
    """F (per-client cross-seed rho) and B (same-seed cross-client rho) from
    ``g`` of shape (K, 2, M)."""
    K = g.shape[0]
    F = [spearman(g[c, 0], g[c, 1]) for c in range(K)]
    B = [spearman(g[i, s], g[j, s])
         for s in range(g.shape[1]) for i, j in itertools.combinations(range(K), 2)]
    return F, B


def cell_inference(g: np.ndarray) -> Dict:
    """Full Phase-2 statistics for one cell from ``g`` (K clients, 2 seeds, M).

    Degenerate guard first: any (client, seed) vector that is all-zero/constant
    (``ST.is_degenerate``) makes every rank statistic an artifact, so the cell
    is returned as status="undefined" with NO stability numbers — never a
    fabricated agreement value.
    """
    g = np.asarray(g, float)
    K = g.shape[0]
    degen = [(c, s) for c in range(K) for s in range(g.shape[1])
             if ST.is_degenerate(g[c, s])]
    collapsed = collapsed_seeds(g)
    base = {"n_clients": K, "degenerate_client_seed": degen,
            "collapsed_seeds": collapsed}
    if degen:
        base.update(status="undefined",
                    reason=f"degenerate attributions at (client, seed) {degen}: "
                           "all-zero/constant importance carries no ranking signal")
        return base
    if collapsed:
        base.update(status="undefined",
                    reason=f"collapsed client axis at seed index {collapsed}: every "
                           "client's importance vector is identical, so all "
                           "cross-client agreement is 1.0 by construction and "
                           "measures nothing (check that clients actually receive "
                           "differing inputs)")
        return base
    if K < 2:
        # single (pseudo-)client: the cross-seed rho is still a valid reliability
        # number, but there is no between-client quantity and no test.
        floor = spearman(g[0, 0], g[0, 1])
        base.update(status="ok", reason=None, floor=[floor], between=[],
                    floor_mean=floor, floor_min=floor,
                    between_mean=float("nan"), between_sd=float("nan"),
                    delta=float("nan"), obs=float("nan"), p_value=float("nan"),
                    n_matchings=0, disattenuated=float("nan"))
        return base

    F, B = floor_and_between(g)
    flat = [g[c, s] for c in range(K) for s in range(g.shape[1])]
    obs, p, n_match = exchangeability_test(flat)
    floor_mean, between_mean = float(np.mean(F)), float(np.mean(B))
    base.update(
        status="ok", reason=None, floor=F, between=B,
        floor_mean=floor_mean, floor_min=float(np.min(F)),
        between_mean=between_mean,
        between_sd=float(np.std(B, ddof=1)) if len(B) > 1 else float("nan"),
        delta=floor_mean - between_mean, obs=obs, p_value=p, n_matchings=n_match,
        # secondary, flagged approximate (over-corrects slightly in simulation):
        disattenuated=(between_mean / floor_mean if floor_mean > 0 else float("nan")),
    )
    return base


# --------------------------------------------------------------------------- #
# stability profile over k
# --------------------------------------------------------------------------- #
def kuncheva_profile(g: np.ndarray, d: int, ks: Sequence[int] | None = None) -> Dict:
    """Chance-corrected consistency at every top-k, between vs within (floor band).

    Returns {"k": [...], "between_mean": [...], "within_mean": [...],
    "within_min": [...], "within_max": [...]} where between uses same-seed
    cross-client pairs and within uses per-client cross-seed pairs. Where the
    between curve leaves the within band is where the ranking stops being
    readable. Comparable across 13/30/55 features (the chance correction's job).
    """
    g = np.asarray(g, float)
    K, S = g.shape[0], g.shape[1]
    ks = list(ks) if ks is not None else list(range(1, d + 1))
    sets = [[ST._topk_set(g[c, s], k) for k in ks] for c in range(K) for s in range(S)]

    def _idx(c, s):
        return c * S + s

    out = {"k": ks, "between_mean": [], "within_mean": [], "within_min": [],
           "within_max": []}
    for ki, k in enumerate(ks):
        bet = [ST.kuncheva_pair(sets[_idx(i, s)][ki], sets[_idx(j, s)][ki], d)
               for s in range(S) for i, j in itertools.combinations(range(K), 2)]
        wit = [ST.kuncheva_pair(sets[_idx(c, 0)][ki], sets[_idx(c, 1)][ki], d)
               for c in range(K)] if S > 1 else []
        out["between_mean"].append(float(np.mean(bet)) if bet else float("nan"))
        out["within_mean"].append(float(np.mean(wit)) if wit else float("nan"))
        out["within_min"].append(float(np.min(wit)) if wit else float("nan"))
        out["within_max"].append(float(np.max(wit)) if wit else float("nan"))
    return out


def weighted_between_within(g: np.ndarray) -> Dict:
    """Magnitude-weighted rank correlation, between vs within, weights = overall
    mean |importance| across all (client, seed) vectors of the cell."""
    g = np.asarray(g, float)
    K, S = g.shape[0], g.shape[1]
    w = np.abs(g).mean(axis=(0, 1))
    bet = [weighted_spearman(g[i, s], g[j, s], w)
           for s in range(S) for i, j in itertools.combinations(range(K), 2)]
    wit = [weighted_spearman(g[c, 0], g[c, 1], w) for c in range(K)] if S > 1 else []
    return {"weighted_between_mean": float(np.mean(bet)) if bet else float("nan"),
            "weighted_within_mean": float(np.mean(wit)) if wit else float("nan")}
