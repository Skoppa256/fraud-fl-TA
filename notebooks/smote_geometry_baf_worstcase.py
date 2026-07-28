"""
SMOTE synthesis geometry — BAF worst-case client (DIAGNOSTIC ONLY).

Sibling of ``tsne_visualization.py`` / ``tsne_visualization_creditcard.py``, but
scoped to ONE client's local training partition instead of the global set,
because the pathology this probes is per-client. It changes NOTHING about SMOTE,
configs, or training — it only visualizes and measures what SMOTE already does.

Target: the documented worst case — BAF, seed 42, Dirichlet alpha=0.5, the client
holding only 21 real fraud samples. The SMOTE target is set by
``--sampling_strategy`` (default 0.01, the uniform experiment rule → ~1:100 →
~3.9k synthetic, ~185x; pass 0.10 to reproduce the earlier ~1:10 → ~39k, ~1862x
comparison). With k_neighbors=5 there are at most 21*5 = 105 directed SMOTE
segments in 55-dim space, so the question is whether the synthetic minority reads
as a genuine cloud or as a low-dimensional wireframe strung between a handful of
real seeds — a property set by the seed count, not the target ratio.

Byte-identical to training: reuses ``preprocessing.baf.load_baf``,
``partitioning.dirichlet.get_partition`` (same seed/alpha/K), and
``preprocessing.smote.apply_smote`` (same k_neighbors, sampling_strategy, and the
same ``base_seed + client_id`` RNG). No raw imblearn call here.

Three-way labelling of the client partition:
  - real majority       : light grey, small, drawn FIRST  (zorder 1)
  - synthetic minority   : green,      small, drawn SECOND (zorder 2)
  - real minority (seeds): red,        LARGE, drawn LAST   (zorder 3)
NOTE: the draw order is deliberately inverted vs the global t-SNE scripts (which
draw synthetic last) so the ~21 real minority points stay visible against the
synthetic mass. Subsample sizes are recorded in the caption so density is not
misread.

Reported to stdout (computed on the FULL partition, not the plotted subsample):
  - # distinct real minority samples that serve as segment endpoints (seeds)
  - synthetic points per SMOTE segment: mean and max
  - fraction of synthetic points whose nearest REAL neighbour is a real MAJORITY
    sample rather than a real minority one (majority-territory landing rate)

Output: results/visualizations/smote_geometry_baf_worstcase_<YYYYMMDD_HHMMSS>.png
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime
from itertools import combinations

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from preprocessing.baf import load_baf
from partitioning.dirichlet import get_partition
from preprocessing.smote import apply_smote


# ---- Worst-case scenario (matches training exactly) -------------------------
SEED: int = 42
ALPHA: float = 0.5
NUM_CLIENTS: int = 5
SAMPLING_STRATEGY: float = 0.01   # default: the uniform experiment rule; override with --sampling_strategy
K_NEIGHBORS: int = 5
BASE_SEED: int = 42               # apply_smote uses random_state = BASE_SEED + client_id
# The worst-case client is identified by the fewest real fraud among clients that
# still qualify for SMOTE (>= k_neighbors + 1). Asserted at runtime.
EXPECTED_MIN_FRAUD: int = 21

# ---- Embedding subsample caps (recorded in the caption) ---------------------
# Majority is ~391k and synthetic ~39k — both are capped for a tractable t-SNE.
# ALL real minority are always kept (there are only ~21). The metrics above are
# computed on the FULL partition regardless of these caps.
N_MAJORITY_EMBED: int = 8_000
N_SYNTH_EMBED: int = 8_000

# ---- t-SNE params (match the sibling scripts) -------------------------------
TSNE_PERPLEXITY = 30
TSNE_RANDOM_STATE = 42


def find_worstcase_client(clients: list[dict]) -> dict:
    """Return the client with the fewest real minority samples (the worst case)."""
    return min(clients, key=lambda c: int((c["y"] == 1).sum()))


def segment_stats(x_synth: np.ndarray, x_min: np.ndarray) -> dict:
    """Attribute each synthetic point to the SMOTE segment it lies on.

    A SMOTE synthetic point s = a + lambda (b - a) lies exactly on the segment
    between two real minority points a, b (both in ``x_min``). So for each
    synthetic point we find the minority PAIR minimising point-to-segment
    distance; that pair is its segment. From the attribution we derive:
      - distinct minority points used as segment endpoints ("seeds")
      - synthetic points per populated segment (mean, max)
      - the residual point-to-segment distance (should be ~0 if truly on-segment)
    """
    n_min = len(x_min)
    pairs = list(combinations(range(n_min), 2))
    # distance from every synthetic point to every candidate segment
    best_dist = np.full(len(x_synth), np.inf, dtype=np.float64)
    best_pair = np.full(len(x_synth), -1, dtype=np.int64)
    for pi, (i, j) in enumerate(pairs):
        a = x_min[i]
        b = x_min[j]
        ab = b - a
        denom = float(ab @ ab)
        if denom == 0.0:
            continue
        t = np.clip((x_synth - a) @ ab / denom, 0.0, 1.0)   # (N,)
        proj = a + t[:, None] * ab                          # (N, D)
        d = np.linalg.norm(x_synth - proj, axis=1)          # (N,)
        upd = d < best_dist
        best_dist[upd] = d[upd]
        best_pair[upd] = pi

    populated = np.unique(best_pair[best_pair >= 0])
    counts = np.array([(best_pair == pi).sum() for pi in populated])
    endpoints = set()
    for pi in populated:
        i, j = pairs[pi]
        endpoints.add(i)
        endpoints.add(j)

    return {
        "n_pairs_possible": len(pairs),
        "n_segments_populated": int(len(populated)),
        "n_distinct_seed_endpoints": int(len(endpoints)),
        "per_segment_mean": float(counts.mean()) if counts.size else 0.0,
        "per_segment_max": int(counts.max()) if counts.size else 0,
        "residual_median": float(np.median(best_dist)),
        "residual_p99": float(np.percentile(best_dist, 99)),
    }


def majority_territory_fraction(
    x_synth: np.ndarray, x_maj: np.ndarray, x_min: np.ndarray
) -> dict:
    """Fraction of synthetic points whose nearest REAL neighbour is majority.

    Builds a single 1-NN index over all real points (majority + minority),
    queries every synthetic point, and checks the label of its nearest real
    neighbour. A high fraction means synthetic minority is landing inside
    majority territory — i.e. SMOTE is manufacturing minority mass where the
    real data is overwhelmingly majority.
    """
    x_real = np.vstack([x_maj, x_min])
    is_majority = np.zeros(len(x_real), dtype=bool)
    is_majority[: len(x_maj)] = True
    nn = NearestNeighbors(n_neighbors=1, n_jobs=-1)
    nn.fit(x_real)
    _, idx = nn.kneighbors(x_synth, return_distance=True)
    nn_is_majority = is_majority[idx[:, 0]]
    frac = float(nn_is_majority.mean())
    return {
        "n_synth": int(len(x_synth)),
        "frac_nn_majority": frac,
        "n_nn_majority": int(nn_is_majority.sum()),
    }


def _subsample(x: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """Return up to ``n`` rows of ``x`` (indices), without replacement."""
    if len(x) <= n:
        return np.arange(len(x))
    return rng.choice(len(x), size=n, replace=False)


def plot_panel(ax, x_2d, kind: str, sizes: dict, title: str) -> None:
    """Three-way scatter. ``kind`` labels x/y axes ('t-SNE' or 'PC')."""
    # x_2d rows are ordered [majority | synthetic | real-minority] (see main()).
    n_maj = sizes["maj"]
    n_syn = sizes["syn"]
    n_min = sizes["min"]
    maj = x_2d[:n_maj]
    syn = x_2d[n_maj : n_maj + n_syn]
    mino = x_2d[n_maj + n_syn :]

    # Draw order: majority first, synthetic second, real minority LAST + LARGER.
    ax.scatter(maj[:, 0], maj[:, 1], s=3, c="lightgrey", alpha=0.5,
               label=f"Real majority (N={n_maj:,})", zorder=1, linewidths=0)
    ax.scatter(syn[:, 0], syn[:, 1], s=6, c="green", alpha=0.5,
               label=f"Synthetic minority (N={n_syn:,})", zorder=2, linewidths=0)
    ax.scatter(mino[:, 0], mino[:, 1], s=90, c="red", alpha=0.95,
               edgecolors="black", linewidths=0.6,
               label=f"Real minority / SMOTE seeds (N={n_min:,})", zorder=3)

    lab = "t-SNE" if kind == "tsne" else "PC"
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(f"{lab} 1", fontsize=9)
    ax.set_ylabel(f"{lab} 2", fontsize=9)
    ax.tick_params(axis="both", labelsize=8)
    ax.legend(loc="best", fontsize=8, markerscale=1.2, framealpha=0.85)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SMOTE synthesis geometry, BAF worst-case client (diagnostic)."
    )
    parser.add_argument(
        "--sampling_strategy", type=float, default=SAMPLING_STRATEGY,
        help="minority:majority target (default 0.01, the experiment rule; "
             "pass 0.10 to reproduce the earlier comparison).",
    )
    args = parser.parse_args()
    sampling_strategy = float(args.sampling_strategy)

    t_start = time.time()
    print(f"Loading BAF data...  (sampling_strategy={sampling_strategy:g})")
    data = load_baf(data_path=os.path.join(PROJECT_ROOT, "data/baf/baf.csv"))
    x_train = np.asarray(data["x_train"])
    y_train = np.asarray(data["y_train"])
    print(f"x_train: {x_train.shape} | fraud: {int(y_train.sum()):,} "
          f"({y_train.mean() * 100:.4f}%)")

    # ---- Reproduce the worst-case client (byte-identical to training) --------
    clients = get_partition(x_train, y_train, scheme="dirichlet", alpha=ALPHA,
                            num_clients=NUM_CLIENTS, random_state=SEED)
    client = find_worstcase_client(clients)
    cid = int(client["client_id"])
    n_before = int(client["n_samples"])
    n_min_real = int((client["y"] == 1).sum())
    n_maj_real = n_before - n_min_real
    print(f"\nWorst-case client: id={cid} | n={n_before:,} | "
          f"real minority={n_min_real} | real majority={n_maj_real:,}")
    if n_min_real != EXPECTED_MIN_FRAUD:
        print(f"  [warn] expected {EXPECTED_MIN_FRAUD} real fraud, got {n_min_real} "
              f"— partition may have changed; proceeding with the true worst case.")

    # ---- Run SMOTE exactly as preprocessing/smote.py does --------------------
    out = apply_smote(client, enabled=True, sampling_strategy=sampling_strategy,
                      k_neighbors=K_NEIGHBORS, base_seed=BASE_SEED)
    if not out.get("smote_applied"):
        raise RuntimeError(
            f"SMOTE did not run on client {cid} (skip_reason={out.get('skip_reason')}). "
            f"This diagnostic requires the worst-case client to oversample."
        )
    x_res = np.asarray(out["x"])
    y_res = np.asarray(out["y"])
    # imblearn contract: originals first (in order), synthetic appended after.
    assert np.array_equal(y_res[:n_before], client["y"]), \
        "imblearn no longer returns originals first — synthetic indexing invalid"
    x_synth = x_res[n_before:]
    n_synth = len(x_synth)
    mult = out.get("synthesis_multiplier", n_synth / max(n_min_real, 1))
    print(f"SMOTE applied — synthetic={n_synth:,} from {n_min_real} real minority "
          f"(x{mult:.0f} synthesis multiplier); SMOTE seed = "
          f"{BASE_SEED} + {cid} = {BASE_SEED + cid}")

    # Real majority / real minority rows (from the original block).
    y_orig = client["y"]
    x_orig = np.asarray(client["x"])
    x_maj = x_orig[y_orig == 0]
    x_min = x_orig[y_orig == 1]

    # ---- Metrics on the FULL partition --------------------------------------
    print("\n=== SMOTE synthesis geometry (full partition) ===")
    seg = segment_stats(x_synth, x_min)
    print(f"  segments possible (C(n_min,2))      : {seg['n_pairs_possible']}")
    print(f"  segments actually populated          : {seg['n_segments_populated']}")
    print(f"  distinct real-minority seed endpoints: {seg['n_distinct_seed_endpoints']}"
          f" / {n_min_real}")
    print(f"  synthetic per segment  mean / max    : "
          f"{seg['per_segment_mean']:.1f} / {seg['per_segment_max']:,}")
    print(f"  on-segment residual  median / p99    : "
          f"{seg['residual_median']:.2e} / {seg['residual_p99']:.2e}  (≈0 confirms on-segment)")

    t_nn = time.time()
    terr = majority_territory_fraction(x_synth, x_maj, x_min)
    print(f"  synthetic whose nearest REAL neighbour is MAJORITY: "
          f"{terr['n_nn_majority']:,}/{terr['n_synth']:,} "
          f"= {terr['frac_nn_majority'] * 100:.2f}%   ({time.time() - t_nn:.1f}s)")

    # ---- Build the plotted subsample -----------------------------------------
    rng = np.random.default_rng(SEED)
    maj_idx = _subsample(x_maj, N_MAJORITY_EMBED, rng)
    syn_idx = _subsample(x_synth, N_SYNTH_EMBED, rng)
    x_maj_e = x_maj[maj_idx]
    x_syn_e = x_synth[syn_idx]
    # Order: majority | synthetic | real minority (matches plot_panel slicing).
    x_embed = np.vstack([x_maj_e, x_syn_e, x_min])
    sizes = {"maj": len(x_maj_e), "syn": len(x_syn_e), "min": len(x_min)}
    print(f"\nPlotted subsample: majority={sizes['maj']:,} (of {n_maj_real:,}) | "
          f"synthetic={sizes['syn']:,} (of {n_synth:,}) | "
          f"real minority={sizes['min']} (all)")

    # ---- PCA (fit on FULL real client data; transform the plotted subsample) --
    pca = PCA(n_components=2, random_state=42)
    pca.fit(np.vstack([x_maj, x_min]))
    x_pca = pca.transform(x_embed)
    evr = pca.explained_variance_ratio_

    # ---- t-SNE (fit on the plotted subsample) --------------------------------
    print("Fitting t-SNE on the plotted subsample...")
    t_tsne = time.time()
    tsne = TSNE(n_components=2, perplexity=TSNE_PERPLEXITY,
                random_state=TSNE_RANDOM_STATE, init="pca", learning_rate="auto")
    x_tsne = tsne.fit_transform(x_embed)
    print(f"  t-SNE done in {time.time() - t_tsne:.1f}s "
          f"(KL={float(tsne.kl_divergence_):.3f})")

    # ---- Figure: t-SNE (left) + PCA (right) ----------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(20, 9))
    plot_panel(axes[0], x_tsne, "tsne",
               sizes, f"t-SNE  (perplexity={TSNE_PERPLEXITY}, KL={float(tsne.kl_divergence_):.3f})")
    plot_panel(axes[1], x_pca, "pca",
               sizes, f"PCA  (PC1+PC2 = {evr.sum() * 100:.1f}% variance)")

    caption = (
        f"BAF worst-case client {cid} — seed {SEED}, Dirichlet α={ALPHA}, K={NUM_CLIENTS}, "
        f"SMOTE sampling_strategy={sampling_strategy:g}, k_neighbors={K_NEIGHBORS}.  "
        f"Full partition: {n_maj_real:,} real majority, {n_min_real} real minority, "
        f"{n_synth:,} synthetic (×{mult:.0f}).  "
        f"Plotted: {sizes['maj']:,} majority + {sizes['syn']:,} synthetic (subsampled) + "
        f"{sizes['min']} minority (all).  "
        f"Segments populated {seg['n_segments_populated']}/{seg['n_pairs_possible']}; "
        f"synthetic-per-segment mean {seg['per_segment_mean']:.0f}, max {seg['per_segment_max']:,}; "
        f"nearest real neighbour is majority for {terr['frac_nn_majority'] * 100:.1f}% of synthetic."
    )
    fig.suptitle(
        "SMOTE synthesis geometry — BAF worst-case client (DIAGNOSTIC)\n" + caption,
        fontsize=11, y=0.99,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    out_dir = os.path.join(PROJECT_ROOT, "results/visualizations")
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    ratio_tag = f"{sampling_strategy:g}".replace(".", "p")
    out_path = os.path.join(
        out_dir, f"smote_geometry_baf_worstcase_ss{ratio_tag}_{ts}.png"
    )
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {out_path}")
    print(f"Total elapsed: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
