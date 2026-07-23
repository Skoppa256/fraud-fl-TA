"""
t-SNE Visualization (Credit Card dataset) — Global Before/After oversampling.

Credit-card sibling of ``tsne_visualization.py`` (which targets PaySim). Same
logic, only the data source and output tags differ, so the PaySim script and its
PNGs are left untouched.

Generates 2 PNGs:
  - tsne_global_smote_creditcard_1to100.png
  - tsne_global_adasyn_creditcard_1to100.png

See ``tsne_visualization.py`` for the full description of the plotting logic.

Output: results/visualizations/tsne_global_{smote,adasyn}_creditcard_1to100.png
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from imblearn.over_sampling import ADASYN, SMOTE

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from preprocessing.creditcard import load_creditcard


# Target ratio after oversampling. 0.01 -> 1:100 fraud:non-fraud.
SAMPLING_STRATEGY: float = 0.01
RATIO_TAG = "1to100"

# Dataset tag baked into output filenames so credit-card PNGs never collide
# with the PaySim ones.
DATASET_TAG = "creditcard"

# Subsample size for the fraction of x_train fed into t-SNE.
# All fraud rows from x_train are always kept (no cap); non-fraud is capped.
#
# NOTE: credit card has only ~344 fraud rows in x_train. For a 1:100 target the
# desired fraud count is N_NONFRAUD * SAMPLING_STRATEGY; if that is <= the real
# fraud count, SMOTE/ADASYN would have to *remove* fraud and refuse (they only
# add). So N_NONFRAUD must be large enough that N_NONFRAUD * SAMPLING_STRATEGY
# exceeds the real fraud count for any synthetic samples to be generated.
N_NONFRAUD = 100_000

# t-SNE parameters.
TSNE_PERPLEXITY = 30
TSNE_RANDOM_STATE = 42


def plot_tsne_scatter(
    ax,
    x_2d: np.ndarray,
    y: np.ndarray,
    title: str,
    fraud_size: int = 8,
    nonfraud_size: int = 2,
) -> None:
    """Plot t-SNE scatter with blue non-fraud and red fraud dots."""
    y = np.asarray(y).astype(np.int32)
    nf_mask = y == 0
    fr_mask = y == 1
    n_nf = int(nf_mask.sum())
    n_fr = int(fr_mask.sum())

    ax.scatter(
        x_2d[nf_mask, 0],
        x_2d[nf_mask, 1],
        s=nonfraud_size,
        c="blue",
        alpha=0.3,
        label=f"Non-Fraud (N={n_nf:,})",
        zorder=1,
        linewidths=0,
    )
    ax.scatter(
        x_2d[fr_mask, 0],
        x_2d[fr_mask, 1],
        s=fraud_size,
        c="red",
        alpha=0.8,
        label=f"Fraud (N={n_fr:,})",
        zorder=2,
        linewidths=0,
    )

    ax.set_title(title, fontsize=10)
    ax.set_xlabel("t-SNE 1", fontsize=9)
    ax.set_ylabel("t-SNE 2", fontsize=9)
    ax.tick_params(axis="both", labelsize=8)
    ax.legend(loc="best", fontsize=9, markerscale=2, framealpha=0.85)


def stratified_subsample(
    x: np.ndarray,
    y: np.ndarray,
    n_nonfraud: int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (x_sub, y_sub) with ALL fraud rows + up to ``n_nonfraud`` non-fraud rows."""
    rng = np.random.default_rng(random_state)
    idx_fr = np.where(y == 1)[0]
    idx_nf = np.where(y == 0)[0]

    if len(idx_nf) > n_nonfraud:
        idx_nf = rng.choice(idx_nf, size=n_nonfraud, replace=False)

    idx = np.concatenate([idx_fr, idx_nf])
    rng.shuffle(idx)
    return x[idx], y[idx]


def oversample_subsample(
    method: str,
    x: np.ndarray,
    y: np.ndarray,
    smote_k: int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Run SMOTE/ADASYN on a pre-subsampled (x, y).

    Returns (x_res, y_res, note). SMOTE/ADASYN preserve the input rows as
    the first len(x) rows of x_res; synthetic rows are appended after.
    """
    if method == "smote":
        sampler = SMOTE(
            sampling_strategy=SAMPLING_STRATEGY,
            k_neighbors=smote_k,
            random_state=random_state,
        )
    elif method == "adasyn":
        sampler = ADASYN(
            sampling_strategy=SAMPLING_STRATEGY,
            n_neighbors=smote_k,
            random_state=random_state,
        )
    else:
        raise ValueError(f"unknown method: {method!r}")
    try:
        x_res, y_res = sampler.fit_resample(x, y)
        return x_res, y_res, ""
    except ValueError as exc:
        print(f"  [oversample] {method.upper()} failed: {exc} — using raw data")
        return x, y, f"({method.upper()} failed)"


def fit_tsne(x: np.ndarray, label: str) -> tuple[np.ndarray, float, float]:
    """Fit a fresh t-SNE on ``x`` and return (embedding_2d, kl_divergence, elapsed_seconds)."""
    t0 = time.time()
    print(
        f"  [t-SNE] fitting on {len(x):,} rows "
        f"(perplexity={TSNE_PERPLEXITY}) — {label}..."
    )
    tsne = TSNE(
        n_components=2,
        perplexity=TSNE_PERPLEXITY,
        random_state=TSNE_RANDOM_STATE,
        init="pca",
        learning_rate="auto",
    )
    x_2d = tsne.fit_transform(x)
    elapsed = time.time() - t0
    kl = float(tsne.kl_divergence_)
    print(f"  [t-SNE] done in {elapsed:.1f}s (KL divergence = {kl:.4f})")
    return x_2d, kl, elapsed


def generate_png(
    method: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    output_path: str,
    fig_title: str,
    smote_k: int = 5,
    random_state: int = 42,
) -> float:
    """Generate one PNG: Global Before vs After oversampling, shared embedding.

    Returns elapsed seconds.
    """
    method = method.lower()
    if method not in ("smote", "adasyn"):
        raise ValueError(f"method must be 'smote' or 'adasyn', got {method!r}")
    method_label = method.upper()
    t0 = time.time()

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(18, 8))

    # ---------- Stratified subsample (shared by both panels) ----------
    print(f"  [subsample] all fraud + up to {N_NONFRAUD:,} non-fraud...")
    x_sub, y_sub = stratified_subsample(
        x_train, y_train, N_NONFRAUD, random_state=random_state
    )
    n_before = len(y_sub)
    n_fr_before = int((y_sub == 1).sum())
    n_nf_before = int((y_sub == 0).sum())
    print(
        f"  [subsample] n={n_before:,} fraud={n_fr_before:,} "
        f"non-fraud={n_nf_before:,}"
    )

    # ---------- Oversample subsample (synthetic rows appended after originals) ----------
    print(
        f"  [oversample] applying {method_label} on subsample "
        f"(target ratio={SAMPLING_STRATEGY:g})..."
    )
    x_res, y_res, note = oversample_subsample(
        method, x_sub, y_sub, smote_k=smote_k, random_state=random_state
    )
    n_after = len(y_res)
    n_synth = n_after - n_before
    n_fr_after = int((y_res == 1).sum())
    print(
        f"  [oversample] {method_label} done — "
        f"before: {n_before:,} after: {n_after:,} "
        f"(fraud: {n_fr_before:,} -> {n_fr_after:,}, synthetic: {n_synth:,})"
    )

    # Guard: if no synthetic rows were generated, the "After" panel is identical
    # to "Before" and the PNG is meaningless. Fail loudly instead of saving it.
    if n_synth <= 0:
        target_fraud = int(round(N_NONFRAUD * SAMPLING_STRATEGY))
        raise RuntimeError(
            f"{method_label} produced 0 synthetic samples "
            f"(note={note or 'n/a'}). The {SAMPLING_STRATEGY:g} target implies "
            f"~{target_fraud:,} fraud vs {n_fr_before:,} real fraud in the "
            f"subsample. Increase N_NONFRAUD (currently {N_NONFRAUD:,}) so that "
            f"N_NONFRAUD * SAMPLING_STRATEGY > real fraud count."
        )

    # imblearn convention: the first n_before rows of x_res are the original
    # input rows in the same order; rows [n_before:] are the synthetic ones.
    # Assert it so a future imblearn change doesn't silently misalign the panels.
    assert np.array_equal(y_res[:n_before], y_sub), (
        "imblearn output no longer starts with the original rows in order — "
        "shared-embedding alignment is invalid"
    )

    # ---------- Single t-SNE fit on (originals + synthetic) ----------
    x_2d, kl, _ = fit_tsne(x_res, label=f"{method_label} combined")

    # ---------- Before: only the original subsample rows ----------
    plot_tsne_scatter(
        axes[0],
        x_2d[:n_before],
        y_sub,
        title=f"Global — Before {method_label}  (KL={kl:.3f})",
    )

    # ---------- After: all rows (originals stay put, synthetic added) ----------
    right_title = f"Global — After {method_label}  (KL={kl:.3f})"
    if note:
        right_title = f"{right_title} {note}"
    plot_tsne_scatter(axes[1], x_2d, y_res, title=right_title)

    fig.suptitle(fig_title, fontsize=13, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return time.time() - t0


def main() -> None:
    print("Loading Credit Card data...")
    data_path = os.path.join(PROJECT_ROOT, "data/creditcard/creditcard.csv")
    data = load_creditcard(data_path=data_path)
    x_train = np.asarray(data["x_train"])
    y_train = np.asarray(data["y_train"])
    print(
        f"x_train: {x_train.shape} | "
        f"fraud: {int(y_train.sum()):,} ({y_train.mean() * 100:.4f}%)"
    )

    out_dir = os.path.join(PROJECT_ROOT, "results/visualizations")
    os.makedirs(out_dir, exist_ok=True)

    ratio_label = (
        "1:1" if SAMPLING_STRATEGY >= 1.0
        else f"1:{int(round(1 / SAMPLING_STRATEGY))}"
    )
    sub_note = (
        f"(subsample: all fraud + <={N_NONFRAUD:,} non-fraud, "
        f"shared embedding, perplexity={TSNE_PERPLEXITY})"
    )
    print(
        f"\nOversampling sampling_strategy = {SAMPLING_STRATEGY!r} "
        f"({ratio_label} fraud:non-fraud)"
    )
    print(f"t-SNE: {sub_note}")

    runs = [
        (
            "smote",
            os.path.join(out_dir, f"tsne_global_smote_{DATASET_TAG}_{RATIO_TAG}.png"),
            f"t-SNE Visualization — Global (Credit Card) | SMOTE [ratio={ratio_label}] {sub_note}",
        ),
        (
            "adasyn",
            os.path.join(out_dir, f"tsne_global_adasyn_{DATASET_TAG}_{RATIO_TAG}.png"),
            f"t-SNE Visualization — Global (Credit Card) | ADASYN [ratio={ratio_label}] {sub_note}",
        ),
    ]

    timings = []
    for method, output_path, fig_title in runs:
        label = f"Global | {method.upper()}"
        print(f"\n=== Generating PNG: {label} -> {output_path} ===")
        elapsed = generate_png(
            method=method,
            x_train=x_train,
            y_train=y_train,
            output_path=output_path,
            fig_title=fig_title,
        )
        timings.append((label, output_path, elapsed))
        print(f"Saved: {output_path}  (elapsed: {elapsed:.1f}s)")

    print("\n=== All PNGs generated ===")
    for label, path, elapsed in timings:
        exists = os.path.exists(path)
        size_mb = os.path.getsize(path) / (1024 * 1024) if exists else 0.0
        print(
            f"  {label:<24} | {os.path.basename(path):<40} | "
            f"{elapsed:6.1f}s | exists={exists} | {size_mb:.2f} MB"
        )


if __name__ == "__main__":
    main()
