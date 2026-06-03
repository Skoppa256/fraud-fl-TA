"""
PCA Visualization — Before/After oversampling across Global and Per-Client partitions.

Generates 8 PNGs (4 partition schemes x 2 oversamplers):
  - schemes : IID, Dirichlet alpha=0.5, Dirichlet alpha=1.0, Dirichlet alpha=5.0
  - methods : SMOTE, ADASYN

Each PNG shows:
  - 6 rows (Global + 5 clients) x 2 columns (Before / After oversampling)
  - PCA fitted once on global x_train (before any resampling), reused for all subplots
  - Non-fraud: blue, Fraud: red

Output: results/visualizations/pca_*_{smote,adasyn}.png
"""

from __future__ import annotations

import os
import sys
import time
from typing import Optional

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from imblearn.over_sampling import ADASYN, SMOTE

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from preprocessing.paysim import load_paysim
from partitioning.dirichlet import get_partition
from preprocessing.smote import apply_smote
from preprocessing.adasyn import apply_adasyn


def plot_pca_scatter(
    ax,
    x_2d: np.ndarray,
    y: np.ndarray,
    title: str,
    fraud_size: int = 8,
    nonfraud_size: int = 1,
) -> None:
    """Plot PCA scatter with blue non-fraud and red fraud dots."""
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

    ax.set_title(title, fontsize=8)
    ax.set_xlabel("PC1", fontsize=8)
    ax.set_ylabel("PC2", fontsize=8)
    ax.tick_params(axis="both", labelsize=7)
    ax.legend(loc="best", fontsize=8, markerscale=2, framealpha=0.85)


def _global_oversample(
    method: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    smote_k: int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Run a single global resampling on x_train. Returns (x, y, note)."""
    if method == "smote":
        sampler = SMOTE(
            sampling_strategy="auto",
            k_neighbors=smote_k,
            random_state=random_state,
        )
    elif method == "adasyn":
        sampler = ADASYN(
            sampling_strategy="auto",
            n_neighbors=smote_k,
            random_state=random_state,
        )
    else:
        raise ValueError(f"unknown method: {method!r}")
    try:
        x_res, y_res = sampler.fit_resample(x_train, y_train)
        return x_res, y_res, ""
    except ValueError as exc:
        print(f"  [global] {method.upper()} failed: {exc} — using raw data")
        return x_train, y_train, f"({method.upper()} failed)"
    except MemoryError:
        print(
            f"  [global] MemoryError on full {method.upper()} — "
            f"falling back to 500K stratified subsample"
        )
        rng = np.random.default_rng(random_state)
        idx_fr = np.where(y_train == 1)[0]
        idx_nf = np.where(y_train == 0)[0]
        sub_nf_n = max(1, 500_000 - len(idx_fr))
        idx_nf_sub = rng.choice(idx_nf, size=min(sub_nf_n, len(idx_nf)), replace=False)
        idx_sub = np.concatenate([idx_fr, idx_nf_sub])
        rng.shuffle(idx_sub)
        try:
            x_res, y_res = sampler.fit_resample(x_train[idx_sub], y_train[idx_sub])
            return x_res, y_res, f"(500K subsample fallback)"
        except ValueError as exc:
            print(f"  [global] {method.upper()} failed on subsample: {exc}")
            return x_train[idx_sub], y_train[idx_sub], f"({method.upper()} failed)"


def generate_png(
    scheme: str,
    alpha: Optional[float],
    method: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    pca: PCA,
    output_path: str,
    fig_title: str,
    num_clients: int = 5,
    smote_k: int = 5,
    random_state: int = 42,
) -> float:
    """Generate one PNG for a given partition scheme and oversampler. Returns elapsed seconds."""
    method = method.lower()
    if method not in ("smote", "adasyn"):
        raise ValueError(f"method must be 'smote' or 'adasyn', got {method!r}")
    t0 = time.time()

    method_label = method.upper()

    fig, axes = plt.subplots(
        nrows=num_clients + 1,
        ncols=2,
        figsize=(20, 30),
    )

    # ---------- Row 0: Global ----------
    print(f"  [global] transforming x_train (before {method_label})...")
    x_train_2d = pca.transform(x_train)
    plot_pca_scatter(
        axes[0, 0],
        x_train_2d,
        y_train,
        title=f"Global — Before {method_label}",
    )

    print(f"  [global] applying {method_label} on full x_train...")
    x_train_resampled, y_train_resampled, note = _global_oversample(
        method, x_train, y_train, smote_k=smote_k, random_state=random_state
    )
    print(
        f"  [global] {method_label} done — before: {len(y_train):,} "
        f"after: {len(y_train_resampled):,}"
    )
    x_train_resampled_2d = pca.transform(x_train_resampled)
    right_title = f"Global — After {method_label}"
    if note:
        right_title = f"{right_title} {note}"
    plot_pca_scatter(
        axes[0, 1],
        x_train_resampled_2d,
        y_train_resampled,
        title=right_title,
    )
    del x_train_2d, x_train_resampled, x_train_resampled_2d

    # ---------- Rows 1..K: Per-client ----------
    print(f"  [clients] partitioning with scheme={scheme}, alpha={alpha}...")
    clients = get_partition(
        x_train=x_train,
        y_train=y_train,
        scheme=scheme,
        alpha=alpha,
        num_clients=num_clients,
        random_state=random_state,
    )

    apply_fn = apply_smote if method == "smote" else apply_adasyn
    applied_key = "smote_applied" if method == "smote" else "adasyn_applied"
    apply_kwargs = (
        {"k_neighbors": smote_k}
        if method == "smote"
        else {"n_neighbors": smote_k}
    )

    for k, client in enumerate(clients):
        row = k + 1
        x_k = client["x"]
        y_k = client["y"]
        n_k = int(client["n_samples"])
        n_fr_k = int(client["n_fraud"])
        print(
            f"  [client {k}] n={n_k:,} fraud={n_fr_k} "
            f"({(n_fr_k / max(n_k, 1)) * 100:.4f}%) — transforming..."
        )

        x_k_2d = pca.transform(x_k) if n_k > 0 else np.zeros((0, 2), dtype=np.float32)
        plot_pca_scatter(
            axes[row, 0],
            x_k_2d,
            y_k,
            title=f"Client {k} — Before {method_label}",
        )

        client_resampled = apply_fn(
            client,
            enabled=True,
            sampling_strategy="auto",
            base_seed=random_state,
            **apply_kwargs,
        )

        x_k_after = client_resampled["x"]
        y_k_after = client_resampled["y"]
        x_k_after_2d = (
            pca.transform(x_k_after)
            if len(y_k_after) > 0
            else np.zeros((0, 2), dtype=np.float32)
        )

        if client_resampled.get(applied_key, False):
            right_title = f"Client {k} — After {method_label}"
        else:
            right_title = (
                f"Client {k} — After {method_label} "
                f"({method_label} skipped/failed: insufficient fraud)"
            )

        plot_pca_scatter(
            axes[row, 1],
            x_k_after_2d,
            y_k_after,
            title=right_title,
        )

    fig.suptitle(fig_title, fontsize=14, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    elapsed = time.time() - t0
    return elapsed


def main() -> None:
    print("Loading PaySim data...")
    data_path = os.path.join(PROJECT_ROOT, "data/paysim/paysim.csv")
    data = load_paysim(data_path=data_path)
    x_train = np.asarray(data["x_train"])
    y_train = np.asarray(data["y_train"])
    print(
        f"x_train: {x_train.shape} | "
        f"fraud: {int(y_train.sum()):,} ({y_train.mean() * 100:.4f}%)"
    )

    print("\nFitting PCA on x_train (before any oversampling)...")
    t_pca = time.time()
    pca = PCA(n_components=2, random_state=42)
    pca.fit(x_train)
    print(f"PCA fit done in {time.time() - t_pca:.1f}s")
    evr = pca.explained_variance_ratio_
    print(
        f"PCA explained variance ratio: PC1={evr[0]:.4f}  PC2={evr[1]:.4f}  "
        f"sum={evr.sum():.4f}"
    )

    out_dir = os.path.join(PROJECT_ROOT, "results/visualizations")
    os.makedirs(out_dir, exist_ok=True)

    evr_note = f" (PC1+PC2 = {evr.sum() * 100:.2f}% variance)"

    schemes = [
        # SMOTE
        ("iid",       None, "smote",
         os.path.join(out_dir, "pca_iid_smote.png"),
         f"PCA Visualization — IID Partition (K=5) | SMOTE{evr_note}"),
        ("dirichlet", 0.5,  "smote",
         os.path.join(out_dir, "pca_dirichlet_alpha0.5_smote.png"),
         f"PCA Visualization — Dirichlet α=0.5 (K=5) | SMOTE{evr_note}"),
        ("dirichlet", 1.0,  "smote",
         os.path.join(out_dir, "pca_dirichlet_alpha1.0_smote.png"),
         f"PCA Visualization — Dirichlet α=1.0 (K=5) | SMOTE{evr_note}"),
        ("dirichlet", 5.0,  "smote",
         os.path.join(out_dir, "pca_dirichlet_alpha5.0_smote.png"),
         f"PCA Visualization — Dirichlet α=5.0 (K=5) | SMOTE{evr_note}"),
        # ADASYN
        ("iid",       None, "adasyn",
         os.path.join(out_dir, "pca_iid_adasyn.png"),
         f"PCA Visualization — IID Partition (K=5) | ADASYN{evr_note}"),
        ("dirichlet", 0.5,  "adasyn",
         os.path.join(out_dir, "pca_dirichlet_alpha0.5_adasyn.png"),
         f"PCA Visualization — Dirichlet α=0.5 (K=5) | ADASYN{evr_note}"),
        ("dirichlet", 1.0,  "adasyn",
         os.path.join(out_dir, "pca_dirichlet_alpha1.0_adasyn.png"),
         f"PCA Visualization — Dirichlet α=1.0 (K=5) | ADASYN{evr_note}"),
        ("dirichlet", 5.0,  "adasyn",
         os.path.join(out_dir, "pca_dirichlet_alpha5.0_adasyn.png"),
         f"PCA Visualization — Dirichlet α=5.0 (K=5) | ADASYN{evr_note}"),
    ]

    timings = []
    for scheme, alpha, method, output_path, fig_title in schemes:
        scheme_label = "IID" if scheme == "iid" else f"Dirichlet α={alpha}"
        label = f"{scheme_label} | {method.upper()}"
        print(f"\n=== Generating PNG: {label} -> {output_path} ===")
        elapsed = generate_png(
            scheme=scheme,
            alpha=alpha,
            method=method,
            x_train=x_train,
            y_train=y_train,
            pca=pca,
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
            f"  {label:<32} | {os.path.basename(path):<40} | "
            f"{elapsed:6.1f}s | exists={exists} | {size_mb:.2f} MB"
        )
    print(
        f"\nPCA explained variance ratio: "
        f"PC1={evr[0]:.4f}  PC2={evr[1]:.4f}  sum={evr.sum():.4f}"
    )


if __name__ == "__main__":
    main()
