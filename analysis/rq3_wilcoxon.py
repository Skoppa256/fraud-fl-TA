#!/usr/bin/env python3
"""IID vs Non-IID test for RQ3 (BAB 4 §4.3.2, BAB 5).

    python analysis/rq3_wilcoxon.py [--seed N]

Reads results/shap_v2/shap_summary_v2.csv (frozen; read only), or with --seed N
results/seeds/seed<N>/shap_v2/shap_summary_v2.csv, and prints the
paired comparison of between-client weighted rank correlation
(`weighted_between`) under IID and Non-IID (Dirichlet). Writes nothing.

Cells: the main measurement (local background, the explainer of record:
linear / tree / sampled kernel), federated conditions only. A pair is one
(dataset, model, SMOTE arm) observed under both conditions; BAF has no IID run
with SMOTE, so it contributes tanpa SMOTE pairs only. That gives 30 pairs for
the six models, 15 of them for the KernelSHAP models (FFD, BERT, FedXGBllr).

Test: one-sided Wilcoxon signed-rank, H1 = IID > Non-IID, scipy defaults
(zero_method="wilcox", exact distribution when n <= 50 without ties/zeros).
Reported per group: pairs, pairs that decrease, mean decrease, W and p.
The 15-pair KernelSHAP group reproduces the values recorded in
docs/shap_rq3_run.md (11/15, mean 0.054, p = 0.0042).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "results" / "shap_v2" / "shap_summary_v2.csv"
MAIN_EXPLAINERS = {"linear", "tree", "kernel"}
KERNEL_MODELS = {"ffd", "bert_fraud", "fedxgbllr"}
GROUPS = {
    "all six models": None,
    "LR, SVM, GBM (LinearSHAP / TreeSHAP)": lambda m: m not in KERNEL_MODELS,
    "FFD, BERT, FedXGBllr (KernelSHAP)": lambda m: m in KERNEL_MODELS,
}


def pairs() -> pd.DataFrame:
    if not SUMMARY.is_file():
        raise SystemExit(f"FATAL: input CSV not found.\n  looked in: {SUMMARY}")
    df = pd.read_csv(SUMMARY)
    df = df[(df["bg"] == "local") & df["explainer"].isin(MAIN_EXPLAINERS)
            & df["condition"].isin(["iid", "dirichlet"])].copy()
    df["weighted_between"] = pd.to_numeric(df["weighted_between"], errors="raise")
    p = df.pivot_table(index=["dataset", "model", "arm"], columns="condition",
                       values="weighted_between", aggfunc="first")
    unpaired = p[p.isna().any(axis=1)]
    if len(unpaired):
        print(f"note: {len(unpaired)} cell(s) without an IID/Non-IID partner, "
              f"left out: {list(unpaired.index)}")
    return p.dropna()


def main() -> None:
    global SUMMARY
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=None,
                    help="read results/seeds/seed<N>/shap_v2/ instead of the seed-42 record")
    args = ap.parse_args()
    if args.seed is not None:
        SUMMARY = (ROOT / "results" / "seeds" / f"seed{args.seed}" / "shap_v2"
                   / "shap_summary_v2.csv")
    p = pairs()
    print(f"source: {SUMMARY.relative_to(ROOT)}")
    for name, keep in GROUPS.items():
        g = p if keep is None else p[[keep(m) for m in p.index.get_level_values("model")]]
        drop = g["iid"] - g["dirichlet"]
        res = wilcoxon(g["iid"], g["dirichlet"], alternative="greater")
        print(f"\n{name}")
        print(f"  pairs            {len(g)}")
        print(f"  decreases        {int((drop > 0).sum())} of {len(g)}")
        print(f"  mean decrease    {drop.mean():.4f}")
        print(f"  Wilcoxon W       {res.statistic:.0f}")
        print(f"  one-sided p      {res.pvalue:.4f}")


if __name__ == "__main__":
    main()
