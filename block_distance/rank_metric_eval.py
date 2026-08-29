"""
Evaluate how well different metrics predict true accuracy drop via:
  - Spearman rank correlation
  - Precision@5: of the top-5 safest predicted, how many are actually in top-5 safest
  - Recall@5: of the top-5 actually safest, how many did the metric identify

Metrics are ranked as-is (no sign flipping). Negative correlation is informative.

Usage:
    python block_distance/rank_metric_eval.py --csv output_level_eval_vitlarge_imagenet1k.csv \
        --true-col true_delta_acc --metric-cols kl_div,top1_flip_pct,knn_overlap@5,cosine_sim,mse,cka_embed
"""

import argparse
import csv

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def precision_recall_at_k(true_acc_drop, metric_scores, metric_name=None, k=5):
    """
    Precision@k: fraction of top-k predicted-safest that are in top-k true-safest.
    Recall@k: fraction of top-k true-safest that were identified by the metric.

    For similarity metrics (cosine_sim, cka_embed, knn_overlap@5) where higher = safer,
    we flip the sign so ascending rank still means "safest".
    """
    similarity_metrics = {'cosine_sim', 'cka_embed', 'knn_overlap@5', 'local_cka'}

    metric_scores = np.asarray(metric_scores).copy()
    if metric_name in similarity_metrics:
        metric_scores = -metric_scores  # Flip so ascending rank = safest

    if len(true_acc_drop) < k:
        k = len(true_acc_drop)

    # Top-k true-safest (lowest true_acc_drop)
    true_indices_sorted = np.argsort(true_acc_drop)[:k]
    true_safest_set = set(true_indices_sorted)

    # Top-k metric-predicted safest (lowest metric_scores)
    metric_indices_sorted = np.argsort(metric_scores)[:k]
    predicted_safest_set = set(metric_indices_sorted)

    # Precision@k: overlap / k
    overlap = len(true_safest_set & predicted_safest_set)
    precision = overlap / k if k > 0 else 0.0

    # Recall@k: overlap / k (same denominator by definition)
    recall = overlap / k if k > 0 else 0.0

    return precision, recall


def run(csv_path, true_col, metric_cols):
    print(f"\nLoading {csv_path}...")
    df = pd.read_csv(csv_path)

    if true_col not in df.columns:
        raise ValueError(f"Column '{true_col}' not found in CSV. Columns: {df.columns.tolist()}")

    missing = [col for col in metric_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Columns not found: {missing}. Available: {df.columns.tolist()}")

    print(f"  {len(df)} rows, {len(metric_cols)} metric columns\n")
    print(f"{'Metric':<20} {'Spearman Corr':<15} {'Precision@5':<15} {'Recall@5':<15}")
    print("-" * 65)

    results = []
    for metric in metric_cols:
        corr, pval = spearmanr(df[metric], df[true_col])
        prec, rec = precision_recall_at_k(df[true_col].values, df[metric].values, metric_name=metric, k=5)

        results.append({
            "metric": metric,
            "spearman_corr": corr,
            "spearman_pval": pval,
            "precision@5": prec,
            "recall@5": rec,
        })

        print(f"{metric:<20} {corr:>14.4f} {prec:>14.4f} {rec:>14.4f}")

    print("\nDone.")
    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Evaluate metrics against true accuracy drop via Spearman, Precision@5, Recall@5"
    )
    p.add_argument("--csv", required=True, help="Input CSV path (e.g., output_level_eval_*.csv joined with true drop)")
    p.add_argument("--true-col", default="true_delta_acc", help="Column name for true accuracy drop")
    p.add_argument("--metric-cols", required=True, help="Comma-separated metric column names")
    args = p.parse_args()

    metric_cols = [col.strip() for col in args.metric_cols.split(",")]
    run(args.csv, args.true_col, metric_cols)
