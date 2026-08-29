"""
Join output_level_eval.csv with distance_analysis.csv (true accuracy drop),
then compute Spearman / Precision@5 / Recall@5 for all metrics.

Usage:
    python metric_experiments/join_and_evaluate.py \
        --eval-csv block_distance/output_level_eval_vitlarge_imagenet1k.csv \
        --truth-csv block_distance/distance_analysis.csv \
        --model google/vit-large-patch16-224 \
        --dataset imagenet-1k
"""

import argparse
import csv
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def precision_at_k(true_acc_drop, metric_scores, metric_name=None, k=5):
    """
    Precision@k: fraction of top-k predicted-safest that are in top-k true-safest.

    For similarity metrics (cosine_sim, cka_embed, knn_overlap@5) where higher = safer,
    we flip the sign so ascending rank still means "safest".
    """
    similarity_metrics = {'cosine_sim', 'cka_embed', 'knn_overlap@5', 'local_cka'}

    metric_scores = np.asarray(metric_scores).copy()
    if metric_name in similarity_metrics:
        metric_scores = -metric_scores  # Flip so ascending rank = safest

    if len(true_acc_drop) < k:
        k = len(true_acc_drop)

    true_indices_sorted = np.argsort(true_acc_drop)[:k]
    true_safest_set = set(true_indices_sorted)

    metric_indices_sorted = np.argsort(metric_scores)[:k]
    predicted_safest_set = set(metric_indices_sorted)

    overlap = len(true_safest_set & predicted_safest_set)
    precision = overlap / k if k > 0 else 0.0

    return precision


def main():
    p = argparse.ArgumentParser(
        description="Join eval CSV with truth CSV, compute Spearman/P@5/R@5 per metric"
    )
    p.add_argument("--eval-csv", required=True, help="output_level_eval_*.csv path")
    p.add_argument("--truth-csv", required=True,
                   help="distance_analysis.csv, or a skipping_heads/calculate_accuracies.py "
                        "summary CSV (pass --truth-layer-col approx_layer --truth-acc-col "
                        "accuracy_mean for the latter)")
    p.add_argument("--model", required=True, help="encoder HF id (e.g., google/vit-large-patch16-224)")
    p.add_argument("--dataset", required=True, help="dataset key (e.g., imagenet-1k)")
    p.add_argument("--truth-layer-col", default="approx_layer_str",
                   help="truth CSV column holding the skip span string, e.g. '[(0, 1)]' "
                        "(default matches distance_analysis.csv; use 'approx_layer' for a "
                        "calculate_accuracies.py summary CSV)")
    p.add_argument("--truth-acc-col", default="accuracy_mean_linear",
                   help="truth CSV column holding the accuracy to treat as ground truth "
                        "(default matches distance_analysis.csv; use 'accuracy_mean' for a "
                        "calculate_accuracies.py summary CSV)")
    p.add_argument("--output", default="metric_experiments/joined_metrics_eval.csv",
                   help="output CSV with all metrics + true drop")
    args = p.parse_args()

    print(f"\n{'=' * 70}")
    print(f"  Join & Evaluate Metrics")
    print(f"{'=' * 70}\n")

    # Load eval CSV (output_level_eval.py output)
    print(f"[1/3] Loading {args.eval_csv}...")
    eval_df = pd.read_csv(args.eval_csv)
    print(f"      {len(eval_df)} rows, columns: {list(eval_df.columns)}")

    # Load truth CSV (distance_analysis.csv) and filter
    print(f"[2/3] Loading {args.truth_csv}...")
    truth_df = pd.read_csv(args.truth_csv)

    # Filter for (model, dataset) pair
    # Note: distance_analysis uses HF id with slashes replaced by underscores
    model_stem = args.model.lower().replace("/", "_")
    truth_filtered = truth_df[
        (truth_df["model"].str.lower().str.replace("/", "_") == model_stem) &
        (truth_df["dataset"].str.lower() == args.dataset.lower())
    ].copy()
    print(f"      Filtered to {len(truth_filtered)} rows for {args.model} x {args.dataset}")

    if len(truth_filtered) == 0:
        print(f"  ERROR: No rows found in {args.truth_csv} for {args.model} x {args.dataset}")
        print(f"  Available (model, dataset) pairs:")
        for _, row in truth_df.iterrows():
            print(f"    {row['model']} x {row['dataset']}")
        sys.exit(1)

    # Extract true accuracy drop from baseline row (empty skip)
    baseline_row = truth_filtered[truth_filtered[args.truth_layer_col] == "[]"]
    if baseline_row.empty:
        print(f"  WARNING: No baseline row (empty skip) found. Using zero-drop as reference.")
        baseline_acc = None
    else:
        baseline_acc = baseline_row[args.truth_acc_col].iloc[0]
        print(f"      Baseline accuracy (empty skip): {baseline_acc:.6f}")

    # Parse the skip-span column for single-layer skips and join.
    # Single-layer skips are [(i, i+1)], which when converted to string look like "[(i, i+1)]"
    joined_rows = []
    for _, eval_row in eval_df.iterrows():
        skip_from = eval_row["skip_from"]
        skip_to = eval_row["skip_to"]

        # Find matching row in truth_filtered
        approx_layer_str = f"[({int(skip_from)}, {int(skip_to)})]"

        truth_match = truth_filtered[truth_filtered[args.truth_layer_col] == approx_layer_str]
        if truth_match.empty:
            print(f"  WARNING: No truth row found for skip [{skip_from}, {skip_to}). Skipping.")
            continue

        # Extract true accuracy drop
        skip_acc = truth_match[args.truth_acc_col].iloc[0]
        if baseline_acc is not None:
            true_delta_acc = baseline_acc - skip_acc
        else:
            true_delta_acc = skip_acc

        # Build joined row
        joined_row = dict(eval_row)
        joined_row["true_delta_acc"] = true_delta_acc
        joined_rows.append(joined_row)

    print(f"\n[3/3] Joining and evaluating metrics...")
    print(f"      Matched {len(joined_rows)} skips from eval CSV to truth CSV")

    joined_df = pd.DataFrame(joined_rows)

    # Write joined CSV
    import os
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    joined_df.to_csv(args.output, index=False)
    print(f"      Joined CSV written to: {args.output}")

    # Compute Spearman and Precision@3/@5 for each metric
    metric_cols = [col for col in joined_df.columns if col not in ["skip_from", "skip_to", "true_delta_acc"]]

    print(f"\n{'Metric':<20} {'Spearman Corr':<15} {'Precision@3':<15} {'Precision@5':<15}")
    print("-" * 65)

    results = []
    for metric in metric_cols:
        corr, pval = spearmanr(joined_df[metric], joined_df["true_delta_acc"])
        prec3 = precision_at_k(
            joined_df["true_delta_acc"].values,
            joined_df[metric].values,
            metric_name=metric,
            k=3
        )
        prec5 = precision_at_k(
            joined_df["true_delta_acc"].values,
            joined_df[metric].values,
            metric_name=metric,
            k=5
        )

        results.append({
            "metric": metric,
            "spearman_corr": corr,
            "spearman_pval": pval,
            "precision@3": prec3,
            "precision@5": prec5,
        })

        print(f"{metric:<20} {corr:>14.4f} {prec3:>14.4f} {prec5:>14.4f}")

    # Write results CSV
    results_csv = args.output.replace(".csv", "_stats.csv")
    with open(results_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["metric", "spearman_corr", "spearman_pval", "precision@3", "precision@5"]
        )
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults written to: {results_csv}")
    print(f"\nDone.")


if __name__ == "__main__":
    main()
