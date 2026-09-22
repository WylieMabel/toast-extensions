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


def precision_at_k(true_col, metric_scores, metric_name=None, k=5, true_higher_is_better=False):
    """
    Precision@k: fraction of top-k predicted-safest that are in top-k true-safest.

    For similarity metrics (cosine_sim, cka_embed, knn_overlap@5) where higher = safer,
    we flip the sign so ascending rank still means "safest".

    true_higher_is_better: False for true_delta_acc (lower drop = safer, the old default);
    True for raw true_accuracy (higher accuracy = safer). Passing the wrong value here
    silently inverts which rows count as "true safest" -- Spearman would just flip sign and
    still be readable, but precision@k has no such self-correcting property.
    """
    similarity_metrics = {'cosine_sim', 'cka_embed', 'knn_overlap@5', 'knn_overlap@10', 'local_cka',
                           'orig_head_accuracy', 'probe_frozen_accuracy', 'probe_refit_accuracy',
                           'probe_baseline_accuracy'}

    metric_scores = np.asarray(metric_scores).copy()
    if metric_name in similarity_metrics:
        metric_scores = -metric_scores  # Flip so ascending rank = safest

    true_col = np.asarray(true_col).copy()
    if true_higher_is_better:
        true_col = -true_col  # Flip so ascending rank = safest, same convention as metric_scores

    if len(true_col) < k:
        k = len(true_col)

    true_indices_sorted = np.argsort(true_col)[:k]
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
    p.add_argument("--truth-orig-head-col", default="orig_head_accuracy_mean",
                   help="truth CSV column holding the frozen-original-head accuracy (no "
                        "retraining) for this skip -- joined in as its own 'orig_head_accuracy' "
                        "metric so you can see how well the cheap frozen-head accuracy predicts "
                        "the expensive retrained true_delta_acc. Pass '' to skip this.")
    p.add_argument("--output", default="metric_experiments/joined_metrics_eval.csv",
                   help="output CSV with all metrics + true drop")
    p.add_argument("--use-raw-accuracy", action="store_true",
                   help="correlate against raw true_accuracy (higher=safer) instead of "
                        "true_delta_acc (lower=safer). Mathematically equivalent for Spearman "
                        "(|corr| unchanged, sign flips) but more directly interpretable -- "
                        "'higher KL divergence, lower accuracy' reads naturally where 'higher "
                        "KL divergence, higher accuracy drop' needs a mental double-negative.")
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

        # Extract true accuracy and delta -- both always computed and both written to the
        # joined CSV for reference; args.use_raw_accuracy only picks which one drives the
        # Spearman/precision@k stats below.
        skip_acc = truth_match[args.truth_acc_col].iloc[0]
        true_accuracy = skip_acc
        true_delta_acc = (baseline_acc - skip_acc) if baseline_acc is not None else skip_acc

        # Build joined row
        joined_row = dict(eval_row)
        joined_row["true_accuracy"] = true_accuracy
        joined_row["true_delta_acc"] = true_delta_acc

        # Frozen-head accuracy on this skip's own embeddings, joined in as an ordinary
        # "metric" column so it gets the same Spearman/precision@k treatment as everything
        # else -- lets you compare the cheap no-retraining accuracy signal against the
        # expensive retrained one it's meant to approximate.
        if args.truth_orig_head_col and args.truth_orig_head_col in truth_match.columns:
            orig_head_val = truth_match[args.truth_orig_head_col].iloc[0]
            if pd.notna(orig_head_val):
                joined_row["orig_head_accuracy"] = orig_head_val

        joined_rows.append(joined_row)

    print(f"\n[3/3] Joining and evaluating metrics...")
    print(f"      Matched {len(joined_rows)} skips from eval CSV to truth CSV")

    joined_df = pd.DataFrame(joined_rows)

    # Write joined CSV
    import os
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    joined_df.to_csv(args.output, index=False)
    print(f"      Joined CSV written to: {args.output}")

    # Compute Spearman and Precision@3/@5 for each metric, against whichever true-value
    # column was requested. Both true_accuracy and true_delta_acc are always in joined_df
    # (see above) -- this only picks which one the stats below are computed against.
    true_col_name = "true_accuracy" if args.use_raw_accuracy else "true_delta_acc"
    print(f"      Correlating against: {true_col_name}")
    metric_cols = [col for col in joined_df.columns
                   if col not in ["skip_from", "skip_to", "true_accuracy", "true_delta_acc"]]

    print(f"\n{'Metric':<20} {'Spearman Corr':<15} {'Precision@3':<15} {'Precision@5':<15}")
    print("-" * 65)

    results = []
    for metric in metric_cols:
        corr, pval = spearmanr(joined_df[metric], joined_df[true_col_name])
        prec3 = precision_at_k(
            joined_df[true_col_name].values,
            joined_df[metric].values,
            metric_name=metric,
            k=3,
            true_higher_is_better=args.use_raw_accuracy,
        )
        prec5 = precision_at_k(
            joined_df[true_col_name].values,
            joined_df[metric].values,
            metric_name=metric,
            k=5,
            true_higher_is_better=args.use_raw_accuracy,
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
