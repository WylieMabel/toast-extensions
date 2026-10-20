"""
Confirms metric polarity (does a higher raw value mean "more similar to the
original / safer to drop"?), writes that table, applies a sign-correction so
every metric follows the same "high = similar = safe to drop" convention, and
recomputes Spearman rho (vs. true_accuracy) on the corrected values.

Polarity convention matches DISSIMILARITY_METRICS already used elsewhere in
this repo (metric_experiments/build_group_stats.py, combined_stats_table.py,
safe_prediction_auc.py) -- confirmed here by checking that every listed
metric's raw (uncorrected) rho against true_accuracy is negative, and every
similarity-style metric's raw rho is positive, in the actual data.
"""
import pandas as pd
from scipy.stats import spearmanr

IN_DIR = "final_folder/combined_metrics"
GROUPED_IN = f"{IN_DIR}/grouped_metrics_cifar100_imagenet1k.csv"

NON_METRIC_COLS = {
    "model", "dataset", "skip_from", "skip_to",
    "true_accuracy", "true_delta_acc", "rank",
}

# raw value increases with DISSIMILARITY (more different from the original) ->
# must be negated so "high" means "similar" like every other metric.
DISSIMILARITY_METRICS = {
    "kl_div", "top1_flip_pct", "entropy_delta",
    "mse", "participation_ratio_change", "angular_dist", "toast_linear_error",
}

grouped = pd.read_csv(GROUPED_IN)
metric_cols = [c for c in grouped.columns if c not in NON_METRIC_COLS]

# --- 1. polarity table, confirmed against the data itself ---------------
rows = []
for m in metric_cols:
    raw_rhos = []
    for (model, dataset), sub in grouped.groupby(["model", "dataset"]):
        vals = sub[m]
        if vals.notna().sum() < 3 or vals.std() == 0:
            continue
        rho, _ = spearmanr(vals, sub["true_accuracy"])
        if pd.notna(rho):
            raw_rhos.append(rho)
    median_raw_rho = pd.Series(raw_rhos).median() if raw_rhos else float("nan")
    is_dissimilarity = m in DISSIMILARITY_METRICS
    rows.append({
        "metric": m,
        "raw_high_means": "more different (dissimilar)" if is_dissimilarity else "more similar",
        "median_raw_rho_vs_true_accuracy": round(median_raw_rho, 3),
        "high_means_similar_raw": not is_dissimilarity,
        "correction_applied": "negated (-value)" if is_dissimilarity else "none",
    })

polarity_df = pd.DataFrame(rows).sort_values("metric").reset_index(drop=True)
polarity_df.to_csv(f"{IN_DIR}/metric_polarity_table.csv", index=False)
print(polarity_df.to_string(index=False))

# --- 2. corrected values: high = similar = recommended to drop ----------
corrected = grouped.copy()
for m in metric_cols:
    if m in DISSIMILARITY_METRICS:
        corrected[m] = -corrected[m]
corrected.to_csv(f"{IN_DIR}/grouped_metrics_cifar100_imagenet1k_corrected.csv", index=False)

# --- 3. recompute Spearman rho on corrected values -----------------------
rho_rows = {}
for (model, dataset), sub in corrected.groupby(["model", "dataset"]):
    key_df = grouped[(grouped.model == model) & (grouped.dataset == dataset)]
    order = key_df.index  # preserve original row order for readability, unused otherwise
    row = {}
    for m in metric_cols:
        vals = sub[m]
        if vals.notna().sum() < 3 or vals.std() == 0:
            row[m] = float("nan")
            continue
        rho, _ = spearmanr(vals, sub["true_accuracy"])
        row[m] = rho
    rho_rows[(model, dataset)] = row

rho_df = pd.DataFrame(rho_rows).T
rho_df.index = pd.MultiIndex.from_tuples(rho_df.index, names=["model", "dataset"])
rho_df = rho_df.reset_index()
rho_df.to_csv(f"{IN_DIR}/combined_spearman_rho_cifar100_imagenet1k_corrected.csv", index=False)
print()
print(rho_df.to_string(index=False))
