import glob
import re

import pandas as pd
from scipy.stats import mannwhitneyu

# Lower value = more "different from top" = less safe for these; direction is flipped
# so that in all cases a higher *adjusted* score should predict "safe".
DISSIMILARITY_METRICS = {
    "mse", "participation_ratio_change", "angular_dist",
    "kl_div", "top1_flip_pct", "entropy_delta",
}
NON_METRIC_COLS = {
    "rank", "span", "true_accuracy", "p_value_vs_top", "accuracy_gap_vs_top",
    "gap_tolerance", "safe",
}

rows_auc = {}
rows_pval = {}

for path in sorted(glob.glob("metric_experiments/grouped_metrics_*.csv")):
    m = re.match(r"metric_experiments/grouped_metrics_(.+?)_(.+)\.csv", path)
    model, tag = m.group(1), m.group(2)
    key = f"{model}_{tag}"

    df = pd.read_csv(path)
    y_safe = df["safe"] == "safe"
    n_safe, n_unsafe = y_safe.sum(), (~y_safe).sum()
    if n_safe == 0 or n_unsafe == 0:
        continue  # no contrast to test

    metrics = [c for c in df.columns if c not in NON_METRIC_COLS]
    rows_auc[key] = {}
    rows_pval[key] = {}

    for metric in metrics:
        vals = df[metric]
        if vals.isna().all():
            continue
        adjusted = -vals if metric in DISSIMILARITY_METRICS else vals
        safe_vals = adjusted[y_safe]
        unsafe_vals = adjusted[~y_safe]
        u_stat, p = mannwhitneyu(safe_vals, unsafe_vals, alternative="greater")
        auc = u_stat / (n_safe * n_unsafe)
        rows_auc[key][metric] = auc
        rows_pval[key][metric] = p

auc_df = pd.DataFrame(rows_auc).T
pval_df = pd.DataFrame(rows_pval).T

auc_df.to_csv("metric_experiments/safe_prediction_auc.csv")
pval_df.to_csv("metric_experiments/safe_prediction_auc_pvalues.csv")

# Annotated version for presenting: AUC with * if p<0.05
annotated_rows = {}
for idx in auc_df.index:
    annotated_rows[idx] = {}
    for col in auc_df.columns:
        auc = auc_df.loc[idx, col]
        p = pval_df.loc[idx, col] if col in pval_df.columns else float("nan")
        if pd.isna(auc):
            annotated_rows[idx][col] = ""
        else:
            star = "*" if pd.notna(p) and p < 0.05 else ""
            annotated_rows[idx][col] = f"{auc:.2f}{star}"
annotated = pd.DataFrame(annotated_rows).T.fillna("")
annotated.to_csv("metric_experiments/safe_prediction_auc_annotated.csv")

print("=== AUC for predicting 'safe' (direction-corrected: higher = more predictive of safe) ===")
print(annotated.to_string())

print("\n=== Mean AUC per metric across all model/dataset combos ===")
mean_auc = auc_df.mean(axis=0).sort_values(ascending=False)
n_sig = (pval_df < 0.05).sum(axis=0)
n_total = pval_df.count(axis=0)
summary = pd.DataFrame({
    "mean_auc": mean_auc,
    "n_significant": n_sig.reindex(mean_auc.index),
    "n_combos": n_total.reindex(mean_auc.index),
})
print(summary.to_string())
summary.to_csv("metric_experiments/safe_prediction_auc_summary.csv")
