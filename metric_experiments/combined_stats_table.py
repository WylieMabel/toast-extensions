import glob
import re

import pandas as pd
from scipy.stats import spearmanr, mannwhitneyu

DISSIMILARITY_METRICS = {
    "mse", "participation_ratio_change", "angular_dist",
    "kl_div", "top1_flip_pct", "entropy_delta",
}
NON_METRIC_COLS = {
    "rank", "span", "true_accuracy", "p_value_vs_top", "accuracy_gap_vs_top",
    "gap_tolerance", "safe",
}

rho_rows, rho_p_rows = {}, {}
auc_rows, auc_p_rows = {}, {}
ir_rows = {}

for path in sorted(glob.glob("metric_experiments/grouped_metrics_*.csv")):
    m = re.match(r"metric_experiments/grouped_metrics_(.+?)_(.+)\.csv", path)
    model, tag = m.group(1), m.group(2)
    key = f"{model}_{tag}"

    df = pd.read_csv(path)
    y_safe = df["safe"] == "safe"
    n_safe, n_unsafe = y_safe.sum(), (~y_safe).sum()
    n_total = len(df)

    metrics = [c for c in df.columns if c not in NON_METRIC_COLS]
    rho_rows[key], rho_p_rows[key] = {}, {}
    auc_rows[key], auc_p_rows[key] = {}, {}
    ir_rows[key] = {}

    for metric in metrics:
        vals = df[metric]
        if vals.isna().all():
            continue

        # Spearman rank correlation vs true_accuracy (reported as-is, no sign flip)
        rho, p_rho = spearmanr(vals, df["true_accuracy"])
        rho_rows[key][metric] = rho
        rho_p_rows[key][metric] = p_rho

        adjusted = -vals if metric in DISSIMILARITY_METRICS else vals

        # AUC for predicting the "safe" (vs top-ranked span) binary label,
        # direction-corrected so higher = more predictive of safe.
        if n_safe > 0 and n_unsafe > 0:
            u_stat, p_auc = mannwhitneyu(
                adjusted[y_safe], adjusted[~y_safe], alternative="greater"
            )
            auc_rows[key][metric] = u_stat / (n_safe * n_unsafe)
            auc_p_rows[key][metric] = p_auc

        # "IR" = precision at the rank needed for 100% recall of the safe spans, i.e.
        # n_safe / (rank of the worst-placed safe span in this metric's own ordering).
        # Answers "how deep would you need to screen by this metric to catch every safe
        # span". Sensitive to a single outlier (the worst-ranked safe span) and has no
        # attached significance test -- report next to AUC, not instead of it.
        if n_safe > 0:
            order = adjusted.sort_values(ascending=False).index
            metric_rank = pd.Series(range(1, n_total + 1), index=order)
            last_safe_rank = metric_rank[df.index[y_safe]].max()
            ir_rows[key][metric] = n_safe / last_safe_rank

rho_df = pd.DataFrame(rho_rows).T
rho_p_df = pd.DataFrame(rho_p_rows).T
auc_df = pd.DataFrame(auc_rows).T
auc_p_df = pd.DataFrame(auc_p_rows).T
ir_df = pd.DataFrame(ir_rows).T

all_metrics = sorted(set(rho_df.columns) | set(auc_df.columns) | set(ir_df.columns))
all_models = rho_df.index

combined = pd.DataFrame(index=all_models, columns=all_metrics, dtype=object)
for idx in all_models:
    for metric in all_metrics:
        rho = rho_df.loc[idx, metric] if metric in rho_df.columns else float("nan")
        p_rho = rho_p_df.loc[idx, metric] if metric in rho_p_df.columns else float("nan")
        auc = auc_df.loc[idx, metric] if metric in auc_df.columns else float("nan")
        p_auc = auc_p_df.loc[idx, metric] if metric in auc_p_df.columns else float("nan")
        ir = ir_df.loc[idx, metric] if metric in ir_df.columns else float("nan")

        if pd.isna(rho) and pd.isna(auc) and pd.isna(ir):
            combined.loc[idx, metric] = ""
            continue

        rho_str = f"rho={rho:.2f}{'*' if pd.notna(p_rho) and p_rho < 0.05 else ''}" if pd.notna(rho) else "rho=n/a"
        auc_str = f"AUC={auc:.2f}{'*' if pd.notna(p_auc) and p_auc < 0.05 else ''}" if pd.notna(auc) else "AUC=n/a"
        ir_str = f"IR={ir:.2f}" if pd.notna(ir) else "IR=n/a"
        combined.loc[idx, metric] = f"{rho_str} | {auc_str} | {ir_str}"

combined.to_csv("metric_experiments/combined_correlation_safety_stats.csv")

# Also keep the raw numeric tables around for anyone who wants to compute on them directly.
rho_df.to_csv("metric_experiments/combined_stats_spearman_rho.csv")
rho_p_df.to_csv("metric_experiments/combined_stats_spearman_pvalues.csv")
auc_df.to_csv("metric_experiments/combined_stats_safe_auc.csv")
auc_p_df.to_csv("metric_experiments/combined_stats_safe_auc_pvalues.csv")
ir_df.to_csv("metric_experiments/combined_stats_ir_precision.csv")

print("(* = p < 0.05; rho = Spearman vs true_accuracy, AUC = predicting safe/not-safe vs top span,")
print(" IR = precision at rank needed for 100% recall of safe spans, no significance test)\n")
print(combined.to_string())
