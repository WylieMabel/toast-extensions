"""
Rebuilds the metric-reliability heatmaps in final_folder/combined_metrics/plots/ for
all three source groups (cifar100_imagenet1k, group1, group2), using polarity-corrected
metric values so that a positive rho / a low regret always means "this metric correctly
predicts a safe span to drop" for every metric, with no per-metric sign bookkeeping
needed by the reader.

The original notebook that produced these plots (final_folder/combined_metrics/
metric_display.ipynb) is empty on disk, so this reconstructs the methodology from
the surviving aggregate CSVs in this directory:
  - regret formula verified to reproduce regret_all_combos.csv exactly:
    rank spans by the metric (best-first, using the corrected "high=similar" sign),
    top1_regret = best true_accuracy in the combo - true_accuracy of the metric's #1 pick,
    top5_avg_regret = best true_accuracy - mean true_accuracy of the metric's top-5 picks.
  - tier ordering (task / global / local / probe) taken from metric_rank_summary.csv.
  - group1 / group2 only ever had one (combined, all datasets) heatmap each -- no
    per-dataset split -- matching the surviving *_group1.png / *_group2.png files.

regret_all_combos.csv and the cross-group ranking CSVs (metric_rank_summary*,
best_metric_by_model.csv) are left untouched: regret already used the direction-
corrected ranking upstream (verified), so those numbers don't change.
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
OUT_DIR = os.path.join(REPO, "final_folder", "combined_metrics")
PLOTS_DIR = os.path.join(OUT_DIR, "plots")

# raw value increases with DISSIMILARITY (more different from the original) -> must be
# negated so "high" means "similar", same convention as every metric.
DISSIMILARITY_METRICS = {
    "kl_div", "top1_flip_pct", "entropy_delta",
    "mse", "participation_ratio_change", "angular_dist", "toast_linear_error",
}

# computed from the original ImageNet classifier head's logits -- only ever populated for
# imagenet-1k rows, so they'd otherwise show as an all-white/no-data block for every other
# dataset in a multi-dataset heatmap.
IMAGENET_HEAD_METRICS = {"kl_div", "top1_flip_pct", "entropy_delta", "orig_head_accuracy"}

BASE_NON_METRIC_COLS = {"model", "dataset", "true_accuracy"}
# excluded: constant within every combo -> rho/regret undefined, never ranked upstream either.
EXCLUDE_FROM_RANKING = {"probe_baseline_accuracy"}

TIER = {
    "orig_head_accuracy": "task", "entropy_delta": "task", "kl_div": "task", "top1_flip_pct": "task",
    "cosine_sim": "global", "participation_ratio_change": "global", "knn_overlap@10": "global",
    "mse": "global", "knn_overlap@5": "global", "cka_embed": "global",
    "angular_dist": "local", "local_cka": "local", "toast_linear_error": "local",
    "probe_refit_accuracy": "probe", "probe_frozen_accuracy": "probe",
}
TIER_ORDER = {"task": 0, "global": 1, "local": 2, "probe": 3}

GROUPS = {
    "cifar100_imagenet1k": {
        "raw_csv": os.path.join(OUT_DIR, "grouped_metrics_cifar100_imagenet1k.csv"),
        "extra_non_metric": {"skip_from", "skip_to", "true_delta_acc", "rank"},
        "variants": [
            ("cifar100", "CIFAR-100", ("model",), lambda df: df[df["dataset"] == "cifar100"].drop(columns=["dataset"])),
            ("imagenet1k", "ImageNet-1k", ("model",), lambda df: df[df["dataset"] == "imagenet-1k"].drop(columns=["dataset"])),
            ("cifar100_imagenet1k_combined", "CIFAR-100 + ImageNet-1k", ("model", "dataset"), lambda df: df),
        ],
    },
    "group1": {
        "raw_csv": os.path.join(OUT_DIR, "grouped_metrics_group1.csv"),
        "extra_non_metric": {"source_file", "span", "rank", "p_value_vs_top", "accuracy_gap_vs_top", "gap_tolerance", "safe"},
        "variants": [
            ("group1", "Group 1", ("model", "dataset"), lambda df: df),
        ],
    },
    "group2": {
        "raw_csv": os.path.join(OUT_DIR, "grouped_metrics_group2.csv"),
        "extra_non_metric": {"source_file", "span", "rank", "p_value_vs_top", "accuracy_gap_vs_top", "gap_tolerance", "safe"},
        "variants": [
            ("group2", "Group 2", ("model", "dataset"), lambda df: df),
        ],
    },
}


def metric_order(metrics):
    return sorted(metrics, key=lambda m: (TIER_ORDER.get(TIER.get(m), 9), m))


def correct_polarity(df, non_metric_cols):
    corrected = df.copy()
    for m in corrected.columns:
        if m not in non_metric_cols and m in DISSIMILARITY_METRICS:
            corrected[m] = -corrected[m]
    return corrected


def compute_rho_and_regret(df, group_cols, non_metric_cols, exclude_metrics=frozenset()):
    """df has corrected metric columns (high=similar) + true_accuracy.
    group_cols: columns identifying one combo (e.g. ["model"] or ["model", "dataset"])."""
    metrics = metric_order(
        c for c in df.columns
        if c not in non_metric_cols and c not in EXCLUDE_FROM_RANKING and c not in exclude_metrics
    )
    rho_rows, regret_rows = [], []
    for combo_key, sub in df.groupby(group_cols):
        combo_key = combo_key if isinstance(combo_key, tuple) else (combo_key,)
        best_true = sub["true_accuracy"].max()
        rho_row, top1_row, top5_row = {}, {}, {}
        for m in metrics:
            vals = sub[m]
            if vals.notna().sum() < 3 or vals.std() == 0:
                rho_row[m] = top1_row[m] = top5_row[m] = np.nan
                continue
            rho_row[m] = sub[[m, "true_accuracy"]].dropna().corr(method="spearman").iloc[0, 1]
            ranked = sub.dropna(subset=[m]).sort_values(m, ascending=False)
            top1_row[m] = best_true - ranked["true_accuracy"].iloc[0]
            top5_row[m] = best_true - ranked["true_accuracy"].iloc[:5].mean()
        rho_rows.append({**dict(zip(group_cols, combo_key)), **rho_row})
        regret_rows.append({
            **dict(zip(group_cols, combo_key)),
            **{f"top1_regret::{m}": top1_row[m] for m in metrics},
            **{f"top5_regret::{m}": top5_row[m] for m in metrics},
        })
    rho_df = pd.DataFrame(rho_rows).set_index(group_cols)[metrics]
    top1_df = pd.DataFrame(regret_rows).set_index(group_cols)[[f"top1_regret::{m}" for m in metrics]]
    top1_df.columns = metrics
    top5_df = pd.DataFrame(regret_rows).set_index(group_cols)[[f"top5_regret::{m}" for m in metrics]]
    top5_df.columns = metrics
    return rho_df, top1_df, top5_df


def rank_rows(mat, ascending):
    """Per-row competition rank across metrics for that combo (1 = best), NaNs stay NaN."""
    return mat.rank(axis=1, ascending=ascending, method="min")


def _row_labels(index, group_cols):
    if len(group_cols) == 1:
        return [str(v).split("/")[-1] for v in index]
    return [f"{m.split('/')[-1]}_{d}" for m, d in index]


def _text_color(rgba):
    r, g, b = rgba[0], rgba[1], rgba[2]
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return "white" if luminance < 0.55 else "black"


def plot_heatmap(mat, title, out_path, cmap, fmt="{:.2f}", vmin=None, vmax=None, group_cols=("model",)):
    # orientation: rows = combo (model[/dataset]), cols = metric.
    fig_w = max(6, 0.85 * mat.shape[1] + 2)
    fig_h = max(4, 0.5 * mat.shape[0] + 1.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    values = mat.values.astype(float)
    cmap_obj = plt.get_cmap(cmap)
    lo = vmin if vmin is not None else np.nanmin(values)
    hi = vmax if vmax is not None else np.nanmax(values)
    norm = Normalize(vmin=lo, vmax=hi)
    im = ax.imshow(values, cmap=cmap_obj, norm=norm, aspect="auto")

    ax.set_xticks(range(mat.shape[1]))
    ax.set_xticklabels(mat.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(mat.shape[0]))
    ax.set_yticklabels(_row_labels(mat.index, group_cols), fontsize=8)

    for row in range(mat.shape[0]):
        for col in range(mat.shape[1]):
            v = mat.iloc[row, col]
            if pd.isna(v):
                continue
            color = _text_color(cmap_obj(norm(v)))
            ax.text(col, row, fmt.format(v), ha="center", va="center", fontsize=7, color=color)

    ax.set_title(title, fontsize=11)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def make_regret_plots(suffix, label, top1_df, top5_df, group_cols):
    rank_top1 = rank_rows(top1_df, ascending=True)
    rank_top5 = rank_rows(top5_df, ascending=True)

    plot_heatmap(top1_df, f"Accuracy Lost Trusting Each Metric's Top Pick — {label}",
                 os.path.join(PLOTS_DIR, f"regret_top1_heatmap_{suffix}.png"),
                 cmap="YlOrRd", vmin=0, group_cols=group_cols)
    plot_heatmap(top5_df, f"Accuracy Lost Trusting Each Metric's Top 5 Picks — {label}",
                 os.path.join(PLOTS_DIR, f"regret_top5_heatmap_{suffix}.png"),
                 cmap="YlOrRd", vmin=0, group_cols=group_cols)
    plot_heatmap(rank_top1, f"Rank by Top-Pick Accuracy Loss, 1 = Best — {label}",
                 os.path.join(PLOTS_DIR, f"rank_top1regret_heatmap_{suffix}.png"),
                 cmap="YlGnBu_r", fmt="{:.0f}", group_cols=group_cols)
    plot_heatmap(rank_top5, f"Rank by Top-5 Accuracy Loss, 1 = Best — {label}",
                 os.path.join(PLOTS_DIR, f"rank_top5regret_heatmap_{suffix}.png"),
                 cmap="YlGnBu_r", fmt="{:.0f}", group_cols=group_cols)


def make_plots(suffix, label, df, group_cols, non_metric_cols, exclude_metrics=frozenset()):
    rho_df, top1_df, top5_df = compute_rho_and_regret(
        df, list(group_cols), non_metric_cols, exclude_metrics=exclude_metrics
    )
    abs_rho = rho_df.abs()
    rank_rho = rank_rows(abs_rho, ascending=False)

    plot_heatmap(rho_df, f"Correlation with True Accuracy — {label}",
                 os.path.join(PLOTS_DIR, f"rho_heatmap_{suffix}.png"),
                 cmap="RdBu", vmin=-1, vmax=1, group_cols=group_cols)
    plot_heatmap(abs_rho, f"Correlation Strength with True Accuracy — {label}",
                 os.path.join(PLOTS_DIR, f"abs_rho_heatmap_{suffix}.png"),
                 cmap="YlGnBu", vmin=0, vmax=1, group_cols=group_cols)
    plot_heatmap(rank_rho, f"Metric Reliability Ranking, 1 = Best — {label}",
                 os.path.join(PLOTS_DIR, f"rank_rho_heatmap_{suffix}.png"),
                 cmap="YlGnBu_r", fmt="{:.0f}", group_cols=group_cols)
    make_regret_plots(suffix, label, top1_df, top5_df, group_cols)

    return rho_df, top1_df, top5_df


def main():
    rho_long_path = os.path.join(OUT_DIR, "rho_ranked_all_combos.csv")
    all_new_rows = []

    for source_group, cfg in GROUPS.items():
        raw = pd.read_csv(cfg["raw_csv"])
        non_metric_cols = BASE_NON_METRIC_COLS | cfg["extra_non_metric"]
        corrected = correct_polarity(raw, non_metric_cols)
        corrected.to_csv(
            os.path.join(OUT_DIR, f"grouped_metrics_{source_group}_corrected.csv"), index=False
        )

        # group1 / group2 never populate the ImageNet-head metrics at all (0 non-null rows,
        # not just some) -- drop them from every plot type for those two groups instead of
        # rendering an all-blank column.
        group_exclude = IMAGENET_HEAD_METRICS if source_group != "cifar100_imagenet1k" else frozenset()

        for suffix, label, group_cols, select in cfg["variants"]:
            sub = select(corrected)
            rho_df, _, _ = make_plots(
                suffix, label, sub, group_cols, non_metric_cols, exclude_metrics=group_exclude
            )

        # cifar100_imagenet1k_combined mixes cifar100 rows (no head metrics at all) with
        # imagenet-1k rows (which have them) -- an alt regret-only view with those 4 columns
        # dropped entirely, so every row/column is populated on equal footing.
        if source_group == "cifar100_imagenet1k":
            combined_df = corrected
            _, top1_nh, top5_nh = compute_rho_and_regret(
                combined_df, ["model", "dataset"], non_metric_cols, exclude_metrics=IMAGENET_HEAD_METRICS
            )
            make_regret_plots(
                "cifar100_imagenet1k_combined_no_imagenet_head",
                "CIFAR-100 + ImageNet-1k, shared metrics only",
                top1_nh, top5_nh, ("model", "dataset"),
            )

        # long-format rho rows keyed by (model, dataset) come from the full corrected group
        # (not a single-dataset variant), so ranks are computed across all of that group's metrics.
        full_rho_df, _, _ = compute_rho_and_regret(corrected, ["model", "dataset"], non_metric_cols)
        for (model, dataset), row in full_rho_df.iterrows():
            ranked = row.dropna().abs().sort_values(ascending=False)
            ranks = {m: i + 1 for i, m in enumerate(ranked.index)}
            for m, rho in row.items():
                if pd.isna(rho):
                    continue
                all_new_rows.append({
                    "source_group": source_group, "model": model, "dataset": dataset,
                    "metric": m, "rho": rho, "abs_rho": abs(rho), "rank": ranks[m],
                })

    updated = pd.DataFrame(all_new_rows)
    updated.to_csv(rho_long_path, index=False)
    print(f"Updated {rho_long_path} ({len(updated)} rows across {len(GROUPS)} groups)")
    print(f"Wrote plots to {PLOTS_DIR}")


if __name__ == "__main__":
    main()
