"""
Stages 2-4 of the output-level metric pipeline, for a whole experiment *group* in one shot.

Given the stage-1 per-span metric CSVs (output_level_eval.py output) and the matching
accuracy ground-truth CSVs, this:

  2. joins metrics <-> true accuracy per (model, dataset)               -> joined_metrics_<group>_<alias>_<tag>.csv (+ _stats.csv)
  3. reshapes to a rank table and adds the 1%-gap "safe" label          -> grouped_metrics_<group>_<alias>_<tag>.csv
  4. aggregates every combo into metric x combo matrices + a summary grid -> combined_stats_*_<group>.csv, combined_stats_corr_summary_<group>.png

The stage-4 PNG is a rho/AUC *summary* (which metric tracks accuracy where), NOT the
baseline pairwise angular-distance heatmap -- that one is layer_priority.py's <stem>_angdist.npy
plotted by analysis_plots.py; run src/run_scripts/run_baseline_pairwise.sh for it.

It is deliberately separate from add_safety_labels.py / combined_stats_table.py (which are
hardwired to the original group-2 backbone list and write unsuffixed files); this one is
config-driven and writes <group>-suffixed outputs so the two never collide.

Usage:
    python metric_experiments/build_group_stats.py --group group1
    python metric_experiments/build_group_stats.py --group group1 --eval-dir metric_experiments --skip-heatmap
"""

import argparse
import ast
import glob
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, mannwhitneyu, ttest_ind_from_stats

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

N_SEEDS = 3
GAP_TOLERANCE_FRAC = 0.01  # a span is "safe" if within 1% of the top span's accuracy_mean

# Higher raw value = *less* safe for these -> flip sign before Spearman / AUC so that in
# every case "higher adjusted score predicts safe". Matches combined_stats_table.py.
DISSIMILARITY_METRICS = {
    "mse", "participation_ratio_change", "angular_dist",
    "kl_div", "top1_flip_pct", "entropy_delta", "toast_linear_error",
}
NON_METRIC_COLS = {
    "rank", "span", "true_accuracy", "p_value_vs_top", "accuracy_gap_vs_top",
    "gap_tolerance", "safe",
}

# group -> {alias: hf_id}, list of (dataset_key, dataset_tag), truth-CSV template.
# dataset_tag is the hyphen-stripped form output_level_eval.py uses in filenames.
#
# "eval_stem" controls where each combo's stage-1 CSV is found:
#   output_level_eval_<eval_stem.format(group=, alias=, tag=)>.csv
# and "truth_csv" is formatted with the same three kwargs. group1's models all share one
# truth CSV per dataset (hence {tag}-only); the ad-hoc multiskip sweeps instead have a
# separate truth CSV per (alias, dataset) -- {alias} appears in both templates for that group.
GROUPS = {
    "group1": {
        "models": {
            "dinov2base": "facebook/dinov2-base",
            "raddino": "microsoft/rad-dino",
            "raddinomaira2": "microsoft/rad-dino-maira-2",
        },
        "datasets": [
            ("chestxray14", "chestxray14"),
            ("chestxray_pneumonia", "chestxray_pneumonia"),
            ("imagenet-1k", "imagenet1k"),
        ],
        "eval_stem": "{group}_{alias}_{tag}",
        # {tag} -> chestxray14 / chestxray_pneumonia / imagenet1k
        "truth_csv": os.path.join(REPO, "final_folder", "final_accuracies", "group1_{tag}.csv"),
        "truth_acc_col": "accuracy_mean",
        "truth_std_col": "accuracy_std",
        "truth_layer_col": "approx_layer",
    },
    "group2": {
        "models": {
            "vitlarge": "google/vit-large-patch16-224",
            "phikonv2": "owkin/phikon-v2",
        },
        "datasets": [
            ("imagenet-1k", "imagenet1k"),
            ("patchcamelyon", "patchcamelyon"),
            ("tcga_ut", "tcga_ut"),
        ],
        "eval_stem": "{group}_{alias}_{tag}",
        # group2's finalized, deduplicated truth now lives in final_folder/final_accuracies/
        # (previously results/group2_combined/, kept as a fallback reference only).
        "truth_csv": os.path.join(REPO, "final_folder", "final_accuracies", "group2_{tag}.csv"),
        "truth_acc_col": "accuracy_mean",
        "truth_std_col": "accuracy_std",
        "truth_layer_col": "approx_layer",
    },
    "multiskip": {
        # Ad-hoc output_level_eval.py runs from metric_experiments/run_output_level_metrics.sh's
        # single-combo (non-GROUP) mode, one per (alias, dataset) -- NOT produced via this
        # script's own GROUP loop, so the stem/truth-csv conventions differ from group1's.
        "models": {
            "deitsmall": "facebook/deit-small-patch16-224",
            "dinov2base": "facebook/dinov2-base",
            "vitlarge": "google/vit-large-patch16-224",
        },
        "datasets": [
            ("cifar100", "cifar100"),
            ("pathmnist", "pathmnist"),
        ],
        "eval_stem": "{alias}_multiskip_{tag}",
        # final_folder's nie_<tag>.csv holds all backbones combined for this dataset (unlike
        # the old per-(alias, dataset) results/ files); stage_join filters by hf_id already,
        # so the shared file works without needing {alias} in the template.
        "truth_csv": os.path.join(REPO, "final_folder", "final_accuracies", "nie_{tag}.csv"),
        "truth_acc_col": "accuracy_mean",
        "truth_std_col": "accuracy_std",
        "truth_layer_col": "approx_layer",
    },
    "s2": {
        # Thesis section s2 (contiguous skips of length 1-4). Stage 1 is launched per combo by
        # thesis/s2_skip_cost_metrics/launch_s2_metrics.sh; every output (stage-1 CSVs included)
        # lives in the section's analysis/ folder rather than metric_experiments/.
        "models": {
            "deitsmall": "facebook/deit-small-patch16-224",
            "vitsmall": "WinKawaks/vit-small-patch16-224",
            "dinov2base": "facebook/dinov2-base",
            "vitlarge": "google/vit-large-patch16-224",
        },
        "datasets": [
            ("cifar100", "cifar100"),
            ("imagenet-1k", "imagenet1k"),
        ],
        "eval_stem": "{group}_{alias}_{tag}",
        "truth_csv": os.path.join(REPO, "thesis", "s2_skip_cost_metrics", "accuracies",
                                  "accuracies_results_{alias}_{tag}.csv"),
        "truth_acc_col": "accuracy_mean",
        "truth_std_col": "accuracy_std",
        "truth_layer_col": "approx_layer",
        "out_dir": os.path.join(REPO, "thesis", "s2_skip_cost_metrics", "analysis"),
    },
    "s4": {
        # Thesis section s4 (medical domain, same skip list as s2). Not a full models x datasets
        # cross product: combos with no stage-1 CSV (e.g. phikonv2 x chestxray14) are skipped.
        # Truth is the per-combo file thesis/s4_medical_domain/combine_s4.py builds from the split
        # job outputs (_len1.._len4, _len12/_len34, _len1_rerun). truth_csv may also be a glob;
        # matched files are concatenated, later ones winning on duplicate spans.
        "models": {
            "dinov2base": "facebook/dinov2-base",
            "raddino": "microsoft/rad-dino",
            "raddinomaira2": "microsoft/rad-dino-maira-2",
            "vitlarge": "google/vit-large-patch16-224",
            "phikonv2": "owkin/phikon-v2",
        },
        # ChestX-ray14 is analysed as the original 14-label task only; the binary any-finding
        # runs (tag "chestxray14") are left out of s4.
        "datasets": [
            ("chestxray_pneumonia", "chestxraypneumonia"),
            ("chestxray14_multilabel", "chestxray14multilabel"),
            ("patchcamelyon", "patchcamelyon"),
            ("tcga_ut", "tcgaut"),
        ],
        "eval_stem": "{group}_{alias}_{tag}",
        "truth_csv": os.path.join(REPO, "thesis", "s4_medical_domain", "accuracies",
                                  "combined", "accuracies_results_{alias}_{tag}.csv"),
        # s4 is scored on macro-AUROC, not accuracy: the multilabel combos only have AUROC, and
        # on the weaker probes (dinov2 x chestxray14 / pneumonia) argmax accuracy's seed-to-seed
        # std is 10-20x AUROC's, so accuracy rankings there are mostly noise. "true_accuracy"
        # in the outputs therefore holds AUROC for s4.
        "truth_acc_col": "auroc_mean",
        "truth_std_col": "auroc_std",
        "truth_layer_col": "approx_layer",
        "out_dir": os.path.join(REPO, "thesis", "s4_medical_domain", "analysis"),
    },
}


def _span_key(approx_layer_str):
    """'[(3, 5)]' -> (3, 5); '[]' -> None; multi-span / disjoint -> None (unsupported here)."""
    parsed = ast.literal_eval(approx_layer_str)
    if not parsed or len(parsed) != 1:
        return None
    return tuple(parsed[0])


def stage_join(eval_csv, truth_csvs, hf_id, ds_key, cfg):
    """Join one combo's metric CSV with its accuracy ground truth (one or more CSVs, later ones
    winning on duplicate spans). Returns a DataFrame with one row per span:
    [span, true_accuracy, acc_std, <metric cols...>]."""
    ev = pd.read_csv(eval_csv)

    truth = pd.concat([pd.read_csv(p) for p in truth_csvs], ignore_index=True)
    truth = truth[
        (truth["model"].str.lower().str.replace("/", "_") == hf_id.lower().replace("/", "_"))
        & (truth["dataset"].str.lower() == ds_key.lower())
    ].drop_duplicates(subset=[cfg["truth_layer_col"]], keep="last").copy()
    if truth.empty:
        raise SystemExit(f"  no truth rows for {hf_id} x {ds_key} in {truth_csvs}")

    truth["span"] = truth[cfg["truth_layer_col"]].apply(_span_key)
    truth = truth.dropna(subset=["span"])
    acc = dict(zip(truth["span"], truth[cfg["truth_acc_col"]]))
    std = dict(zip(truth["span"], truth[cfg["truth_std_col"]]))

    metric_cols = [c for c in ev.columns if c not in ("skip_from", "skip_to")]
    rows = []
    for _, r in ev.iterrows():
        span = (int(r["skip_from"]), int(r["skip_to"]))
        if span not in acc:
            print(f"    warn: span {span} in eval CSV missing from truth -- skipped")
            continue
        row = {"span": f"({span[0]},{span[1]})", "true_accuracy": acc[span], "acc_std": std[span]}
        for m in metric_cols:
            row[m] = r[m]
        rows.append(row)
    return pd.DataFrame(rows)


def stage_grouped(joined, out_path):
    """Rank by true_accuracy (desc), add the 1%-gap safe label, write grouped_metrics CSV."""
    df = joined.sort_values("true_accuracy", ascending=False).reset_index(drop=True)
    df.insert(0, "rank", np.arange(1, len(df) + 1))

    top_mean = df["true_accuracy"].iloc[0]
    top_std = df["acc_std"].iloc[0]
    tolerance = top_mean * GAP_TOLERANCE_FRAC

    pvals, gaps, safe = [], [], []
    for _, r in df.iterrows():
        _, p = ttest_ind_from_stats(
            mean1=top_mean, std1=top_std, nobs1=N_SEEDS,
            mean2=r["true_accuracy"], std2=r["acc_std"], nobs2=N_SEEDS,
            equal_var=False, alternative="greater",
        )
        gap = top_mean - r["true_accuracy"]
        pvals.append(p)
        gaps.append(gap)
        safe.append("safe" if gap <= tolerance else "not safe")

    df = df.drop(columns=["acc_std"])
    df["p_value_vs_top"] = pvals
    df["accuracy_gap_vs_top"] = gaps
    df["gap_tolerance"] = tolerance
    df["safe"] = safe
    df.to_csv(out_path, index=False)
    return df


def stage_aggregate(group, grouped_entries, skip_heatmap, out_dir=HERE):
    """grouped_entries: list of (key, grouped_csv_path) -- key is the combo label used as the
    row index in every combined_stats_* output (e.g. "dinov2base_chestxray14")."""
    rho_rows, rho_p_rows, auc_rows, auc_p_rows, ir_rows = {}, {}, {}, {}, {}

    for key, path in sorted(grouped_entries):
        df = pd.read_csv(path)
        y_safe = df["safe"] == "safe"
        n_safe, n_unsafe, n_total = int(y_safe.sum()), int((~y_safe).sum()), len(df)

        metrics = [c for c in df.columns if c not in NON_METRIC_COLS]
        rho_rows[key], rho_p_rows[key] = {}, {}
        auc_rows[key], auc_p_rows[key] = {}, {}
        ir_rows[key] = {}

        for metric in metrics:
            vals = df[metric]
            if vals.isna().all():
                continue
            rho, p_rho = spearmanr(vals, df["true_accuracy"])
            rho_rows[key][metric] = rho
            rho_p_rows[key][metric] = p_rho

            adjusted = -vals if metric in DISSIMILARITY_METRICS else vals
            if n_safe and n_unsafe:
                u, p_auc = mannwhitneyu(adjusted[y_safe], adjusted[~y_safe], alternative="greater")
                auc_rows[key][metric] = u / (n_safe * n_unsafe)
                auc_p_rows[key][metric] = p_auc
            if n_safe:
                order = adjusted.sort_values(ascending=False).index
                metric_rank = pd.Series(range(1, n_total + 1), index=order)
                ir_rows[key][metric] = n_safe / metric_rank[df.index[y_safe]].max()

    rho_df = pd.DataFrame(rho_rows).T
    rho_p_df = pd.DataFrame(rho_p_rows).T
    auc_df = pd.DataFrame(auc_rows).T
    auc_p_df = pd.DataFrame(auc_p_rows).T
    ir_df = pd.DataFrame(ir_rows).T

    sfx = f"_{group}"
    rho_df.to_csv(os.path.join(out_dir, f"combined_stats_spearman_rho{sfx}.csv"))
    rho_p_df.to_csv(os.path.join(out_dir, f"combined_stats_spearman_pvalues{sfx}.csv"))
    auc_df.to_csv(os.path.join(out_dir, f"combined_stats_safe_auc{sfx}.csv"))
    auc_p_df.to_csv(os.path.join(out_dir, f"combined_stats_safe_auc_pvalues{sfx}.csv"))
    ir_df.to_csv(os.path.join(out_dir, f"combined_stats_ir_precision{sfx}.csv"))

    all_metrics = sorted(set(rho_df.columns) | set(auc_df.columns) | set(ir_df.columns))
    combined = pd.DataFrame(index=rho_df.index, columns=all_metrics, dtype=object)
    for idx in rho_df.index:
        for metric in all_metrics:
            rho = rho_df.loc[idx, metric] if metric in rho_df.columns else float("nan")
            p_rho = rho_p_df.loc[idx, metric] if metric in rho_p_df.columns else float("nan")
            auc = auc_df.loc[idx, metric] if metric in auc_df.columns else float("nan")
            p_auc = auc_p_df.loc[idx, metric] if metric in auc_p_df.columns else float("nan")
            ir = ir_df.loc[idx, metric] if metric in ir_df.columns else float("nan")
            if pd.isna(rho) and pd.isna(auc) and pd.isna(ir):
                combined.loc[idx, metric] = ""
                continue
            rho_s = f"rho={rho:.2f}{'*' if pd.notna(p_rho) and p_rho < 0.05 else ''}" if pd.notna(rho) else "rho=n/a"
            auc_s = f"AUC={auc:.2f}{'*' if pd.notna(p_auc) and p_auc < 0.05 else ''}" if pd.notna(auc) else "AUC=n/a"
            ir_s = f"IR={ir:.2f}" if pd.notna(ir) else "IR=n/a"
            combined.loc[idx, metric] = f"{rho_s} | {auc_s} | {ir_s}"
    combined.to_csv(os.path.join(out_dir, f"combined_correlation_safety_stats{sfx}.csv"))

    print("\n=== Spearman rho (metric vs true_accuracy), * = p<0.05 ===")
    print(combined.to_string())

    if not skip_heatmap:
        _heatmap(rho_df, rho_p_df, auc_df, auc_p_df, group, out_dir)
    return rho_df, auc_df


def _heatmap(rho_df, rho_p_df, auc_df, auc_p_df, group, out_dir=HERE):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [("Spearman rho vs true_accuracy", rho_df, rho_p_df, -1.0, 1.0, "RdBu_r"),
              ("AUC: predicting safe vs top span", auc_df, auc_p_df, 0.0, 1.0, "RdBu_r")]
    fig, axes = plt.subplots(1, 2, figsize=(1.1 * max(len(rho_df.columns), 6) + 4, 0.6 * len(rho_df) + 2.5))
    for ax, (title, dfv, dfp, vmin, vmax, cmap) in zip(np.atleast_1d(axes), panels):
        cols = list(dfv.columns)
        im = ax.imshow(dfv.values.astype(float), aspect="auto", vmin=vmin, vmax=vmax, cmap=cmap)
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels(cols, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(dfv.index)))
        ax.set_yticklabels(dfv.index, fontsize=8)
        ax.set_title(title, fontsize=10)
        for i in range(dfv.shape[0]):
            for j in range(dfv.shape[1]):
                v = dfv.values[i, j]
                if pd.isna(v):
                    continue
                p = dfp.values[i, j] if dfp.shape == dfv.shape else float("nan")
                star = "*" if pd.notna(p) and p < 0.05 else ""
                ax.text(j, i, f"{v:.2f}{star}", ha="center", va="center", fontsize=7)
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle(f"{group}: metric vs true accuracy -- correlation/AUC summary "
                 f"(NOT the baseline pairwise angular-distance heatmap)", fontsize=11)
    fig.tight_layout()
    out = os.path.join(out_dir, f"combined_stats_corr_summary_{group}.png")
    fig.savefig(out, dpi=200)
    print(f"\nheatmap -> {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--group", required=True, choices=sorted(GROUPS))
    ap.add_argument("--eval-dir", default=None,
                    help="directory holding output_level_eval_<group>_<alias>_<tag>.csv "
                         "(default: the group's out_dir, metric_experiments/ unless the group sets one)")
    ap.add_argument("--skip-heatmap", action="store_true")
    args = ap.parse_args()

    cfg = GROUPS[args.group]
    out_dir = cfg.get("out_dir", HERE)
    os.makedirs(out_dir, exist_ok=True)
    if args.eval_dir is None:
        args.eval_dir = out_dir
    eval_stem_tmpl = cfg.get("eval_stem", "{group}_{alias}_{tag}")
    grouped_entries = []
    for alias, hf_id in cfg["models"].items():
        for ds_key, tag in cfg["datasets"]:
            stem = eval_stem_tmpl.format(group=args.group, alias=alias, tag=tag)
            eval_csv = os.path.join(args.eval_dir, f"output_level_eval_{stem}.csv")
            if not os.path.exists(eval_csv):
                print(f"[skip] {stem}: no {eval_csv}")
                continue
            truth_pattern = cfg["truth_csv"].format(group=args.group, alias=alias, tag=tag)
            truth_csvs = sorted(glob.glob(truth_pattern))
            if not truth_csvs:
                print(f"[skip] {stem}: no truth csv {truth_pattern}")
                continue
            print(f"[{stem}] joining {os.path.basename(eval_csv)} <- "
                  f"{', '.join(os.path.basename(p) for p in truth_csvs)}")

            joined = stage_join(eval_csv, truth_csvs, hf_id, ds_key, cfg)
            joined_path = os.path.join(out_dir, f"joined_metrics_{stem}.csv")
            joined.drop(columns=["acc_std"]).to_csv(joined_path, index=False)

            grouped_path = os.path.join(out_dir, f"grouped_metrics_{stem}.csv")
            g = stage_grouped(joined, grouped_path)
            grouped_entries.append((f"{alias}_{tag}", grouped_path))
            print(f"    -> {os.path.basename(grouped_path)}  ({(g['safe'] == 'safe').sum()}/{len(g)} safe)")

    if not grouped_entries:
        raise SystemExit("no combos built -- run stage 1 (output_level_eval.py) first")
    stage_aggregate(args.group, grouped_entries, args.skip_heatmap, out_dir)


if __name__ == "__main__":
    main()
