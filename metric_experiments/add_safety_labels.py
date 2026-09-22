import ast
import pandas as pd
from scipy.stats import ttest_ind_from_stats

N_SEEDS = 3
ALPHA = 0.05
# Pure effect-size tolerance: a span counts as safe if it's within this fraction of the
# top span's accuracy, e.g. 0.01 = within 1% of top_mean (not 1 percentage point -- 1% of
# whatever top_mean happens to be, so the window scales with each combo's accuracy level).
GAP_TOLERANCE_FRAC = 0.01

MODEL_HF_ID = {
    "deitsmall": "facebook/deit-small-patch16-224",
    "vitsmall": "WinKawaks/vit-small-patch16-224",
    "vitlarge": "google/vit-large-patch16-224",
    "dinov2base": "facebook/dinov2-base",
}

# (grouped_metrics model, tag) -> (accuracies_results source file, model filter or None)
SOURCES = {
    ("deitsmall", "cifar10"): ("results/accuracies_results_deitsmall_cifar10_contiguous_multiskip.csv", None),
    ("deitsmall", "cifar100"): ("results/accuracies_results_deitsmall_cifar100_contiguous_multiskip.csv", None),
    ("deitsmall", "pathmnist"): ("results/accuracies_results_deitsmall_pathmnist_contiguous_multiskip.csv", None),
    ("deitsmall", "multiskip_imagenet1k"): ("results/accuracies_results_vitlarge_deitsmall_contiguous_multiskip.csv", "deitsmall"),
    ("vitsmall", "cifar10"): ("results/accuracies_results_vitsmall_cifar10_contiguous_multiskip.csv", None),
    ("vitsmall", "cifar100"): ("results/accuracies_results_vitsmall_cifar100_contiguous_multiskip.csv", None),
    ("vitsmall", "pathmnist"): ("results/accuracies_results_vitsmall_pathmnist_contiguous_multiskip.csv", None),
    ("vitsmall", "multiskip_imagenet1k"): ("results/accuracies_results_vitsmall_contiguous_multiskip.csv", None),
    ("vitlarge", "cifar10"): ("results/accuracies_results_vitlarge_cifar10_contiguous_multiskip.csv", None),
    ("vitlarge", "cifar100"): ("results/accuracies_results_vitlarge_cifar100_contiguous_multiskip.csv", None),
    ("vitlarge", "pathmnist"): ("results/accuracies_results_vitlarge_pathmnist_contiguous_multiskip.csv", None),
    ("vitlarge", "multiskip_imagenet1k"): ("results/accuracies_results_vitlarge_deitsmall_contiguous_multiskip.csv", "vitlarge"),
    ("dinov2base", "cifar10"): ("results/accuracies_results_dinov2base_cifar10_contiguous_multiskip.csv", None),
    ("dinov2base", "cifar100"): ("results/accuracies_results_dinov2base_cifar100_contiguous_multiskip.csv", None),
    ("dinov2base", "pathmnist"): ("results/accuracies_results_dinov2base_pathmnist_contiguous_multiskip.csv", None),
}


def load_spans(src_path, model_key):
    df = pd.read_csv(src_path)
    if model_key is not None:
        df = df[df["model"] == MODEL_HF_ID[model_key]]
    df = df.copy()
    df["span_parsed"] = df["approx_layer"].apply(ast.literal_eval)
    span_stats = {}
    for _, row in df[df["span_parsed"].apply(lambda s: s != [])].iterrows():
        span = tuple(row["span_parsed"][0])
        span_stats[span] = (row["accuracy_mean"], row["accuracy_std"])
    return span_stats


for (model, tag), (src_path, model_key) in SOURCES.items():
    gm_path = f"metric_experiments/grouped_metrics_{model}_{tag}.csv"
    span_stats = load_spans(src_path, model_key)

    # Reference = top-ranked (highest true_accuracy) skip span, not the no-skip baseline.
    # Since it's the max by construction, a two-sided test against it can only ever flag
    # "significantly worse" -- never "significantly better" -- so there's no risk of the
    # earlier bug where an improvement over baseline got mislabeled "not safe".
    top_span = max(span_stats, key=lambda s: span_stats[s][0])
    top_mean, top_std = span_stats[top_span]

    gm = pd.read_csv(gm_path)
    gm = gm.drop(columns=[c for c in (
        "p_value_vs_baseline", "p_value_vs_top", "accuracy_gap_vs_top",
        "gap_tolerance", "safe",
    ) if c in gm.columns])
    gm["span_parsed"] = gm["span"].apply(ast.literal_eval)
    assert tuple(gm.loc[gm["rank"] == 1, "span_parsed"].iloc[0]) == top_span, (
        f"{gm_path}: rank-1 row doesn't match the max-accuracy span from {src_path}"
    )

    tolerance = top_mean * GAP_TOLERANCE_FRAC

    pvals, gaps, safe_labels = [], [], []
    for span in gm["span_parsed"]:
        mean, std = span_stats[span]
        # Kept for reference/comparison, no longer drives the "safe" label.
        stat, p = ttest_ind_from_stats(
            mean1=top_mean, std1=top_std, nobs1=N_SEEDS,
            mean2=mean, std2=std, nobs2=N_SEEDS,
            equal_var=False,
            alternative="greater",
        )
        pvals.append(p)
        gap = top_mean - mean
        gaps.append(gap)
        safe_labels.append("safe" if gap <= tolerance else "not safe")

    gm["p_value_vs_top"] = pvals
    gm["accuracy_gap_vs_top"] = gaps
    gm["gap_tolerance"] = tolerance
    gm["safe"] = safe_labels
    gm = gm.drop(columns=["span_parsed"])
    gm.to_csv(gm_path, index=False)

    n_safe = safe_labels.count("safe")
    print(f"{model:11s} {tag:20s} top={top_mean:.4f} tol={tolerance:.4f}  "
          f"safe={n_safe}/{len(safe_labels)}")
