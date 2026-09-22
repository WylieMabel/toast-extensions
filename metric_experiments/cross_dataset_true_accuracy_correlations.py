import itertools

import pandas as pd
from scipy.stats import spearmanr

DATASETS_PER_MODEL = {
    "vitlarge": ["imagenet-1k", "cifar10", "cifar100", "pathmnist"],
    "deitsmall": ["imagenet-1k", "cifar10", "cifar100", "pathmnist"],
    "vitsmall": ["imagenet-1k", "cifar10", "cifar100", "pathmnist"],
    "dinov2base": ["cifar10", "cifar100", "pathmnist"],  # no imagenet-1k head
}
DATASET_FILE_TAG = {
    "imagenet-1k": "multiskip_imagenet1k",
    "cifar10": "cifar10",
    "cifar100": "cifar100",
    "pathmnist": "pathmnist",
}

rows = []
for model, datasets in DATASETS_PER_MODEL.items():
    for a, b in itertools.combinations(datasets, 2):
        df_a = pd.read_csv(f"metric_experiments/grouped_metrics_{model}_{DATASET_FILE_TAG[a]}.csv")[["span", "true_accuracy"]]
        df_b = pd.read_csv(f"metric_experiments/grouped_metrics_{model}_{DATASET_FILE_TAG[b]}.csv")[["span", "true_accuracy"]]
        merged = df_a.merge(df_b, on="span", suffixes=("_a", "_b"))

        rho, pval = spearmanr(merged["true_accuracy_a"], merged["true_accuracy_b"])
        rows.append({
            "model": model,
            "pair": f"{a} vs {b}",
            "rho": rho,
            "pval": pval,
            "n": len(merged),
            "sig": pval < 0.05,
        })

out = pd.DataFrame(rows)
out.to_csv("metric_experiments/cross_dataset_true_accuracy_correlations.csv", index=False)
print(out.round(4).to_string(index=False))
