"""Stand-in s4 data built from the defunct group1 / group2 runs, so s4_metric_analysis.ipynb can be
developed before the s4 sweep finishes. Delete this folder (and set USE_TEMP = False in the
notebook) once the real s4 results are in.

Only the (model, dataset) combos that s4 is re-running are kept (the group1/group2 ImageNet-1k
combos are dropped), and within each combo only the spans in that combo's s4 configs. Files are
written with s4 names, so the notebook reads them exactly as it will read the real ones:
    accuracies_results_<alias>_<tag>.csv      <- see ACC_SOURCES
    grouped_metrics_s4_<alias>_<tag>.csv      <- metric_experiments/grouped_metrics_group{1,2}_<alias>_<dataset>.csv

Run from anywhere:
    python thesis/s4_medical_domain/analysis/temp_data/make_temp_data.py
"""

import ast
import glob
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
S4 = HERE.parents[1]
REPO = S4.parents[1]

# s4 alias -> (old group, hf id)
MODELS = {
    "dinov2base": ("group1", "facebook/dinov2-base"),
    "raddino": ("group1", "microsoft/rad-dino"),
    "raddinomaira2": ("group1", "microsoft/rad-dino-maira-2"),
    "vitlarge": ("group2", "google/vit-large-patch16-224"),
    "phikonv2": ("group2", "owkin/phikon-v2"),
}
# s4 file tag -> dataset key (as in the accuracy CSVs and the old group file names)
DATASETS = {
    "chestxray14": "chestxray14",
    "chestxraypneumonia": "chestxray_pneumonia",
    "patchcamelyon": "patchcamelyon",
    "tcgaut": "tcga_ut",
}
# final_folder's group2 CSVs only hold 45 of the 86 medical spans; results/group2_combined/ has all
# 86 (identical values where they overlap) and is what the old grouped_metrics were built from.
ACC_SOURCES = {
    "group1": str(REPO / "final_folder" / "final_accuracies" / "group1_{ds_key}.csv"),
    "group2": str(REPO / "results" / "group2_combined" / "accuracies_results_group2_{ds_key}.csv"),
}


def span_set(approx_layers):
    """Single contiguous spans only, as (start, end) tuples."""
    out = set()
    for s in approx_layers:
        parsed = ast.literal_eval(s)
        if len(parsed) == 1:
            out.add(tuple(parsed[0]))
    return out


def main():
    for alias, (group, hf_id) in MODELS.items():
        for tag, ds_key in DATASETS.items():
            configs = sorted(glob.glob(str(S4 / "configs" / f"experiments_{alias}_{tag}*.csv")))
            if not configs:
                continue  # not an s4 combo
            new_spans = set().union(*(span_set(pd.read_csv(c)["skip"]) for c in configs))

            acc = pd.read_csv(ACC_SOURCES[group].format(ds_key=ds_key))
            acc = acc[(acc["model"] == hf_id) & (acc["dataset"] == ds_key)]
            keep = acc["approx_layer"].apply(
                lambda s: s == "[]" or (len(p := ast.literal_eval(s)) == 1 and tuple(p[0]) in new_spans))
            acc = acc[keep]
            acc.to_csv(HERE / f"accuracies_results_{alias}_{tag}.csv", index=False)

            g = pd.read_csv(REPO / "metric_experiments" / f"grouped_metrics_{group}_{alias}_{ds_key}.csv")
            n_old = len(g)
            g = g[g["span"].apply(lambda s: ast.literal_eval(s) in new_spans)]
            g.to_csv(HERE / f"grouped_metrics_s4_{alias}_{tag}.csv", index=False)

            print(f"{alias:14s} {tag:19s} {len(g):3d}/{len(new_spans):3d} s4 spans "
                  f"(dropped {n_old - len(g)} old-only)  baseline={'yes' if (acc['approx_layer'] == '[]').any() else 'NO'}")


if __name__ == "__main__":
    main()
