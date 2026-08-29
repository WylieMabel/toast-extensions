# %%
import argparse

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# One row per unique config -- same fields a config CSV / results row already carries, so a
# summary row reads the same way as the raw data instead of needing a string parsed back apart.
# fit_dataset is included whenever the column exists: two transfer rows that only differ in
# fit_dataset (translator fitted on a different dataset than it's evaluated on) must NOT be
# averaged together as if they were the same config.
CONFIG_COLS = [
    "model", "dataset", "approx_layer", "translator",
    "mlp_linearize", "mlp_mode", "attn_linearize", "attn_mode", "head_dict",
]


def accuracies(input_path, output_path):
    tab = pd.read_csv(input_path)
    tab.reset_index(inplace=True)

    config_cols = CONFIG_COLS + (["fit_dataset"] if "fit_dataset" in tab.columns else [])

    # orig_head_accuracy (frozen pretrained head, no retraining) is a newer column and won't
    # exist in results CSVs written before it was added -- aggregated only when present so
    # older sweeps keep summarizing exactly as before.
    agg_cols = ["accuracy"]
    if "orig_head_accuracy" in tab.columns:
        agg_cols.append("orig_head_accuracy")

    # NaN in a groupby key drops the whole group in pandas -- fit_dataset is blank/NaN for
    # every ordinary (non-transfer) row, which would silently discard almost everything.
    tab[config_cols] = tab[config_cols].fillna("")

    accuracies = (
        tab[config_cols + ["seed", "index"] + agg_cols]
        .groupby(config_cols, dropna=False)
        .agg({**{c: ["mean", "std"] for c in agg_cols}, "index": ["first"]})
        .reset_index()
        .sort_values(by=("index", "first"), ascending=True)
    )
    flat_cols = list(config_cols)
    for c in agg_cols:
        flat_cols += [f"{c}_mean", f"{c}_std"]
    flat_cols += ["order"]
    accuracies.columns = flat_cols
    accuracies.to_csv(output_path, index=False)
    return accuracies


def __main__():
    path = "table3_medical_completed.csv"
    input_path = "results/results_" + path
    output_path = "skipping_heads/accuracies/accuracies_" + path
    p = argparse.ArgumentParser(description="Mean/std of accuracy per unique run")
    p.add_argument("--input", default=input_path, help="results CSV with per-seed rows")
    p.add_argument("--output", default=output_path, help="where to write the mean/std summary")
    args = p.parse_args()
    accuracies(args.input, args.output)


# Guarded so `accuracies` can be imported and reused (notebooks, other scripts) without this
# firing on import -- it used to run unconditionally, parsing whatever argv happened to be
# present, which made the module impossible to import from a notebook.
if __name__ == "__main__":
    __main__()

# %%
