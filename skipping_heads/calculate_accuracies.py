# %%
import argparse

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def _config_key(row):
    key = (f"Encoder: {row['model']}, Dataset: {row['dataset']}, "
           f"Block: {row['approx_layer']}, Block Mode: {row['translator']}, "
           f"MLP: {row['mlp_linearize']}, MLP Mode: {row['mlp_mode']}, "
           f"Attn: {row['attn_linearize']} Heads: {row['head_dict']}")

    # Transfer rows differ from their same-dataset control ONLY in fit_dataset -- same model,
    # dataset, span and translator. Without this the four transfer sources collapse into one
    # group and get averaged into a single meaningless number.
    #
    # Appended only when it actually differs, so keys for ordinary runs are unchanged and stay
    # comparable with previously generated summaries.
    fit = row.get("fit_dataset")
    if isinstance(fit, str) and fit and fit != row["dataset"]:
        key += f", Translator fit on: {fit}"
    return key


def accuracies(input_path, output_path):
    tab = pd.read_csv(input_path)
    tab.reset_index(inplace=True)
    tab["skip_config"] = tab.apply(_config_key, axis=1)
    accuracies = tab[["skip_config", "seed", "accuracy","index"]].groupby(["skip_config"]).agg({"accuracy": ["mean", "std"], "index": ["first"]}).reset_index().sort_values(by=("index", "first"), ascending=True)
    accuracies.columns = ["skip_config", "accuracy_mean", "accuracy_std", "order"]
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
