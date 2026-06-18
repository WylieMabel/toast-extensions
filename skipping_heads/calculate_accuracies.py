# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def accuracies(input_path, output_path):
    tab = pd.read_csv(input_path)
    tab.reset_index(inplace=True)
    tab["skip_config"] = tab.apply(lambda row: f"Encoder: {row['model']}, Dataset: {row['dataset']}, Block: {row['approx_layer']}, Block Mode: {row['translator']}, MLP: {row['mlp_linearize']}, MLP Mode: {row['mlp_mode']}, Attn: {row['attn_linearize']} Heads: {row['head_dict']}", axis=1)
    accuracies = tab[["skip_config", "seed", "accuracy","index"]].groupby(["skip_config"]).agg({"accuracy": ["mean", "std"], "index": ["first"]}).reset_index().sort_values(by=("index", "first"), ascending=True)
    accuracies.columns = ["skip_config", "accuracy_mean", "accuracy_std", "order"]
    accuracies.to_csv(output_path, index=False)
    return accuracies


def __main__():
    path = "cifar_heads.csv"
    input_path = "skipping_heads/results/results_" + path
    output_path = "skipping_heads/accuracies/accuracies_" + path
    accuracies(input_path, output_path)

__main__()

# %%
