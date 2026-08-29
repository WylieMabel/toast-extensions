#!/usr/bin/env python3
"""
Process transfer learning experiments:
1. Combine results by model (excluding skip00)
2. Apply calculate_accuracies to compute mean/std per configuration
3. Pivot data so each row represents a configuration, with columns for different (dataset, fit_dataset, translator) combinations
"""

import os
import re
from pathlib import Path
from collections import defaultdict
import pandas as pd
import numpy as np

# Paths
RESULTS_DIR = Path("results")
INPUT_DIR = Path("transfer_learning_experiments")
OUTPUT_DIR = Path("transfer_learning_experiments/pivoted")

# Create output directory
OUTPUT_DIR.mkdir(exist_ok=True)


def _config_key(row):
    """Create a configuration key for grouping, excluding dataset and fit_dataset."""
    key = (f"Encoder: {row['model']}, "
           f"Block: {row.get('approx_layer', '')}, Block Mode: {row.get('translator', '')}, "
           f"MLP: {row.get('mlp_linearize', '')}, MLP Mode: {row.get('mlp_mode', '')}, "
           f"Attn: {row.get('attn_linearize', '')}, Heads: {row.get('head_dict', '')}")
    return key


def calculate_accuracies(df):
    """
    Group by configuration and compute mean/std of accuracy.
    Returns dataframe with config_key, accuracy_mean, accuracy_std.
    """
    df = df.copy()
    df["config_key"] = df.apply(_config_key, axis=1)

    # Group by config and compute statistics
    stats = df.groupby("config_key").agg({
        "accuracy": ["mean", "std"],
        "model": "first"
    }).reset_index()

    stats.columns = ["config_key", "accuracy_mean", "accuracy_std", "model"]
    return stats


def create_pivot_table(df, model_name):
    """
    Pivot the dataframe so columns represent different (dataset, fit_dataset, translator) combinations.
    Each row is a unique configuration.
    """
    df = df.copy()

    # Create column names for each (dataset, fit_dataset, translator) combination
    df["dataset_fit_translator"] = df.apply(
        lambda row: f"{row['translator']}_{row['dataset']}_{row['fit_dataset']}",
        axis=1
    )

    # Create base config (excluding dataset/fit_dataset/translator)
    def base_config_key(row):
        return (f"Encoder: {row['model']}, "
                f"Block: {row.get('approx_layer', '')}, "
                f"MLP: {row.get('mlp_linearize', '')}, MLP Mode: {row.get('mlp_mode', '')}, "
                f"Attn: {row.get('attn_linearize', '')}, Heads: {row.get('head_dict', '')}")

    df["base_config"] = df.apply(base_config_key, axis=1)

    # Pivot for accuracies and stds
    pivot_data = []
    base_configs = df["base_config"].unique()

    for base_config in sorted(base_configs):
        config_data = df[df["base_config"] == base_config]
        row_dict = {"config": base_config}

        # Collect accuracy and std for each (dataset, fit_dataset, translator) combo
        combos = config_data[["translator", "dataset", "fit_dataset", "accuracy"]].drop_duplicates()

        for _, combo_row in combos.iterrows():
            translator = combo_row["translator"]
            dataset = combo_row["dataset"]
            fit_dataset = combo_row["fit_dataset"]
            accuracy = combo_row["accuracy"]

            # Create column names
            if dataset == fit_dataset:
                # Same dataset control
                col_name = f"{translator}_{dataset}"
            else:
                # Transfer learning
                col_name = f"{translator}_{dataset}_fit{fit_dataset}"

            row_dict[f"{col_name}_acc"] = accuracy

        pivot_data.append(row_dict)

    pivot_df = pd.DataFrame(pivot_data)

    # Reorganize columns: accuracies first, then stds (if we had them)
    acc_cols = sorted([c for c in pivot_df.columns if c.endswith("_acc")])
    final_cols = ["config"] + acc_cols

    return pivot_df[final_cols]


# Process each model
for input_file in sorted(INPUT_DIR.glob("*.csv")):
    if input_file.is_dir():
        continue

    model_name = input_file.stem
    print(f"\nProcessing {model_name}...")

    # Read the combined results
    df = pd.read_csv(input_file)
    print(f"  Loaded {len(df)} rows")

    # Apply calculate_accuracies for grouped statistics
    acc_stats = calculate_accuracies(df)
    print(f"  Computed statistics for {len(acc_stats)} configurations")

    # Save accuracy statistics
    stats_output = OUTPUT_DIR / f"{model_name}_accuracies.csv"
    acc_stats.to_csv(stats_output, index=False)
    print(f"  Saved accuracy statistics to {stats_output}")

    # Create pivot table
    pivot_df = create_pivot_table(df, model_name)
    print(f"  Created pivot table with {len(pivot_df)} rows")

    # Save pivot table
    pivot_output = OUTPUT_DIR / f"{model_name}_pivoted.csv"
    pivot_df.to_csv(pivot_output, index=False)
    print(f"  Saved pivoted table to {pivot_output}")

print(f"\nAll processed results saved to {OUTPUT_DIR}/")
