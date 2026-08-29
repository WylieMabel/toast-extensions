#!/usr/bin/env python3
"""
Combines transfer learning results by model, excluding skip00 files.
Removes duplicates and saves combined results to transfer_learning_experiments folder.
"""

import os
import re
from pathlib import Path
import pandas as pd
from collections import defaultdict

# Paths
RESULTS_DIR = Path("results")
OUTPUT_DIR = Path("transfer_learning_experiments")

# Create output directory if it doesn't exist
OUTPUT_DIR.mkdir(exist_ok=True)

# Find all transfer_learning result files
result_files = sorted(RESULTS_DIR.glob("results_transfer_learning_*.csv"))

# Group files by model name
models = defaultdict(list)
for file in result_files:
    # Extract model name and skip number
    # Pattern: results_transfer_learning_[model]_skip[number].csv
    match = re.match(r"results_transfer_learning_(.+)_skip(\d+)\.csv", file.name)
    if match:
        model_name = match.group(1)
        skip_num = int(match.group(2))

        # Skip skip00 files
        if skip_num != 0:
            models[model_name].append(file)

print(f"Found {len(models)} models with transfer learning results:")
for model, files in sorted(models.items()):
    print(f"  {model}: {len(files)} files")

# Combine results for each model
for model_name, files in sorted(models.items()):
    print(f"\nProcessing {model_name}...")

    # Read and combine all CSV files for this model
    dfs = []
    for file in sorted(files):
        try:
            df = pd.read_csv(file)
            dfs.append(df)
            print(f"  Loaded {file.name}: {len(df)} rows")
        except Exception as e:
            print(f"  Error loading {file.name}: {e}")

    if dfs:
        # Combine all dataframes
        combined_df = pd.concat(dfs, ignore_index=True)
        print(f"  Combined: {len(combined_df)} total rows")

        # Remove duplicates
        initial_count = len(combined_df)
        combined_df = combined_df.drop_duplicates()
        duplicates_removed = initial_count - len(combined_df)
        print(f"  Removed {duplicates_removed} duplicates: {len(combined_df)} unique rows")

        # Save combined results
        output_file = OUTPUT_DIR / f"{model_name}.csv"
        combined_df.to_csv(output_file, index=False)
        print(f"  Saved to {output_file}")

print(f"\nAll results combined and saved to {OUTPUT_DIR}/")
