import pandas as pd
from pathlib import Path

# Read all result files
results_dir = Path("/Users/mabelwylie/Documents/toast-extensions/results")
files = sorted(results_dir.glob("results_window_grid_part*.csv"))

print(f"Found {len(files)} files to combine:")
for f in files:
    print(f"  - {f.name}")

# Read and combine
dfs = [pd.read_csv(f) for f in files]
combined = pd.concat(dfs, ignore_index=True)

print(f"\nBefore deduplication: {len(combined)} rows")

# Remove duplicates
combined_dedup = combined.drop_duplicates()

print(f"After removing duplicates: {len(combined_dedup)} rows")
print(f"Removed {len(combined) - len(combined_dedup)} duplicate rows")

# Save combined file
output_file = results_dir / "results_window_grid_combined.csv"
combined_dedup.to_csv(output_file, index=False)
print(f"\nSaved to: {output_file}")
