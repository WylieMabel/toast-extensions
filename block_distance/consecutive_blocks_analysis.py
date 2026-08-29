import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# Load the processed distance analysis data
df = pd.read_csv("distance_analysis.csv")

# Filter for consecutive block skips: skip_count == 2 and skip_distance == 1
# (skip_distance = 1 means blocks are adjacent, e.g., (0,1), (1,2), etc.)
consecutive_df = df[
    (df["skip_count"] == 2) & (df["skip_distance"] == 1)
].copy()

if consecutive_df.empty:
    print("No data found for consecutive block skips (skip_count == 2, skip_distance == 1)")
    exit()

print(f"Found {len(consecutive_df)} configurations for consecutive block skips")

# Get unique models and datasets
models = sorted(consecutive_df["model"].unique())
datasets = sorted(consecutive_df["dataset"].unique())

print(f"Models: {models}")
print(f"Datasets: {datasets}")

# Create subplot grid: rows = models, columns = datasets
fig, axs = plt.subplots(
    len(models),
    len(datasets),
    figsize=(4 * len(datasets), 3 * len(models)),
    sharex=False,
    sharey=True,
    squeeze=False
)

# Calculate global y-axis limits
best_val_linear = consecutive_df['accuracy_mean_linear'].max()
best_val_identity = consecutive_df['accuracy_mean_identity'].max()
worst_val_linear = consecutive_df['accuracy_mean_linear'].min()
worst_val_identity = consecutive_df['accuracy_mean_identity'].min()
best_val = max(best_val_linear, best_val_identity) + 0.02
worst_val = min(worst_val_linear, worst_val_identity) - 0.02

for model_idx, model in enumerate(models):
    for dataset_idx, dataset in enumerate(datasets):
        ax = axs[model_idx, dataset_idx]
        model_dataset_df = consecutive_df[
            (consecutive_df["model"] == model) & (consecutive_df["dataset"] == dataset)
        ].copy()

        if model_dataset_df.empty:
            ax.set_title(f"{model}\n{dataset}\n(no data)")
            continue

        # Sort by first_skip to get the order of consecutive block pairs
        model_dataset_df = model_dataset_df.sort_values("first_skip")

        # Extract model and dataset names for title
        model_short = model.split("/")[-1]
        dataset_short = dataset

        ax.scatter(
            model_dataset_df["first_skip"],
            model_dataset_df["accuracy_mean_linear"],
            marker="o",
            color="blue",
            label="Linear Skip",
            s=60,
            alpha=0.7
        )
        ax.scatter(
            model_dataset_df["first_skip"],
            model_dataset_df["accuracy_mean_identity"],
            marker="o",
            color="orange",
            label="Identity Skip",
            s=60,
            alpha=0.7
        )

        ax.set_title(f"{model_short}\n{dataset_short}")
        ax.set_xlabel("First Block in Pair")
        ax.set_ylabel("Accuracy")
        ax.set_ylim(worst_val, best_val)
        ax.grid(alpha=0.3, linestyle="--")

        # Set x-axis to show integer positions
        unique_first_skip = sorted(model_dataset_df["first_skip"].unique())
        if len(unique_first_skip) <= 12:
            ax.set_xticks(unique_first_skip)

# Add shared legend at the top
handles, labels = axs[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.99))

# Adjust layout and save
fig.suptitle("Accuracy for Consecutive Block Skips (e.g., (0,1), (1,2)...)",
             fontsize=16, y=0.995)
os.makedirs("plots", exist_ok=True)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig("plots/overall_consecutive_blocks_analysis.png", bbox_inches="tight", dpi=150)
print("Saved: plots/overall_consecutive_blocks_analysis.png")
plt.show()
plt.close(fig)

# Also save the filtered data for reference
consecutive_df.to_csv("consecutive_blocks_analysis.csv", index=False)
print("Saved: consecutive_blocks_analysis.csv")
