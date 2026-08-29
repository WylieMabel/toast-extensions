import pandas as pd
import numpy as np
import json
from pathlib import Path

# Load data
results = pd.read_csv("/Users/mabelwylie/Documents/toast-extensions/results/results_window_grid_combined.csv")
accuracies = pd.read_csv("/Users/mabelwylie/Documents/toast-extensions/skipping_heads/accuracies/accuracies_window_grid_combined.csv")

# Extract model name (short form)
results['model_short'] = results['model'].str.split('/').str[-1]
results['num_approx_layers'] = results['num_layers']

# Get baseline (identity translator) accuracies for each model-dataset combination
baseline_stats = results[results['translator'] == 'identity'].groupby(['model_short', 'dataset']).agg({
    'original_accuracy': 'mean',
    'accuracy': 'mean',
    'params_saved_pct': 'mean'
}).reset_index()
baseline_stats.columns = ['Model', 'Dataset', 'Original Accuracy', 'Baseline Accuracy', 'Params Saved %']

# Summary by model and translator
summary_by_model = results[results['translator'] == 'linear'].groupby(['model_short', 'dataset']).agg({
    'accuracy': ['mean', 'std', 'min'],
    'delta_acc': ['mean', 'max'],
    'params_saved_pct': 'max',
    'num_layers': 'max'
}).reset_index()

summary_by_model.columns = ['Model', 'Dataset', 'Mean Accuracy', 'Accuracy Std', 'Min Accuracy',
                            'Mean Delta', 'Max Delta', 'Max Params Saved %', 'Max Layers']

# Group by model to show overall performance
model_summary = results[results['translator'] == 'linear'].groupby('model_short').agg({
    'accuracy': ['mean', 'min'],
    'delta_acc': 'max',
    'params_saved_pct': 'max',
    'original_accuracy': 'mean'
}).reset_index()

model_summary.columns = ['Model', 'Mean Accuracy', 'Min Accuracy', 'Max Accuracy Loss', 'Max Params Saved %', 'Baseline Accuracy']

print("="*80)
print("THESIS PRESENTATION SUMMARY")
print("="*80)

print("\n1. BASELINE ACCURACY BY MODEL AND DATASET")
print("-"*80)
print(baseline_stats.to_string(index=False))

print("\n\n2. LINEAR APPROXIMATION PERFORMANCE BY MODEL AND DATASET")
print("-"*80)
print(summary_by_model.to_string(index=False))

print("\n\n3. OVERALL MODEL PERFORMANCE (Linear Approximation)")
print("-"*80)
print(model_summary.sort_values('Max Params Saved %', ascending=False).to_string(index=False))

# Create per-model analysis
print("\n\n4. DETAILED ANALYSIS BY MODEL")
print("="*80)

for model in sorted(results['model_short'].unique()):
    print(f"\n{model.upper()}")
    print("-"*80)

    model_data = results[results['model_short'] == model]

    for dataset in sorted(model_data['dataset'].unique()):
        dataset_data = model_data[model_data['dataset'] == dataset]

        # Baseline
        baseline = dataset_data[dataset_data['translator'] == 'identity']['accuracy'].mean()

        # Linear approximation stats
        linear = dataset_data[dataset_data['translator'] == 'linear']

        if len(linear) > 0:
            min_acc = linear['accuracy'].min()
            max_loss = linear['delta_acc'].max()
            max_params = linear['params_saved_pct'].max()

            print(f"  {dataset}:")
            print(f"    Baseline accuracy (no approximation): {baseline:.4f}")
            print(f"    Minimum accuracy with approximation:  {min_acc:.4f}")
            print(f"    Maximum accuracy loss:                {max_loss:.4f}")
            print(f"    Maximum parameter savings:            {max_params:.2f}%")

# Prepare data for visualization (accuracy vs compression tradeoff)
print("\n\n5. ACCURACY-COMPRESSION TRADEOFF DATA")
print("="*80)

tradeoff_data = {}
for model in sorted(results['model_short'].unique()):
    model_data = results[results['model_short'] == model]
    model_data = model_data[model_data['translator'] == 'linear']  # Only linear approximations

    # Aggregate by number of layers approximated
    by_layers = model_data.groupby('num_approx_layers').agg({
        'accuracy': 'mean',
        'params_saved_pct': 'mean',
        'delta_acc': 'mean'
    }).reset_index()

    tradeoff_data[model] = by_layers.to_dict('records')

# Save tradeoff data for visualization
with open('/private/tmp/claude-501/-Users-mabelwylie-Documents-toast-extensions/f0ded13b-31c0-4c77-a367-179810b6683a/scratchpad/tradeoff_data.json', 'w') as f:
    json.dump(tradeoff_data, f, indent=2)

# Save model summary for visualization
model_summary.sort_values('Max Params Saved %', ascending=False).to_csv(
    '/private/tmp/claude-501/-Users-mabelwylie-Documents-toast-extensions/f0ded13b-31c0-4c77-a367-179810b6683a/scratchpad/model_summary.csv', index=False)

baseline_stats.to_csv(
    '/private/tmp/claude-501/-Users-mabelwylie-Documents-toast-extensions/f0ded13b-31c0-4c77-a367-179810b6683a/scratchpad/baseline_stats.csv', index=False)

summary_by_model.to_csv(
    '/private/tmp/claude-501/-Users-mabelwylie-Documents-toast-extensions/f0ded13b-31c0-4c77-a367-179810b6683a/scratchpad/summary_by_model.csv', index=False)

print("\nVisualization data saved to scratchpad/")
