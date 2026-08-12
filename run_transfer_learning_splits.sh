#!/bin/bash
# Run all transfer learning experiments split by model and skip pattern
# Each model runs its skips sequentially, 4 models can run in parallel

set -e

echo "Starting transfer learning experiments (4 models in parallel)"
echo "=============================================================="

# Model 1: deit-small-patch16-224
echo ""
echo "Submitting deit-small-patch16-224 (8 skips)..."
for f in src/configs/transfer_learning/experiments_transfer_learning_deit-small-patch16-224_skip*.csv; do
    CONFIG_CSV="$f" sbatch --output="logs/transfer_learning_$(basename "$f" .csv).out.txt" src/run_scripts/run_pipeline_row_by_row.sh
    sleep 1
done

# Model 2: dinov2-base
echo ""
echo "Submitting dinov2-base (7 skips)..."
for f in src/configs/transfer_learning/experiments_transfer_learning_dinov2-base_skip*.csv; do
    CONFIG_CSV="$f" sbatch --output="logs/transfer_learning_$(basename "$f" .csv).out.txt" src/run_scripts/run_pipeline_row_by_row.sh
    sleep 1
done

# Model 3: rad-dino
echo ""
echo "Submitting rad-dino (7 skips)..."
for f in src/configs/transfer_learning/experiments_transfer_learning_rad-dino_skip*.csv; do
    CONFIG_CSV="$f" sbatch --output="logs/transfer_learning_$(basename "$f" .csv).out.txt" src/run_scripts/run_pipeline_row_by_row.sh
    sleep 1
done

# Model 4: vit-large-patch16-224
echo ""
echo "Submitting vit-large-patch16-224 (15 skips)..."
for f in src/configs/transfer_learning/experiments_transfer_learning_vit-large-patch16-224_skip*.csv; do
    CONFIG_CSV="$f" sbatch --output="logs/transfer_learning_$(basename "$f" .csv).out.txt" src/run_scripts/run_pipeline_row_by_row.sh
    sleep 1
done

echo ""
echo "=============================================================="
echo "All experiments submitted!"
echo ""
echo "Monitor progress:"
echo "  squeue -u $USER | grep transfer_learning"
echo ""
echo "Monitor translator accumulation:"
echo "  watch 'ls -1 /path/to/translators | wc -l'"
