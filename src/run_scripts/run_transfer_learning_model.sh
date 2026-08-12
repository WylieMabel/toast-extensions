#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=1
#SBATCH --mem=1000
#SBATCH --time=72:00:00

# Runs all skips for a single model sequentially
# Submit 4 copies in parallel (one per model)
# Usage:
#   MODEL=deit-small-patch16-224 sbatch src/run_scripts/run_transfer_learning_model.sh
#   MODEL=dinov2-base sbatch src/run_scripts/run_transfer_learning_model.sh
#   MODEL=rad-dino sbatch src/run_scripts/run_transfer_learning_model.sh
#   MODEL=vit-large-patch16-224 sbatch src/run_scripts/run_transfer_learning_model.sh

MODEL="${MODEL:-deit-small-patch16-224}"
PROJECT_DIR="/cluster/home/mwylie/toast-extensions"

cd $PROJECT_DIR

echo "=========================================="
echo "Starting $MODEL"
echo "=========================================="

prev_job_id=""

for config_file in src/configs/transfer_learning/experiments_transfer_learning_${MODEL}_skip*.csv; do
    [ ! -f "$config_file" ] && continue

    skip_name=$(basename "$config_file" .csv)
    log_file="logs/${skip_name}.out.txt"

    echo ""
    echo ">>> Submitting $skip_name..."

    # Submit with dependency on previous job (if any)
    if [ -n "$prev_job_id" ]; then
        job_cmd="CONFIG_CSV=\"$config_file\" sbatch --dependency=afterok:$prev_job_id --output=\"$log_file\" src/run_scripts/run_pipeline_row_by_row.sh"
    else
        job_cmd="CONFIG_CSV=\"$config_file\" sbatch --output=\"$log_file\" src/run_scripts/run_pipeline_row_by_row.sh"
    fi

    job_id=$(eval $job_cmd | awk '{print $NF}')
    prev_job_id=$job_id

    echo "    Job ID: $job_id"
    if [ -n "$prev_job_id" ]; then
        echo "    (depends on previous job)"
    fi
done

echo ""
echo "=========================================="
echo "All $MODEL skips queued (chained with dependencies)"
echo "=========================================="
