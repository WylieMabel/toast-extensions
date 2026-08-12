#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=1
#SBATCH --mem=1000
#SBATCH --time=240:00:00
#SBATCH --output=logs/transfer_learning_coordinator_%j.out.txt

# Coordinator script: launches 4 SLURM jobs (one per model)
# Each job runs all skips for that model sequentially
# Jobs run in parallel, but skips within each model run one-after-another

BASE_DIR="/cluster/customapps/biomed/vogtlab/users/mwylie/toast"
PROJECT_DIR="/cluster/home/mwylie/toast-extensions"

cd $PROJECT_DIR

# Function to run all skips for a model
run_model_skips() {
    local model=$1
    local model_short=$2

    echo "=========================================="
    echo "Starting $model_short (PID: $$)"
    echo "=========================================="

    for config_file in src/configs/transfer_learning/experiments_transfer_learning_${model_short}_skip*.csv; do
        [ ! -f "$config_file" ] && continue

        skip_name=$(basename "$config_file" .csv)
        log_file="logs/${skip_name}_%j.out.txt"

        echo ""
        echo ">>> Submitting $skip_name..."

        # Submit and wait for completion
        job_id=$(CONFIG_CSV="$config_file" sbatch --output="$log_file" \
            src/run_scripts/run_pipeline_row_by_row.sh | awk '{print $NF}')

        echo "    Job ID: $job_id"
        squeue -j $job_id

        # Wait for this job to finish before starting the next skip
        while squeue -j $job_id &>/dev/null; do
            sleep 10
        done

        # Check if job succeeded
        if tail -1 "$log_file" | grep -q "All.*rows completed"; then
            echo "    ✓ Completed successfully"
        else
            echo "    ✗ Failed (check $log_file)"
        fi
    done

    echo ""
    echo "=========================================="
    echo "Finished $model_short"
    echo "=========================================="
}

# Launch all 4 models in parallel (each runs its skips sequentially)
echo "Launching 4 models in parallel (each runs its skips sequentially)"
echo "=================================================================="
echo ""

run_model_skips "deit-small-patch16-224" "deit-small-patch16-224" &
PID1=$!

run_model_skips "dinov2-base" "dinov2-base" &
PID2=$!

run_model_skips "rad-dino" "rad-dino" &
PID3=$!

run_model_skips "vit-large-patch16-224" "vit-large-patch16-224" &
PID4=$!

# Wait for all 4 models to finish
echo ""
echo "Waiting for all models to complete..."
wait $PID1 $PID2 $PID3 $PID4

echo ""
echo "=================================================================="
echo "All models completed!"
echo "=================================================================="
