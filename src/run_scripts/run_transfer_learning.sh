#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=1
#SBATCH --mem=1000
#SBATCH --time=72:00:00
#SBATCH --output=logs/transfer_learning_submission.out.txt

# Transfer Learning Experiments: Submit all patterns in parallel batches
#
# Usage (option 1 - run on login node):
#   bash run_transfer_learning.sh
#
# Usage (option 2 - submit as Slurm job):
#   sbatch run_transfer_learning.sh
#
# What it does:
#   1. Submits 6 patterns in parallel to GPU queue
#   2. Waits for batch to complete
#   3. Submits next 6
#   4. Repeats until all 70 patterns done
#
# Timeline: ~24-36 hours total (1-2 days)
# Storage: ~100MB peak per batch

set -e

CONFIG_DIR="src/configs/transfer_learning_patterns"
RESULTS_DIR="results/transfer_learning"
LOGS_DIR="logs"

mkdir -p "$LOGS_DIR" "$RESULTS_DIR"

# Verify configs exist
if [ ! -d "$CONFIG_DIR" ]; then
    echo "ERROR: Config directory not found: $CONFIG_DIR"
    exit 1
fi

pattern_count=$(ls -1 "$CONFIG_DIR"/*.csv 2>/dev/null | wc -l)
if [ $pattern_count -eq 0 ]; then
    echo "ERROR: No pattern CSV files found in $CONFIG_DIR"
    exit 1
fi

total_batches=$(( (pattern_count + 5) / 6 ))

echo "=========================================================================="
echo "Transfer Learning Experiments: Parallel Batch Submission"
echo "=========================================================================="
echo ""
echo "Patterns: $pattern_count"
echo "Batch size: 6 (parallel)"
echo "Total batches: $total_batches"
echo ""
echo "Timeline:"
echo "  Each pattern: ~1-2 hours"
echo "  Each batch (6 parallel): ~1-2 hours"
echo "  Total: ~$((total_batches * 2)) hours (~$((total_batches / 12))-$((total_batches / 6)) days)"
echo ""
echo "Storage: ~100MB per batch (cleaned up automatically)"
echo ""
echo "=========================================================================="
echo ""

# Get sorted pattern list
patterns=$(ls -1 "$CONFIG_DIR"/experiments_transfer_learning_pattern_*.csv | \
    xargs -n1 basename | \
    sed 's/experiments_transfer_learning_pattern_//' | \
    sed 's/\.csv$//' | \
    sort -V)

batch_num=0
jobs_submitted=()

# Submit patterns in batches
for pattern in $patterns; do
    config_file="$CONFIG_DIR/experiments_transfer_learning_pattern_${pattern}.csv"
    results_file="results_transfer_learning_pattern_${pattern}.csv"
    log_file="$LOGS_DIR/transfer_learning_pattern_${pattern}.out.txt"

    if [ ! -f "$config_file" ]; then
        continue
    fi

    # Submit this pattern
    echo "Submitting: $pattern"
    sbatch --output="$log_file" \
        --export=CONFIG_CSV="$config_file",RESULTS_CSV_NAME="$RESULTS_DIR/$results_file" \
        src/run_scripts/run_pipeline_row_by_row.sh > /tmp/sbatch_out.txt 2>&1

    if grep -q "Submitted batch job" /tmp/sbatch_out.txt; then
        job_id=$(grep -oE '[0-9]+$' /tmp/sbatch_out.txt)
        jobs_submitted+=("$job_id")
        echo "  Job ID: $job_id"
    fi

    # When batch reaches 6 jobs, wait for completion
    if [ ${#jobs_submitted[@]} -eq 6 ]; then
        batch_num=$((batch_num + 1))
        echo ""
        echo "Batch $batch_num complete (6 jobs submitted)"
        echo "Waiting for all 6 to finish..."
        echo ""

        # Wait for all jobs to complete
        all_done=0
        while [ $all_done -eq 0 ]; do
            all_done=1
            for job_id in "${jobs_submitted[@]}"; do
                if squeue -j "$job_id" &>/dev/null; then
                    all_done=0
                    break
                fi
            done

            if [ $all_done -eq 0 ]; then
                sleep 30
            fi
        done

        echo "✓ Batch $batch_num finished!"
        echo "  Completed so far: $(ls -1 "$RESULTS_DIR"/results*.csv 2>/dev/null | wc -l) / $pattern_count"
        echo ""

        jobs_submitted=()
    fi

    sleep 1
done

# Wait for final batch
if [ ${#jobs_submitted[@]} -gt 0 ]; then
    batch_num=$((batch_num + 1))
    echo ""
    echo "Batch $batch_num (final, ${#jobs_submitted[@]} jobs)"
    echo "Waiting for completion..."
    echo ""

    all_done=0
    while [ $all_done -eq 0 ]; do
        all_done=1
        for job_id in "${jobs_submitted[@]}"; do
            if squeue -j "$job_id" &>/dev/null; then
                all_done=0
                break
            fi
        done

        if [ $all_done -eq 0 ]; then
            sleep 30
        fi
    done

    echo "✓ Batch $batch_num finished!"
fi

rm -f /tmp/sbatch_out.txt

echo ""
echo "=========================================================================="
echo "✓ All experiments submitted and completed!"
echo "=========================================================================="
echo ""
echo "Next steps:"
echo ""
echo "1. Consolidate results:"
echo "   bash consolidate_transfer_learning.sh"
echo ""
echo "2. Analyze transfer efficiency:"
echo "   python analyze_transfer_results.py \\"
echo "     --results results_transfer_learning_all.csv \\"
echo "     --ranking source_rankings.csv \\"
echo "     --transfer-losses transfer_losses.csv"
echo ""
echo "3. View top results:"
echo "   head -20 source_rankings.csv | column -t -s','"
echo ""
