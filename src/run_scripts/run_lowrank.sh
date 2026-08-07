#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:rtx4090:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=48000
#SBATCH --time=24:00:00
#SBATCH --output=logs/lowrank.out.txt

# Low-rank translator sweep: {lowrank, rrr, lora} x rank, against identity and full-rank
# linear controls, over three skip spans.
#
#   sbatch src/run_scripts/run_lowrank.sh
#
# Run src/run_scripts/run_preflight.sh first -- it smoke-tests the translators and tells you
# whether the rank grid can be trimmed.
#
# Overridable:
#   CONFIG_CSV        which sweep to run (default: the imagenet-1k one)
#   RESULTS_CSV_NAME  output under results/
#   SEEDS             passed through to train_skipped_full.sh (default there: "1 2 3")

CONFIG_CSV="${CONFIG_CSV:-src/configs/experiments_lowrank.csv}"
RESULTS_CSV_NAME="${RESULTS_CSV_NAME:-results_lowrank.csv}"
export CONFIG_CSV RESULTS_CSV_NAME
[ -n "$SEEDS" ] && export SEEDS

PROJECT_DIR="/cluster/home/mwylie/toast-extensions"
cd $PROJECT_DIR

# run_pipeline_row_by_row.sh activates the venv, but it runs in a subshell so that activation
# does not reach the aggregation step below. Activate here too.
WRITABLE_CACHE="/cluster/home/mwylie/hf_writable"
mkdir -p $WRITABLE_CACHE logs
source $GPU_ENV/bin/activate
export HF_HOME="/cluster/customapps/biomed/vogtlab/users/mwylie/toast/hf_cache"
export HF_DATASETS_CACHE="$WRITABLE_CACHE/datasets"
export TRANSFORMERS_CACHE="$WRITABLE_CACHE/transformers"
export HF_MODULES_CACHE="$WRITABLE_CACHE/modules"
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH=$PYTHONPATH:$PROJECT_DIR/src

# The row-by-row runner does its own venv activation, env setup and per-row cleanup; running
# it here rather than duplicating that logic keeps the two in step.
bash src/run_scripts/run_pipeline_row_by_row.sh
STATUS=$?

echo ""
echo "============================================================"
echo "  Aggregating"
echo "============================================================"
python skipping_heads/calculate_accuracies.py \
    --input "results/${RESULTS_CSV_NAME}" \
    --output "results/accuracies_${RESULTS_CSV_NAME}"

echo ""
echo "Results:    results/${RESULTS_CSV_NAME}"
echo "Aggregated: results/accuracies_${RESULTS_CSV_NAME}"
echo ""
echo "Sanity checks before trusting these numbers:"
echo "  - 'linear' and a full-rank low-rank row should agree (same map)"
echo "  - params_saved should scale as 2*768*r across the sweep"
echo "  - no row should have delta_acc = NaN (that means its baseline row was missing)"

# Propagate the sweep's status: run_pipeline_row_by_row.sh exits non-zero if any row failed.
exit $STATUS
