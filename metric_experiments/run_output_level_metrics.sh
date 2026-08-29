#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:rtx4090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80000
#SBATCH --time=02:00:00
#SBATCH --output=logs/output_level_metrics.out.txt

# Output-level skip-safety metrics (Exp 1 + Exp 2) + evaluation
# Runs output_level_eval.py then joins + evaluates via Spearman/P@5/R@5
#
#   sbatch metric_experiments/run_output_level_metrics.sh
#   MODEL=google/vit-large-patch16-224 MODEL_ALIAS=vitlarge \
#     TRUTH_CSV=block_distance/distance_analysis.csv \
#     TRUTH_LAYER_COL=approx_layer_str TRUTH_ACC_COL=accuracy_mean_linear \
#     sbatch metric_experiments/run_output_level_metrics.sh

set -e

# ---- CONFIG ---- (override any of these via env var, e.g. MODEL_ALIAS=vitlarge sbatch ...)
MODEL="${MODEL:-facebook/deit-small-patch16-224}"
MODEL_ALIAS="${MODEL_ALIAS:-deitsmall}"
DATASET="${DATASET:-imagenet-1k}"
NUM_SAMPLES="${NUM_SAMPLES:-500}"
TRUTH_CSV="${TRUTH_CSV:-results/accuracies_results_vitlarge_vitsmall_deitsmall_imagenet1k_singleskip.csv}"
TRUTH_LAYER_COL="${TRUTH_LAYER_COL:-approx_layer}"
TRUTH_ACC_COL="${TRUTH_ACC_COL:-accuracy_mean}"
# Stem used for output filenames -- lets deitsmall/vitlarge/vitsmall runs land in separate files.
STEM="${STEM:-$MODEL_ALIAS}"
# ----------------

PROJECT_DIR="/cluster/home/mwylie/toast-extensions"
WRITABLE_CACHE="/cluster/home/mwylie/hf_writable"
mkdir -p logs metric_experiments "$WRITABLE_CACHE"

cd "$PROJECT_DIR"
source $GPU_ENV/bin/activate

export HF_HOME="/cluster/customapps/biomed/vogtlab/users/mwylie/toast/hf_cache"
export HF_DATASETS_CACHE="$WRITABLE_CACHE/datasets"
export TRANSFORMERS_CACHE="$WRITABLE_CACHE/transformers"
export HF_MODULES_CACHE="$WRITABLE_CACHE/modules"
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH=$PYTHONPATH:$PROJECT_DIR/src

echo "============================================================"
echo "  Output-level Skip-Safety Metrics"
echo "============================================================"
echo "  Model   : $MODEL ($MODEL_ALIAS)"
echo "  Dataset : $DATASET"
echo "  Samples : $NUM_SAMPLES"
echo "  Start   : $(date)"
echo ""

# Step 1: Compute metrics
python metric_experiments/output_level_eval.py \
    --model "$MODEL_ALIAS" \
    --dataset "$DATASET" \
    --num-samples "$NUM_SAMPLES" \
    --output "metric_experiments/output_level_eval_${STEM}_imagenet1k.csv"

# Step 2: Join + evaluate. --truth-layer-col/--truth-acc-col let this point at either
# distance_analysis.csv's linear/identity-merged shape (approx_layer_str/accuracy_mean_linear)
# or a skipping_heads/calculate_accuracies.py summary CSV (approx_layer/accuracy_mean) --
# see join_and_evaluate.py's --help for the distinction.
python metric_experiments/join_and_evaluate.py \
    --eval-csv "metric_experiments/output_level_eval_${STEM}_imagenet1k.csv" \
    --truth-csv "$TRUTH_CSV" \
    --model "$MODEL" \
    --dataset "$DATASET" \
    --truth-layer-col "$TRUTH_LAYER_COL" \
    --truth-acc-col "$TRUTH_ACC_COL" \
    --output "metric_experiments/joined_metrics_${STEM}_imagenet1k.csv"

echo ""
echo "============================================================"
echo "  Complete!"
echo "  End: $(date)"
echo "============================================================"
