#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:rtx4090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80000
#SBATCH --time=2:00:00
#SBATCH --output=logs/output_level_metrics_%j.out.txt

# Output-level skip-safety metrics (Exp 1 + Exp 2) + evaluation
# Runs output_level_eval.py then joins + evaluates via Spearman/P@5/R@5
#
#   sbatch metric_experiments/run_output_level_metrics.sh
#   MODEL=google/vit-large-patch16-224 MODEL_ALIAS=vitlarge \
#     TRUTH_CSV=block_distance/distance_analysis.csv \
#     TRUTH_LAYER_COL=approx_layer_str TRUTH_ACC_COL=accuracy_mean_linear \
#     sbatch metric_experiments/run_output_level_metrics.sh
#
#   # multi-span sweep, spans + truth both pulled from the same accuracies CSV:
#   MODEL=facebook/deit-small-patch16-224 MODEL_ALIAS=deitsmall STEM=deitsmall_multiskip \
#     TRUTH_CSV=results/accuracies_results_vitlarge_deitsmall_contiguous_multiskip.csv \
#     SPANS_CSV=results/accuracies_results_vitlarge_deitsmall_contiguous_multiskip.csv \
#     sbatch metric_experiments/run_output_level_metrics.sh
#   (repeat with MODEL=google/vit-large-patch16-224 MODEL_ALIAS=vitlarge STEM=vitlarge_multiskip)

set -e

# ---- CONFIG ---- (override any of these via env var, e.g. MODEL_ALIAS=vitlarge sbatch ...)
MODEL="${MODEL:-facebook/deit-small-patch16-224}"
MODEL_ALIAS="${MODEL_ALIAS:-deitsmall}"
DATASET="${DATASET:-imagenet-1k}"
NUM_SAMPLES="${NUM_SAMPLES:-500}"
# Train-split images for fitting/scoring the linear-probe metrics (probe_frozen_accuracy,
# probe_refit_accuracy, probe_baseline_accuracy). Independent of NUM_SAMPLES, which stays
# test/validation-only and feeds every other metric.
PROBE_NUM_SAMPLES="${PROBE_NUM_SAMPLES:-500}"
TRUTH_CSV="${TRUTH_CSV:-results/accuracies_results_vitlarge_vitsmall_deitsmall_imagenet1k_singleskip.csv}"
TRUTH_LAYER_COL="${TRUTH_LAYER_COL:-approx_layer}"
TRUTH_ACC_COL="${TRUTH_ACC_COL:-accuracy_mean}"
# When set, output_level_eval.py evaluates exactly the spans found in this CSV for
# (MODEL, DATASET) instead of the default single-layer adjacent skips. Typically the same
# file as TRUTH_CSV, so the metrics and the ground truth cover identical candidates.
SPANS_CSV="${SPANS_CSV:-}"
# Stem used for output filenames -- lets deitsmall/vitlarge/vitsmall runs land in separate files.
STEM="${STEM:-$MODEL_ALIAS}"
# Filename tag for the dataset, hyphens stripped to match the pre-existing "imagenet1k" style
# (DATASET=imagenet-1k -> imagenet1k; cifar10/cifar100/pathmnist pass through unchanged).
# Set DATASET_TAG to override it (s4 names files chestxraypneumonia / tcgaut).
DATASET_TAG="${DATASET_TAG:-${DATASET//-/}}"
# Where the stage-1 CSV is written (single-combo mode), e.g. a thesis section's analysis/ folder.
OUT_DIR="${OUT_DIR:-metric_experiments}"
# EVAL_ONLY=1 stops after stage 1 (output_level_eval.py); the join/stats are then done for the
# whole group at once by build_group_stats.py (see thesis/s2_skip_cost_metrics/launch_s2_metrics.sh).
EVAL_ONLY="${EVAL_ONLY:-}"
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

# ---- GROUP MODE ---------------------------------------------------------------
#   GROUP=group1 sbatch metric_experiments/run_output_level_metrics.sh
# Ignores the single-combo CONFIG above. Loops every (backbone, dataset) combo in
# the group through stage 1 (output_level_eval.py -- spans + truth both taken from
# results/accuracies_results_group1_<dataset>.csv), then build_group_stats.py for
# stages 2-4: join -> 1%-gap "safe" labels -> metric x combo matrices + summary
# grid, all written with a _<group> suffix so group-2 outputs are untouched.
if [ -n "$GROUP" ]; then
    case "$GROUP" in
      group1)
        ALIASES=(dinov2base raddino raddinomaira2)
        DATASETS=(chestxray14 chestxray_pneumonia imagenet-1k)
        TRUTH_TMPL="results/accuracies_results_group1_%s.csv"   # %s = hyphen-stripped dataset
        ;;
      group2)
        ALIASES=(vitlarge phikonv2)
        DATASETS=(imagenet-1k patchcamelyon tcga_ut)
        # group2's truth lives under group2_combined/ (deduplicated merge of the original run
        # + shard reruns + seed-completion rerun -- see the session that built it), not
        # directly under results/ like group1's.
        TRUTH_TMPL="results/group2_combined/accuracies_results_group2_%s.csv"
        ;;
      *) echo "unknown GROUP '$GROUP'"; exit 1 ;;
    esac

    for a in "${ALIASES[@]}"; do
        for d in "${DATASETS[@]}"; do
            dtag="${d//-/}"
            truth=$(printf "$TRUTH_TMPL" "$dtag")
            stem="${GROUP}_${a}_${dtag}"
            echo "---- stage 1: $stem  (spans + truth <- $truth) ----"
            python metric_experiments/output_level_eval.py \
                --model "$a" --dataset "$d" --num-samples "$NUM_SAMPLES" \
                --probe-num-samples "$PROBE_NUM_SAMPLES" \
                --spans-csv "$truth" \
                --output "metric_experiments/output_level_eval_${stem}.csv"
        done
    done

    echo "---- stages 2-4: build_group_stats.py --group $GROUP ----"
    python metric_experiments/build_group_stats.py --group "$GROUP"

    echo ""
    echo "============================================================"
    echo "  Group '$GROUP' complete!  End: $(date)"
    echo "============================================================"
    exit 0
fi

# Step 1: Compute metrics
SPANS_ARGS=()
if [ -n "$SPANS_CSV" ]; then
    SPANS_ARGS=(--spans-csv "$SPANS_CSV")
fi
python metric_experiments/output_level_eval.py \
    --model "$MODEL_ALIAS" \
    --dataset "$DATASET" \
    --num-samples "$NUM_SAMPLES" \
    --probe-num-samples "$PROBE_NUM_SAMPLES" \
    --output "${OUT_DIR}/output_level_eval_${STEM}_${DATASET_TAG}.csv" \
    "${SPANS_ARGS[@]}"

if [ -n "$EVAL_ONLY" ]; then
    echo ""
    echo "============================================================"
    echo "  Stage 1 complete (EVAL_ONLY) -> ${OUT_DIR}/output_level_eval_${STEM}_${DATASET_TAG}.csv"
    echo "  End: $(date)"
    echo "============================================================"
    exit 0
fi

# Step 2: Join + evaluate. --truth-layer-col/--truth-acc-col let this point at either
# distance_analysis.csv's linear/identity-merged shape (approx_layer_str/accuracy_mean_linear)
# or a skipping_heads/calculate_accuracies.py summary CSV (approx_layer/accuracy_mean) --
# see join_and_evaluate.py's --help for the distinction. USE_RAW_ACCURACY=1 correlates
# against raw true_accuracy instead of true_delta_acc (same |corr|, more interpretable sign).
USE_RAW_ACCURACY_ARGS=()
if [ -n "$USE_RAW_ACCURACY" ]; then
    USE_RAW_ACCURACY_ARGS=(--use-raw-accuracy)
fi
python metric_experiments/join_and_evaluate.py \
    --eval-csv "${OUT_DIR}/output_level_eval_${STEM}_${DATASET_TAG}.csv" \
    --truth-csv "$TRUTH_CSV" \
    --model "$MODEL" \
    --dataset "$DATASET" \
    --truth-layer-col "$TRUTH_LAYER_COL" \
    --truth-acc-col "$TRUTH_ACC_COL" \
    --output "metric_experiments/joined_metrics_${STEM}_${DATASET_TAG}.csv" \
    "${USE_RAW_ACCURACY_ARGS[@]}"

echo ""
echo "============================================================"
echo "  Complete!"
echo "  End: $(date)"
echo "============================================================"
