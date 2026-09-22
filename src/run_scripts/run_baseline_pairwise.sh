#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:rtx4090:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=48000
#SBATCH --time=12:00:00
#SBATCH --output=logs/baseline_pairwise.out.txt

# =============================================================================
# Baseline pairwise layer geometry -- the "angular distance heatmap".
#
# For every ENCODER x DATASET combo, one clean forward pass of the UNMODIFIED
# model, then the pairwise angular distance / CKA between ALL hidden-state pairs
# (layer i vs layer j), i.e. layer_priority.py's analyse():
#   1. layer_priority.py  -> <stem>_angdist.npy, <stem>_cka.npy, <stem>_block_scores.csv
#   2. analysis_plots.py  -> <stem>/angular_distance.png (heatmap + best-span overlay)
#                            <stem>/cka.png (linear-CKA heatmap), + sublayer/head plots
#
# This is NOT the post-skip angular_dist that output_level_eval.py computes inside
# each skipped model -- it is the baseline representation structure, for reading
# alongside the group-1/2 accuracy sweeps. Nothing here touches the accuracies.
#
#   sbatch src/run_scripts/run_baseline_pairwise.sh                 # group-1 default
#   DATASETS="cifar10 pathmnist" ENCODERS="owkin/phikon-v2" \
#     sbatch src/run_scripts/run_baseline_pairwise.sh
# =============================================================================

set -e

# -------------------------------- CONFIG -------------------------------------
DATASETS="${DATASETS:-chestxray14 chestxray_pneumonia imagenet-1k}"
ENCODERS="${ENCODERS:-facebook/dinov2-base microsoft/rad-dino microsoft/rad-dino-maira-2}"
NUM_SAMPLES="${NUM_SAMPLES:-5000}"
OUTPUTS_DIR="${OUTPUTS_DIR:-src/layers/outputs}"
PLOTS_DIR="${PLOTS_DIR:-src/layers/plots}"
FORCE="${FORCE:-0}"                # 1 = recompute score files even if they exist
# -----------------------------------------------------------------------------

PROJECT_DIR="/cluster/home/mwylie/toast-extensions"
WRITABLE_CACHE="/cluster/home/mwylie/hf_writable"
mkdir -p "$WRITABLE_CACHE" logs "$OUTPUTS_DIR" "$PLOTS_DIR"

source $GPU_ENV/bin/activate

export HF_HOME="/cluster/customapps/biomed/vogtlab/users/mwylie/toast/hf_cache"
export HF_DATASETS_CACHE="$WRITABLE_CACHE/datasets"
export TRANSFORMERS_CACHE="$WRITABLE_CACHE/transformers"
export HF_MODULES_CACHE="$WRITABLE_CACHE/modules"
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TQDM_DISABLE=1
export TRANSFORMERS_VERBOSITY=error
export HF_HUB_DISABLE_PROGRESS_BARS=1
export DATASETS_VERBOSITY=error
export PYTHONPATH=$PYTHONPATH:$PROJECT_DIR/src
cd "$PROJECT_DIR"

force_flag=""
[ "$FORCE" -eq 1 ] && force_flag="--force"

for enc in $ENCODERS; do
    for ds in $DATASETS; do
        # Same stem layer_priority.run() builds: resolved HF id, slashes -> underscores.
        stem="$(echo "${enc//\//_}_${ds}" | tr '[:upper:]' '[:lower:]')"
        echo ""
        echo "############################################################"
        echo "  ${enc}  x  ${ds}   (stem: ${stem})"
        echo "############################################################"

        python src/layers/layer_priority.py \
            --model "$enc" --dataset "$ds" \
            --num-samples "$NUM_SAMPLES" --output-dir "$OUTPUTS_DIR" $force_flag

        python src/layers/analysis_plots.py \
            --stem "$stem" --output-dir "$OUTPUTS_DIR" \
            --plot-dir "$PLOTS_DIR/$stem"
    done
done

echo ""
echo "Done. Heatmaps in $PLOTS_DIR/<stem>/angular_distance.png and cka.png."
