#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:rtx4090:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=48000
#SBATCH --time=48:00:00
#SBATCH --output=logs/layer_priority.out

# Runs layer_priority.py over the full encoder x dataset grid backing
# src/configs/experiments_skip_grid.csv. Combinations whose score files already exist are
# skipped by layer_priority.py itself (pass FORCE=1 to recompute), so this is safe to
# re-submit after a timeout or a partial run -- it picks up exactly what is still missing.

# 1. Path setup
PROJECT_DIR="/cluster/home/mwylie/toast-extensions"
WRITABLE_CACHE="/cluster/home/mwylie/hf_writable"
OUTPUT_DIR="$PROJECT_DIR/src/layers/outputs"

# 2. Activate venv
source $GPU_ENV/bin/activate
pip install -e .

# 3. Environment variables
export HF_HOME="/cluster/customapps/biomed/vogtlab/users/mwylie/toast/hf_cache"
export HF_DATASETS_CACHE="$WRITABLE_CACHE/datasets"
export TRANSFORMERS_CACHE="$WRITABLE_CACHE/transformers"
export HF_MODULES_CACHE="$WRITABLE_CACHE/modules"
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# 4. Fix Python imports
export PYTHONPATH=$PYTHONPATH:$PROJECT_DIR/src

# 5. Run the analyser over model x dataset combinations
cd $PROJECT_DIR

MODELS=(
    "facebook/deit-small-patch16-224"
    "facebook/deit-base-patch16-224"
    "facebook/dinov2-base"
    "google/vit-base-patch16-224"
    "google/vit-large-patch16-224"
    "microsoft/rad-dino"
)
DATASETS=("imagenet-1k" "cifar100" "pneumoniamnist" "dermamnist")

NUM_SAMPLES=5000
FORCE=${FORCE:-0}                      # FORCE=1 recomputes combos that already have scores

force_flag=""
[ "$FORCE" -eq 1 ] && force_flag="--force"

# Dataset-major, so a run that hits the wall clock still leaves whole datasets finished
# rather than a fragment of every one.
for DATASET in "${DATASETS[@]}"; do
    for MODEL in "${MODELS[@]}"; do
        echo "------------------------------------------------------------"
        echo "  Model: ${MODEL}  |  Dataset: ${DATASET}"
        echo "------------------------------------------------------------"
        python src/layers/layer_priority.py \
            --model "${MODEL}" \
            --dataset "${DATASET}" \
            --num-samples "${NUM_SAMPLES}" \
            --output-dir "${OUTPUT_DIR}" $force_flag
    done
done

# Report anything still missing, so a truncated run is obvious from the log tail.
echo ""
echo "============================================================"
echo "  Coverage"
echo "============================================================"
missing=0
for DATASET in "${DATASETS[@]}"; do
    for MODEL in "${MODELS[@]}"; do
        stem="$(echo "${MODEL//\//_}_${DATASET}" | tr '[:upper:]' '[:lower:]')"
        if [ -f "${OUTPUT_DIR}/${stem}_block_scores.csv" ]; then
            echo "  ok      ${stem}"
        else
            echo "  MISSING ${stem}"
            missing=$((missing + 1))
        fi
    done
done
echo "  ${missing} combination(s) still missing."
