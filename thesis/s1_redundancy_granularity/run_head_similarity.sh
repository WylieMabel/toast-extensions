#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:rtx4090:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16000
#SBATCH --time=01:00:00
#SBATCH --job-name=s1_head_similarity
#SBATCH --output=logs/s1_head_similarity_%j.out.txt

# Head redundancy by linear CKA and angular distance on imagenet-1k (500 samples), swept over
# thresholds 0.05..0.95, for vitsmall, deitsmall, dinobase (facebook/dinov2-base) and vitlarge.
# Run from the project root:
#   sbatch thesis/s1_redundancy_granularity/run_head_similarity.sh
# Subset of models / recompute existing scores:
#   MODELS="deitsmall vitlarge" FORCE=1 sbatch thesis/s1_redundancy_granularity/run_head_similarity.sh
#
# Writes thesis/s1_redundancy_granularity/results/head_similarity/head_keep_imagenet1k.csv
# (+ per-model .npy matrices) and configs/experiments_heads_<model>_imagenet1k.csv.

# 1. Path setup
PROJECT_DIR="/cluster/home/mwylie/toast-extensions"
WRITABLE_CACHE="/cluster/home/mwylie/hf_writable"

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

# 5. Run
cd $PROJECT_DIR

MODELS=${MODELS:-"vitsmall deitsmall dinobase vitlarge"}
FORCE=${FORCE:-0}
force_flag=""
[ "$FORCE" = "1" ] && force_flag="--force"

python thesis/s1_redundancy_granularity/head_similarity_thresholds.py \
    --models $MODELS \
    --dataset imagenet-1k \
    --num-samples 500 \
    $force_flag
