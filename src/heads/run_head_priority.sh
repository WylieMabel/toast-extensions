#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:rtx4090:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24000
#SBATCH --time=6:00:00
#SBATCH --output=logs/head_priority.out

# 1. Path setup
PROJECT_DIR="/cluster/home/mwylie/toast-extensions"
WRITABLE_CACHE="/cluster/home/mwylie/hf_writable"
OUTPUT_DIR="$PROJECT_DIR/src/heads/outputs"

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

# 5. Run all 9 model x dataset combinations
cd $PROJECT_DIR

#MODELS=("deitsmall" "dinobase")
MODELS=("deitsmall")
#MODELS=("vitlarge")
DATASETS=("mnist" "cifar100" "imagenet1k")

for MODEL in "${MODELS[@]}"; do
    for DATASET in "${DATASETS[@]}"; do
        echo "------------------------------------------------------------"
        echo "  Model: ${MODEL}  |  Dataset: ${DATASET}"
        echo "------------------------------------------------------------"
        python src/heads/head_priority.py \
            --model "${MODEL}" \
            --dataset "${DATASET}" \
            --num-samples 5000 \
            --output-dir "${OUTPUT_DIR}"
    done
done
