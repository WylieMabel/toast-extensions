#!/bin/bash

export HF_HOME="/cluster/customapps/biomed/vogtlab/users/mwylie/toast/hf_cache"
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

CONFIG_CSV="${CONFIG_CSV:-src/configs/experiments.csv}"

echo "GPU Memory before:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader

python src/toast/utils/encode_vision_full.py \
    --translator_name=identity \
    --seed=0 \
    --config_csv="$CONFIG_CSV" \
    --batch_size=8 \
    --samples_to_extract=250

echo "GPU Memory after:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
