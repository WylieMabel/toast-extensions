#!/bin/bash

export HF_HOME="/cluster/customapps/biomed/vogtlab/users/mwylie/toast/hf_cache"
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

CONFIG_CSV="${CONFIG_CSV:-src/configs/experiments.csv}"

python src/toast/utils/encode_vision_full.py \
    --translator_name=identity \
    --seed=0 \
    --config_csv="$CONFIG_CSV" \
    --samples_to_extract=500
