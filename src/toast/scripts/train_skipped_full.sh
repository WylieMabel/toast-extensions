#!/bin/bash

export HF_HOME="/cluster/customapps/biomed/vogtlab/users/mwylie/toast/hf_cache"
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

CONFIG_CSV="${CONFIG_CSV:-src/configs/experiments.csv}"

for seed in 1 2 3 4 5
do
    python src/toast/utils/train_skipped_full.py \
        --seed=$seed \
        --classifier_type=linear \
        --translator_name=identity \
        --samples_to_extract=250 \
        --config_csv="$CONFIG_CSV"
done
