#!/bin/bash

# --- NEW: Point HF to your customapps local assets ---
export HF_HOME="/cluster/customapps/biomed/vogtlab/users/mwylie/toast/hf_cache"
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Use the absolute path for the encoder
LOCAL_ENCODER="/cluster/customapps/biomed/vogtlab/users/mwylie/toast/local_assets/deit-small"

SKIPS='[[], [(0, 1)]]'

for dataset_name in mnist
do
    # Change the loop to use the LOCAL_ENCODER path
    for encoder_name in "facebook/deit-small-patch16-224"
    do
        for translator_name in linear
        do
            for samples_to_extract in 500
            do
                python src/toast/utils/encode_vision.py \
                    --dataset_name=$dataset_name \
                    --encoder_name="$encoder_name" \
                    --translator_name=$translator_name \
                    --seed=0 \
                    --skips="$SKIPS" \
                    --samples_to_extract=$samples_to_extract
            done
        done
    done
done
