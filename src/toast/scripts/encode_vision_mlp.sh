#!/bin/bash

# --- NEW: Point HF to your customapps local assets ---
export HF_HOME="/cluster/customapps/biomed/vogtlab/users/mwylie/toast/hf_cache"
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

SKIPS='[[], [(0, 1)]]'
MLP_SKIPS='[[], [2,3]]'

for dataset_name in mnist
do
    # Change the loop to use the LOCAL_ENCODER path
    for encoder_name in "facebook/deit-small-patch16-224"
    do
        for translator_name in linear
        do
            for samples_to_extract in 500
            do
                python src/toast/utils/encode_vision_mlp.py \
                    --dataset_name=$dataset_name \
                    --encoder_name="$encoder_name" \
                    --translator_name=$translator_name \
                    --seed=0 \
                    --skips="$SKIPS" \
                    --mlp_skips="$MLP_SKIPS" \
                    --samples_to_extract=$samples_to_extract
            done
        done
    done
done
