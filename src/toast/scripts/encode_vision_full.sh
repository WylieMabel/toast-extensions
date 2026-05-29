#!/bin/bash

# --- NEW: Point HF to your customapps local assets ---
export HF_HOME="/cluster/customapps/biomed/vogtlab/users/mwylie/toast/hf_cache"
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

SKIPS='[[]]'
#SKIPS='[[]]'
#MLP_SKIPS='[]'
#SELF_ATTENTION_SKIPS='[],[0],[1],[2],[3],[4],[5],[6],[7],[8],[9],[10],[11]'
MLP_SKIPS='[],[0],[1],[2],[3],[4],[5],[6],[7],[8],[9],[10],[11]'
SELF_ATTENTION_SKIPS='[]'

for dataset_name in "${DATASET_NAME:-mnist}"
do
    for encoder_name in "${ENCODER_NAME:-google/vit-base-patch16-224}"
    do
        for translator_name in linear
        do
            for samples_to_extract in 500
            do
                python src/toast/utils/encode_vision_full.py \
                    --dataset_name=$dataset_name \
                    --encoder_name="$encoder_name" \
                    --translator_name=$translator_name \
                    --seed=0 \
                    --skips="$SKIPS" \
                    --mlp_skips="$MLP_SKIPS" \
                    --attention_skips="$SELF_ATTENTION_SKIPS" \
                    --samples_to_extract=$samples_to_extract
            done
        done
    done
done
