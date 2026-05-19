#!/bin/bash

declare -a skips=("[]")
declare -a mlp_skips=("[]")
declare -a attention_skips=("[]" "[0]" "[1]" "[2]" "[3]" "[4]" "[5]" "[6]" "[7]" "[8]" "[9]" "[10]" "[11]")

export HF_HOME="/cluster/customapps/biomed/vogtlab/users/mwylie/toast/hf_cache"
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

for dataset_name in "${DATASET_NAME:-mnist}"
do
    for seed in 0
    do
        for classifier_type in linear
        do
            for translator_name in linear
            do
                for model_name in "${ENCODER_NAME:-google/vit-base-patch16-224}"
                do
                    for to_approximate in "${skips[@]}"
                    do
                        for attn_to_linearize in "${attention_skips[@]}"
                        do
                            for mlp_to_linearize in "${mlp_skips[@]}"
                            do
                                echo "=== FULL | dataset=$dataset_name | skip=$to_approximate | attn=$attn_to_linearize | mlp=$mlp_to_linearize ==="
                                python src/toast/utils/train_skipped_full.py \
                                    --dataset_name=$dataset_name \
                                    --model_name=$model_name \
                                    --layers_to_approximate="$to_approximate" \
                                    --mlp_layers_to_linearize="$mlp_to_linearize" \
                                    --attention_layers_to_linearize="$attn_to_linearize" \
                                    --seed=$seed \
                                    --classifier_type=$classifier_type \
                                    --translator_name=$translator_name \
                                    --samples_to_extract=500
                                echo
                            done
                        done
                    done
                done
            done
        done
    done
done
