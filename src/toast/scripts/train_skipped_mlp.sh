#!/bin/bash

declare -a skips=("[]" "[(0, 1)]")
declare -a mlp_skips=("[]" "[1,2]" "[3,4]" "[5,6]" "[7,8]" "[9,10]" "[2,4,6,8,10]" "[1,3,5,7,9]")

export HF_HOME="/cluster/customapps/biomed/vogtlab/users/mwylie/toast/hf_cache"
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

for dataset_name in mnist
do
    for seed in 0
    do
        for classifier_type in linear
        do
            for translator_name in linear
            do
                for model_name in "google/vit-base-patch16-224"
                do
                    for to_approximate in "${skips[@]}"
                    do
                        for mlp_to_linearize in "${mlp_skips[@]}"
                        do
                            echo "=== MLP | dataset=$dataset_name | model=$model_name | skip=$to_approximate | mlp=$mlp_to_linearize | translator=$translator_name | seed=$seed ==="
                            python src/toast/utils/train_skipped_mlp.py \
                                --dataset_name=$dataset_name \
                                --model_name=$model_name \
                                --layers_to_approximate="$to_approximate" \
                                --mlp_layers_to_linearize="$mlp_to_linearize" \
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
