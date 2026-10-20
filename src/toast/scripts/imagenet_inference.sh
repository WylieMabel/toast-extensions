#!/bin/bash

declare -a skips=("[]" "[(0, 1)]")

for model_name in "WinKawaks/vit-small-patch16-224"
do
    for to_approximate in "${skips[@]}"
    do
        echo "=== IN1K inference | model=$model_name | skip=$to_approximate ==="
        python src/toast/utils/finetune_e2e.py \
            --dataset_name="imagenet-1k" \
            --model_name="$model_name" \
            --layers_to_approximate="$to_approximate" \
            --seed=0 \
            --use_pretrained_head=True \
            --translator_name="linear" \
            --samples_to_extract=500 \
            --use_wandb=False \
            --results_file="imagenet_inference.csv"
        echo
    done
done
