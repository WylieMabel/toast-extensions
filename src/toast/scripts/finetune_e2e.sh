#!/bin/bash

declare -a skips=("[]" "[(0, 1)]")

for dataset_name in mnist
do
    for model_name in facebook/deit-small-patch16-224
    do
        if [[ "$model_name" == *"dinov2"* ]]; then
            encoder_lr_flag="--encoder_lr=1e-6"
        else
            encoder_lr_flag=""
        fi

        for translator_name in linear
        do
            for seed in 0
            do
                for to_approximate in "${skips[@]}"
                do
                    echo "=== E2E | dataset=$dataset_name | model=$model_name | skip=$to_approximate | translator=$translator_name | seed=$seed ==="
                    python src/toast/utils/finetune_e2e.py \
                        --dataset_name="$dataset_name" \
                        --model_name="$model_name" \
                        --layers_to_approximate="$to_approximate" \
                        --seed=$seed \
                        --use_pretrained_head=False \
                        --classifier_type=linear \
                        --translator_name="$translator_name" \
                        --samples_to_extract=500 \
                        --lr=2e-4 \
                        $encoder_lr_flag \
                        --num_epochs=2 \
                        --batch_size=128 \
                        --weight_decay=1e-4 \
                        --results_file=finetune_e2e.csv
                    echo
                done
            done
        done
    done
done
