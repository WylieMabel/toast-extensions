#!/bin/bash

declare -a skips=("[]" "[(0, 1)]")

for dataset_name in mnist
do
    for seed in 0 
    do
        for classifier_type in linear
        do
            for translator_name in linear
            do
                for model_name in "facebook/deit-small-patch16-224"
                do
                    for to_approximate in "${skips[@]}"
                    do
                        for samples_to_extract in 500
                        do
                            python src/toast/utils/train_skipped.py \
                                --dataset_name=$dataset_name \
                                --model_name=$model_name \
                                --layers_to_approximate="$to_approximate" \
                                --seed=$seed \
                                --classifier_type=$classifier_type \
                                --translator_name=$translator_name \
                                --samples_to_extract=$samples_to_extract
                            echo
                        done
                    done
                done
            done
        done
    done
done
