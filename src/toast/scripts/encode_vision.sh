#!/bin/bash

SKIPS='[[], [(0, 1)]]'

for dataset_name in mnist
do
    for encoder_name in "facebook/deit-small-patch16-224"
    do
        for translator_name in linear
        do
            for samples_to_extract in 500
            do
                python src/toast/utils/encode_vision.py \
                    --dataset_name=$dataset_name \
                    --encoder_name=$encoder_name \
                    --translator_name=$translator_name \
                    --seed=0 \
                    --skips="$SKIPS" \
                    --samples_to_extract=$samples_to_extract
            done
        done
    done
done
