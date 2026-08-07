#!/bin/bash

export HF_HOME="/cluster/customapps/biomed/vogtlab/users/mwylie/toast/hf_cache"
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

CONFIG_CSV="${CONFIG_CSV:-src/configs/experiments.csv}"

# Seeds to average each config over. Override without editing this file, e.g. SEEDS="1 2 3 4 5".
SEEDS="${SEEDS:-1 2 3}"

FAILED_SEEDS=""

for seed in $SEEDS
do
    python src/toast/utils/train_skipped_full.py \
        --seed=$seed \
        --classifier_type=linear \
        --translator_name=identity \
        --samples_to_extract=250 \
        --config_csv="$CONFIG_CSV"
    [ $? -ne 0 ] && FAILED_SEEDS="$FAILED_SEEDS $seed"
done

# A bare loop exits with the status of the LAST seed only, so a failure on an earlier seed was
# invisible to the caller and silently produced a results CSV with missing seeds.
if [ -n "$FAILED_SEEDS" ]; then
    echo "ERROR: training failed for seed(s):$FAILED_SEEDS"
    exit 1
fi
