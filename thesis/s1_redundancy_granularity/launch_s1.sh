#!/bin/bash
# Submits one run_pipeline_row_by_row.sh job per s1 head-pruning config (head_dicts from
# head_similarity_thresholds.py, CKA + angular distance on imagenet-1k). Run from the project root:
#   bash thesis/s1_redundancy_granularity/launch_s1.sh
#
# Defaults inherited from run_pipeline_row_by_row.sh (flags below override them per job):
#   partition gpu, 1x rtx4090, 2 CPUs, 96000 MB memory, 12:00:00, seeds 0 1 2, 500 samples
#
# Estimated job lengths = rows x the imagenet-1k smoke-test time per row (3 seeds, thesis/smoke_timing):
#   vitsmall  21 x 362s = 2.1h    deitsmall 19 x 377s = 2.0h
#   dinov2base 26 x 506s = 3.7h   vitlarge  19 x 855s = 4.5h
# Every job is estimated to finish at least 7h before its limit.

TIME="12:00:00"
# Smoke tests used 2-3 cores even with 8 allocated (encoding is GPU-bound at batch 8);
# run_pipeline_row_by_row.sh sets NUM_WORKERS to CPUS - 1.
CPUS=4

# --mem is ~1.5x the peak memory measured in the smoke tests (sacct MaxRSS):
#   deitsmall 4.4-6.5G, vitsmall 6.7-6.9G, dinov2base 16.1-16.5G, vitlarge 28.4-29.6G
MEM_DEITSMALL="12G"
MEM_VITSMALL="12G"
MEM_DINOV2BASE="24G"
MEM_VITLARGE="48G"

MAIL_USER="mabel.wylie@gmail.com"
MAIL_TYPE="BEGIN,END"

# Embeddings go to the node's local /scratch (home quota is only 21 GB). --tmp keeps jobs off
# nodes without local disk (gpu-biomed-02, -04).
LOCAL_DISK="20G"
export LOCAL_EMBEDDINGS=1

# sbatch forwards this shell's environment; stale values would override the per-config
# results name and the section results/ folder.
unset RESULTS_CSV_NAME RESULTS_DIR ACCURACIES_DIR CONFIG_CSV EMBEDDINGS_DIR SEEDS NUM_WORKERS

CONFIGS=thesis/s1_redundancy_granularity/configs
COMMON="--time=$TIME --cpus-per-task=$CPUS --tmp=$LOCAL_DISK --mail-type=$MAIL_TYPE --mail-user=$MAIL_USER"

mkdir -p logs

# --- imagenet-1k ---
sbatch $COMMON --job-name=s1_heads_vitsmall_imagenet1k --mem=$MEM_VITSMALL --output=logs/s1_heads_vitsmall_imagenet1k_%j.out.txt \
    src/run_scripts/run_pipeline_row_by_row.sh $CONFIGS/experiments_heads_vitsmall_imagenet1k.csv

sbatch $COMMON --job-name=s1_heads_deitsmall_imagenet1k --mem=$MEM_DEITSMALL --output=logs/s1_heads_deitsmall_imagenet1k_%j.out.txt \
    src/run_scripts/run_pipeline_row_by_row.sh $CONFIGS/experiments_heads_deitsmall_imagenet1k.csv

sbatch $COMMON --job-name=s1_heads_dinov2base_imagenet1k --mem=$MEM_DINOV2BASE --output=logs/s1_heads_dinov2base_imagenet1k_%j.out.txt \
    src/run_scripts/run_pipeline_row_by_row.sh $CONFIGS/experiments_heads_dinobase_imagenet1k.csv

sbatch $COMMON --job-name=s1_heads_vitlarge_imagenet1k --mem=$MEM_VITLARGE --output=logs/s1_heads_vitlarge_imagenet1k_%j.out.txt \
    src/run_scripts/run_pipeline_row_by_row.sh $CONFIGS/experiments_heads_vitlarge_imagenet1k.csv
