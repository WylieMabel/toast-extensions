#!/bin/bash
# Submits one run_pipeline_row_by_row.sh job per s2 config. Run from the project root:
#   bash thesis/s2_skip_cost_metrics/launch_s2.sh
#
# Defaults inherited from run_pipeline_row_by_row.sh (flags below override them per job):
#   partition gpu, 1x rtx4090, 2 CPUs, 96000 MB memory, 12:00:00, seeds 0 1 2, 500 samples

SMALL_TIME="12:00:00"
VITLARGE_TIME="36:00:00"
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
unset RESULTS_CSV_NAME RESULTS_DIR ACCURACIES_DIR CONFIG_CSV EMBEDDINGS_DIR NUM_WORKERS

mkdir -p logs

# --- cifar100 ---
# deitsmall_cifar100 is launched separately as the first full-run test.

sbatch --job-name=s2_vitsmall_cifar100 --time=$SMALL_TIME --mem=$MEM_VITSMALL --cpus-per-task=$CPUS --tmp=$LOCAL_DISK --mail-type=$MAIL_TYPE --mail-user=$MAIL_USER --output=logs/s2_vitsmall_cifar100_%j.out.txt \
    src/run_scripts/run_pipeline_row_by_row.sh thesis/s2_skip_cost_metrics/configs/experiments_vitsmall_cifar100.csv

sbatch --job-name=s2_dinov2base_cifar100 --time=$SMALL_TIME --mem=$MEM_DINOV2BASE --cpus-per-task=$CPUS --tmp=$LOCAL_DISK --mail-type=$MAIL_TYPE --mail-user=$MAIL_USER --output=logs/s2_dinov2base_cifar100_%j.out.txt \
    src/run_scripts/run_pipeline_row_by_row.sh thesis/s2_skip_cost_metrics/configs/experiments_dinov2base_cifar100.csv

sbatch --job-name=s2_vitlarge_cifar100 --time=$VITLARGE_TIME --mem=$MEM_VITLARGE --cpus-per-task=$CPUS --tmp=$LOCAL_DISK --mail-type=$MAIL_TYPE --mail-user=$MAIL_USER --output=logs/s2_vitlarge_cifar100_%j.out.txt \
    src/run_scripts/run_pipeline_row_by_row.sh thesis/s2_skip_cost_metrics/configs/experiments_vitlarge_cifar100.csv

# --- imagenet-1k ---
sbatch --job-name=s2_deitsmall_imagenet1k --time=$SMALL_TIME --mem=$MEM_DEITSMALL --cpus-per-task=$CPUS --tmp=$LOCAL_DISK --mail-type=$MAIL_TYPE --mail-user=$MAIL_USER --output=logs/s2_deitsmall_imagenet1k_%j.out.txt \
    src/run_scripts/run_pipeline_row_by_row.sh thesis/s2_skip_cost_metrics/configs/experiments_deitsmall_imagenet1k.csv

sbatch --job-name=s2_vitsmall_imagenet1k --time=$SMALL_TIME --mem=$MEM_VITSMALL --cpus-per-task=$CPUS --tmp=$LOCAL_DISK --mail-type=$MAIL_TYPE --mail-user=$MAIL_USER --output=logs/s2_vitsmall_imagenet1k_%j.out.txt \
    src/run_scripts/run_pipeline_row_by_row.sh thesis/s2_skip_cost_metrics/configs/experiments_vitsmall_imagenet1k.csv

sbatch --job-name=s2_dinov2base_imagenet1k --time=$SMALL_TIME --mem=$MEM_DINOV2BASE --cpus-per-task=$CPUS --tmp=$LOCAL_DISK --mail-type=$MAIL_TYPE --mail-user=$MAIL_USER --output=logs/s2_dinov2base_imagenet1k_%j.out.txt \
    src/run_scripts/run_pipeline_row_by_row.sh thesis/s2_skip_cost_metrics/configs/experiments_dinov2base_imagenet1k.csv

sbatch --job-name=s2_vitlarge_imagenet1k --time=$VITLARGE_TIME --mem=$MEM_VITLARGE --cpus-per-task=$CPUS --tmp=$LOCAL_DISK --mail-type=$MAIL_TYPE --mail-user=$MAIL_USER --output=logs/s2_vitlarge_imagenet1k_%j.out.txt \
    src/run_scripts/run_pipeline_row_by_row.sh thesis/s2_skip_cost_metrics/configs/experiments_vitlarge_imagenet1k.csv
