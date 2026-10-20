#!/bin/bash
# Submits one run_pipeline_row_by_row.sh job per s3 config. Run from the project root:
#   bash thesis/s3_multiskip_prediction/launch_s3.sh
#
# Defaults inherited from run_pipeline_row_by_row.sh (flags below override them per job):
#   partition gpu, 1x rtx4090, 2 CPUs, 96000 MB memory, 12:00:00, seeds 0 1 2, 500 samples

SMALL_TIME="12:00:00"
VITLARGE_TIME="24:00:00"
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

mkdir -p logs

# --- imagenet-1k, 12-block models ---
sbatch --job-name=s3_deitsmall_imagenet1k --time=$SMALL_TIME --mem=$MEM_DEITSMALL --cpus-per-task=$CPUS --tmp=$LOCAL_DISK --mail-type=$MAIL_TYPE --mail-user=$MAIL_USER --output=logs/s3_deitsmall_imagenet1k_%j.out.txt \
    src/run_scripts/run_pipeline_row_by_row.sh thesis/s3_multiskip_prediction/configs/experiments_deitsmall_imagenet1k.csv

sbatch --job-name=s3_vitsmall_imagenet1k --time=$SMALL_TIME --mem=$MEM_VITSMALL --cpus-per-task=$CPUS --tmp=$LOCAL_DISK --mail-type=$MAIL_TYPE --mail-user=$MAIL_USER --output=logs/s3_vitsmall_imagenet1k_%j.out.txt \
    src/run_scripts/run_pipeline_row_by_row.sh thesis/s3_multiskip_prediction/configs/experiments_vitsmall_imagenet1k.csv

sbatch --job-name=s3_dinov2base_imagenet1k --time=$SMALL_TIME --mem=$MEM_DINOV2BASE --cpus-per-task=$CPUS --tmp=$LOCAL_DISK --mail-type=$MAIL_TYPE --mail-user=$MAIL_USER --output=logs/s3_dinov2base_imagenet1k_%j.out.txt \
    src/run_scripts/run_pipeline_row_by_row.sh thesis/s3_multiskip_prediction/configs/experiments_dinov2base_imagenet1k.csv

# --- imagenet-1k, vit-large (split into 4 parts) ---
sbatch --job-name=s3_vitlarge_imagenet1k_part1 --time=$VITLARGE_TIME --mem=$MEM_VITLARGE --cpus-per-task=$CPUS --tmp=$LOCAL_DISK --mail-type=$MAIL_TYPE --mail-user=$MAIL_USER --output=logs/s3_vitlarge_imagenet1k_part1_%j.out.txt \
    src/run_scripts/run_pipeline_row_by_row.sh thesis/s3_multiskip_prediction/configs/experiments_vitlarge_imagenet1k_part1.csv

sbatch --job-name=s3_vitlarge_imagenet1k_part2 --time=$VITLARGE_TIME --mem=$MEM_VITLARGE --cpus-per-task=$CPUS --tmp=$LOCAL_DISK --mail-type=$MAIL_TYPE --mail-user=$MAIL_USER --output=logs/s3_vitlarge_imagenet1k_part2_%j.out.txt \
    src/run_scripts/run_pipeline_row_by_row.sh thesis/s3_multiskip_prediction/configs/experiments_vitlarge_imagenet1k_part2.csv

sbatch --job-name=s3_vitlarge_imagenet1k_part3 --time=$VITLARGE_TIME --mem=$MEM_VITLARGE --cpus-per-task=$CPUS --tmp=$LOCAL_DISK --mail-type=$MAIL_TYPE --mail-user=$MAIL_USER --output=logs/s3_vitlarge_imagenet1k_part3_%j.out.txt \
    src/run_scripts/run_pipeline_row_by_row.sh thesis/s3_multiskip_prediction/configs/experiments_vitlarge_imagenet1k_part3.csv

sbatch --job-name=s3_vitlarge_imagenet1k_part4 --time=$VITLARGE_TIME --mem=$MEM_VITLARGE --cpus-per-task=$CPUS --tmp=$LOCAL_DISK --mail-type=$MAIL_TYPE --mail-user=$MAIL_USER --output=logs/s3_vitlarge_imagenet1k_part4_%j.out.txt \
    src/run_scripts/run_pipeline_row_by_row.sh thesis/s3_multiskip_prediction/configs/experiments_vitlarge_imagenet1k_part4.csv
