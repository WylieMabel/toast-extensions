#!/bin/bash
# One-row timing smoke tests (skip [(4, 6)], seeds 0 1 2) per model x dataset, to estimate how
# long the full sweeps take. Run from the project root:
#   bash thesis/smoke_timing/launch_smoke_timing.sh
#
# Defaults inherited from run_pipeline_row_by_row.sh (flags below override them per job):
#   partition gpu, 1x rtx4090, 2 CPUs, 96000 MB memory, 12:00:00, seeds 0 1 2, 500 samples

NATURAL_TIME="01:00:00"
LARGE_IMAGENET_TIME="02:00:00"
MEDICAL_TIME="04:00:00"
# encode_vision_full.sh defaults to NUM_WORKERS=7; give the loader enough cores to use them.
CPUS=8

MAIL_USER="mwylie@ethz.ch"
MAIL_TYPE="BEGIN,END"

# Embeddings go to the node's local /scratch (home quota is only 21 GB). --tmp keeps jobs off
# nodes without local disk (gpu-biomed-02, -04).
LOCAL_DISK="20G"
export LOCAL_EMBEDDINGS=1

# sbatch forwards this shell's environment; stale values would override the per-config
# results name and the section results/ folder.
unset RESULTS_CSV_NAME RESULTS_DIR ACCURACIES_DIR CONFIG_CSV EMBEDDINGS_DIR SEEDS

mkdir -p logs

# --- cifar100 ---
sbatch --job-name=smoke_deitsmall_cifar100 --time=$NATURAL_TIME --cpus-per-task=$CPUS --tmp=$LOCAL_DISK --mail-type=$MAIL_TYPE --mail-user=$MAIL_USER --output=logs/smoke_deitsmall_cifar100_%j.out.txt \
    src/run_scripts/run_pipeline_row_by_row.sh thesis/smoke_timing/configs/experiments_deitsmall_cifar100.csv

sbatch --job-name=smoke_vitsmall_cifar100 --time=$NATURAL_TIME --cpus-per-task=$CPUS --tmp=$LOCAL_DISK --mail-type=$MAIL_TYPE --mail-user=$MAIL_USER --output=logs/smoke_vitsmall_cifar100_%j.out.txt \
    src/run_scripts/run_pipeline_row_by_row.sh thesis/smoke_timing/configs/experiments_vitsmall_cifar100.csv

sbatch --job-name=smoke_dinov2base_cifar100 --time=$NATURAL_TIME --cpus-per-task=$CPUS --tmp=$LOCAL_DISK --mail-type=$MAIL_TYPE --mail-user=$MAIL_USER --output=logs/smoke_dinov2base_cifar100_%j.out.txt \
    src/run_scripts/run_pipeline_row_by_row.sh thesis/smoke_timing/configs/experiments_dinov2base_cifar100.csv

sbatch --job-name=smoke_vitlarge_cifar100 --time=$NATURAL_TIME --cpus-per-task=$CPUS --tmp=$LOCAL_DISK --mail-type=$MAIL_TYPE --mail-user=$MAIL_USER --output=logs/smoke_vitlarge_cifar100_%j.out.txt \
    src/run_scripts/run_pipeline_row_by_row.sh thesis/smoke_timing/configs/experiments_vitlarge_cifar100.csv

# --- imagenet-1k ---
sbatch --job-name=smoke_deitsmall_imagenet1k --time=$NATURAL_TIME --cpus-per-task=$CPUS --tmp=$LOCAL_DISK --mail-type=$MAIL_TYPE --mail-user=$MAIL_USER --output=logs/smoke_deitsmall_imagenet1k_%j.out.txt \
    src/run_scripts/run_pipeline_row_by_row.sh thesis/smoke_timing/configs/experiments_deitsmall_imagenet1k.csv

sbatch --job-name=smoke_vitsmall_imagenet1k --time=$NATURAL_TIME --cpus-per-task=$CPUS --tmp=$LOCAL_DISK --mail-type=$MAIL_TYPE --mail-user=$MAIL_USER --output=logs/smoke_vitsmall_imagenet1k_%j.out.txt \
    src/run_scripts/run_pipeline_row_by_row.sh thesis/smoke_timing/configs/experiments_vitsmall_imagenet1k.csv

sbatch --job-name=smoke_dinov2base_imagenet1k --time=$LARGE_IMAGENET_TIME --cpus-per-task=$CPUS --tmp=$LOCAL_DISK --mail-type=$MAIL_TYPE --mail-user=$MAIL_USER --output=logs/smoke_dinov2base_imagenet1k_%j.out.txt \
    src/run_scripts/run_pipeline_row_by_row.sh thesis/smoke_timing/configs/experiments_dinov2base_imagenet1k.csv

sbatch --job-name=smoke_vitlarge_imagenet1k --time=$LARGE_IMAGENET_TIME --cpus-per-task=$CPUS --tmp=$LOCAL_DISK --mail-type=$MAIL_TYPE --mail-user=$MAIL_USER --output=logs/smoke_vitlarge_imagenet1k_%j.out.txt \
    src/run_scripts/run_pipeline_row_by_row.sh thesis/smoke_timing/configs/experiments_vitlarge_imagenet1k.csv

# --- tcga_ut ---
sbatch --job-name=smoke_vitlarge_tcgaut --time=$MEDICAL_TIME --cpus-per-task=$CPUS --tmp=$LOCAL_DISK --mail-type=$MAIL_TYPE --mail-user=$MAIL_USER --output=logs/smoke_vitlarge_tcgaut_%j.out.txt \
    src/run_scripts/run_pipeline_row_by_row.sh thesis/smoke_timing/configs/experiments_vitlarge_tcgaut.csv

sbatch --job-name=smoke_phikonv2_tcgaut --time=$MEDICAL_TIME --cpus-per-task=$CPUS --tmp=$LOCAL_DISK --mail-type=$MAIL_TYPE --mail-user=$MAIL_USER --output=logs/smoke_phikonv2_tcgaut_%j.out.txt \
    src/run_scripts/run_pipeline_row_by_row.sh thesis/smoke_timing/configs/experiments_phikonv2_tcgaut.csv

# --- patchcamelyon ---
sbatch --job-name=smoke_vitlarge_patchcamelyon --time=$MEDICAL_TIME --cpus-per-task=$CPUS --tmp=$LOCAL_DISK --mail-type=$MAIL_TYPE --mail-user=$MAIL_USER --output=logs/smoke_vitlarge_patchcamelyon_%j.out.txt \
    src/run_scripts/run_pipeline_row_by_row.sh thesis/smoke_timing/configs/experiments_vitlarge_patchcamelyon.csv

sbatch --job-name=smoke_phikonv2_patchcamelyon --time=$MEDICAL_TIME --cpus-per-task=$CPUS --tmp=$LOCAL_DISK --mail-type=$MAIL_TYPE --mail-user=$MAIL_USER --output=logs/smoke_phikonv2_patchcamelyon_%j.out.txt \
    src/run_scripts/run_pipeline_row_by_row.sh thesis/smoke_timing/configs/experiments_phikonv2_patchcamelyon.csv

# --- chestxray14 ---
sbatch --job-name=smoke_raddino_chestxray14 --time=$MEDICAL_TIME --cpus-per-task=$CPUS --tmp=$LOCAL_DISK --mail-type=$MAIL_TYPE --mail-user=$MAIL_USER --output=logs/smoke_raddino_chestxray14_%j.out.txt \
    src/run_scripts/run_pipeline_row_by_row.sh thesis/smoke_timing/configs/experiments_raddino_chestxray14.csv

sbatch --job-name=smoke_raddinomaira2_chestxray14 --time=$MEDICAL_TIME --cpus-per-task=$CPUS --tmp=$LOCAL_DISK --mail-type=$MAIL_TYPE --mail-user=$MAIL_USER --output=logs/smoke_raddinomaira2_chestxray14_%j.out.txt \
    src/run_scripts/run_pipeline_row_by_row.sh thesis/smoke_timing/configs/experiments_raddinomaira2_chestxray14.csv

sbatch --job-name=smoke_dinov2base_chestxray14 --time=$MEDICAL_TIME --cpus-per-task=$CPUS --tmp=$LOCAL_DISK --mail-type=$MAIL_TYPE --mail-user=$MAIL_USER --output=logs/smoke_dinov2base_chestxray14_%j.out.txt \
    src/run_scripts/run_pipeline_row_by_row.sh thesis/smoke_timing/configs/experiments_dinov2base_chestxray14.csv

# --- chestxray_pneumonia ---
sbatch --job-name=smoke_raddino_chestxraypneumonia --time=$MEDICAL_TIME --cpus-per-task=$CPUS --tmp=$LOCAL_DISK --mail-type=$MAIL_TYPE --mail-user=$MAIL_USER --output=logs/smoke_raddino_chestxraypneumonia_%j.out.txt \
    src/run_scripts/run_pipeline_row_by_row.sh thesis/smoke_timing/configs/experiments_raddino_chestxraypneumonia.csv

sbatch --job-name=smoke_raddinomaira2_chestxraypneumonia --time=$MEDICAL_TIME --cpus-per-task=$CPUS --tmp=$LOCAL_DISK --mail-type=$MAIL_TYPE --mail-user=$MAIL_USER --output=logs/smoke_raddinomaira2_chestxraypneumonia_%j.out.txt \
    src/run_scripts/run_pipeline_row_by_row.sh thesis/smoke_timing/configs/experiments_raddinomaira2_chestxraypneumonia.csv

sbatch --job-name=smoke_dinov2base_chestxraypneumonia --time=$MEDICAL_TIME --cpus-per-task=$CPUS --tmp=$LOCAL_DISK --mail-type=$MAIL_TYPE --mail-user=$MAIL_USER --output=logs/smoke_dinov2base_chestxraypneumonia_%j.out.txt \
    src/run_scripts/run_pipeline_row_by_row.sh thesis/smoke_timing/configs/experiments_dinov2base_chestxraypneumonia.csv
