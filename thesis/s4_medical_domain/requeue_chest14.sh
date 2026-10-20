#!/bin/bash
# Resubmits the four rad-dino / rad-dino-maira-2 chestxray14 jobs that OOM'd on every row (before
# the host-RAM fixes in utils.py / encode_vision_full.py), appended to the end of their original
# s4 groups. Run from the project root:
#   bash thesis/s4_medical_domain/requeue_chest14.sh
#
# Same settings as launch_s4.sh; 48G passed the memfix smoke test for rad-dino / chestxray14.

set -e

AFTER_G1=9728295   # s4_g1_phikonv2_patchcamelyon_len2 (last job of group 1)
AFTER_G2=9728300   # s4_g2_vitlarge_patchcamelyon_len1 (last job of group 2)

MEDIUM_TIME="24:00:00"   # len34: estimated ~14.9h
LONG_TIME="36:00:00"     # len12: estimated ~19.3h
CPUS=6
MEM="48G"

MAIL_USER="mabel.wylie@gmail.com"
MAIL_TYPE="BEGIN,END"

LOCAL_DISK="20G"
export LOCAL_EMBEDDINGS=1

unset RESULTS_CSV_NAME RESULTS_DIR ACCURACIES_DIR CONFIG_CSV EMBEDDINGS_DIR SEEDS NUM_WORKERS

CONFIGS=thesis/s4_medical_domain/configs
COMMON="--parsable --cpus-per-task=$CPUS --mem=$MEM --tmp=$LOCAL_DISK --mail-type=$MAIL_TYPE --mail-user=$MAIL_USER"

mkdir -p logs

# --- group 1 (appended) ---
G1_6=$(sbatch $COMMON --job-name=s4_g1_raddino_chestxray14_len12 --time=$LONG_TIME --dependency=afterany:$AFTER_G1 --output=logs/s4_g1_raddino_chestxray14_len12_%j.out.txt \
    src/run_scripts/run_pipeline_row_by_row.sh $CONFIGS/experiments_raddino_chestxray14_len12.csv)
echo "group 1: raddino_chestxray14_len12 -> $G1_6"
G1_7=$(sbatch $COMMON --job-name=s4_g1_raddinomaira2_chestxray14_len12 --time=$LONG_TIME --dependency=afterany:$G1_6 --output=logs/s4_g1_raddinomaira2_chestxray14_len12_%j.out.txt \
    src/run_scripts/run_pipeline_row_by_row.sh $CONFIGS/experiments_raddinomaira2_chestxray14_len12.csv)
echo "group 1: raddinomaira2_chestxray14_len12 -> $G1_7"

# --- group 2 (appended) ---
G2_6=$(sbatch $COMMON --job-name=s4_g2_raddino_chestxray14_len34 --time=$MEDIUM_TIME --dependency=afterany:$AFTER_G2 --output=logs/s4_g2_raddino_chestxray14_len34_%j.out.txt \
    src/run_scripts/run_pipeline_row_by_row.sh $CONFIGS/experiments_raddino_chestxray14_len34.csv)
echo "group 2: raddino_chestxray14_len34 -> $G2_6"
G2_7=$(sbatch $COMMON --job-name=s4_g2_raddinomaira2_chestxray14_len34 --time=$MEDIUM_TIME --dependency=afterany:$G2_6 --output=logs/s4_g2_raddinomaira2_chestxray14_len34_%j.out.txt \
    src/run_scripts/run_pipeline_row_by_row.sh $CONFIGS/experiments_raddinomaira2_chestxray14_len34.csv)
echo "group 2: raddinomaira2_chestxray14_len34 -> $G2_7"
