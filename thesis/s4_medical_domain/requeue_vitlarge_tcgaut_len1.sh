#!/bin/bash
# Resubmits the 11 vit-large / tcga_ut len1 rows that OOM'd in 9728304 (before the host-RAM fixes
# in utils.py / encode_vision_full.py): rows 1 5 6 7 8 10 11 12 13 14 15 of
# experiments_vitlarge_tcgaut_len1.csv, including the baseline (row 1), so delta_acc is filled in.
# Appended to the end of s4 group 4. Run from the project root:
#   bash thesis/s4_medical_domain/requeue_vitlarge_tcgaut_len1.sh
#
# Results go to results_vitlarge_tcgaut_len1_rerun.csv (derived from the config name), so the
# 13 good rows in results_vitlarge_tcgaut_len1.csv are left alone; merge the two in analysis.
# Same settings as launch_s4.sh; 36G passed the memfix smoke test for vit-large / tcga_ut.

set -e

AFTER_G4=9728307   # s4_g4_vitlarge_tcgaut_len4 (last job of group 4)

MEDIUM_TIME="24:00:00"   # 11 rows x ~52 min = ~9.5h
CPUS=4
MEM="36G"

MAIL_USER="mabel.wylie@gmail.com"
MAIL_TYPE="BEGIN,END"

LOCAL_DISK="20G"
export LOCAL_EMBEDDINGS=1

unset RESULTS_CSV_NAME RESULTS_DIR ACCURACIES_DIR CONFIG_CSV EMBEDDINGS_DIR SEEDS NUM_WORKERS

CONFIGS=thesis/s4_medical_domain/configs
COMMON="--parsable --cpus-per-task=$CPUS --mem=$MEM --tmp=$LOCAL_DISK --mail-type=$MAIL_TYPE --mail-user=$MAIL_USER"

mkdir -p logs

# --- group 4 (appended) ---
G4_5=$(sbatch $COMMON --job-name=s4_g4_vitlarge_tcgaut_len1_rerun --time=$MEDIUM_TIME --dependency=afterany:$AFTER_G4 --output=logs/s4_g4_vitlarge_tcgaut_len1_rerun_%j.out.txt \
    src/run_scripts/run_pipeline_row_by_row.sh $CONFIGS/experiments_vitlarge_tcgaut_len1_rerun.csv)
echo "group 4: vitlarge_tcgaut_len1_rerun -> $G4_5"
