#!/bin/bash
# Checks the two host-RAM fixes (extract_representations frees per-layer batch pieces; the
# 500-sample capture is dropped once the translator is fitted). Run from the project root:
#   bash thesis/smoke_timing/launch_memfix_test.sh
#
# Same single row (skip [(4, 6)], seeds 0 1 2) as the original timing smoke tests, at the
# exact memory/CPUs the s4 jobs that OOM'd were given:
#   rad-dino / chestxray14:  48G, 6 CPUs -- every row OOM'd ~40s in (capture concat spike)
#   vit-large / tcga_ut:     36G, 4 CPUs -- ~40% of seeds OOM'd at the end of the full encode
#
# Pass: both finish with no "Killed" / oom-kill lines, and the accuracies match the original
# smoke runs exactly (the fixes only free memory after its last use):
#   rad-dino / chestxray14:  0.6970, 0.6887, 0.7020   (job 9726027)
#   vit-large / tcga_ut:     0.5472, 0.5528, 0.5535   (job 9726023)

TIME="02:00:00"

MAIL_USER="mabel.wylie@gmail.com"
MAIL_TYPE="BEGIN,END"

LOCAL_DISK="20G"
export LOCAL_EMBEDDINGS=1

unset RESULTS_CSV_NAME RESULTS_DIR ACCURACIES_DIR CONFIG_CSV EMBEDDINGS_DIR SEEDS NUM_WORKERS

mkdir -p logs

sbatch --job-name=memfix_raddino_chestxray14 --time=$TIME --cpus-per-task=6 --mem=48G --tmp=$LOCAL_DISK --mail-type=$MAIL_TYPE --mail-user=$MAIL_USER --output=logs/memfix_raddino_chestxray14_%j.out.txt \
    src/run_scripts/run_pipeline_row_by_row.sh thesis/smoke_timing/configs/experiments_memfix_raddino_chestxray14.csv

sbatch --job-name=memfix_vitlarge_tcgaut --time=$TIME --cpus-per-task=4 --mem=36G --tmp=$LOCAL_DISK --mail-type=$MAIL_TYPE --mail-user=$MAIL_USER --output=logs/memfix_vitlarge_tcgaut_%j.out.txt \
    src/run_scripts/run_pipeline_row_by_row.sh thesis/smoke_timing/configs/experiments_memfix_vitlarge_tcgaut.csv
