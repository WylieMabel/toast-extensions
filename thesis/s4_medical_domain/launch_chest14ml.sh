#!/bin/bash
# Full multilabel ChestX-ray14 sweep: 39 rows (baseline + contiguous skips len 1-4, linear
# translator, seeds 0 1 2) per encoder, one job each, run in parallel. Run from the project root:
#   bash thesis/s4_medical_domain/launch_chest14ml.sh
#
# Fill in the values below from launch_chest14ml_smoke.sh first:
#   TIME ~ 1.3 x (smoke elapsed x 39)   rows run 2-17% slower than their smoke test in s4
#   MEM  ~ 1.5 x smoke MaxRSS           sacct can miss short spikes (see STATUS.md)
#   CPUS ~ smoke TotalCPU / Elapsed, rounded up
#
# Smoke tests 9742856-58 (2026-10-14, 8 CPUs / 64G each), per row of 3 seeds:
#                     elapsed   x39     cores used   MaxRSS
#   rad-dino          53:19     34.7h   3.4          13.4G
#   rad-dino-maira-2  53:21     34.7h   3.4          13.6G
#   dinov2-base       16:20     10.6h   5.6          12.3G
# CPUs stay at the binary chestxray14 values (6 / 6 / 8): NUM_WORKERS is CPUs - 1 and image
# loading is the bottleneck, so cutting cores further would slow the rows down. Memory is kept
# at the binary chestxray14 s4 requests (48 / 48 / 28G) rather than sized from these peaks:
# sacct missed a >15G spike on rad-dino before (33G reported, OOM at 48G), and nothing has
# been tested below 48G since the memfix.
#
# The baseline row is first in each CSV, so delta_auroc / delta_auprc are filled in for every
# row. Results: thesis/s4_medical_domain/results/results_<enc>_chestxray14multilabel.csv

set -e

TIME_RADDINO="48:00:00";        CPUS_RADDINO=6;        MEM_RADDINO="48G"
TIME_RADDINOMAIRA2="48:00:00";  CPUS_RADDINOMAIRA2=6;  MEM_RADDINOMAIRA2="48G"
TIME_DINOV2BASE="16:00:00";     CPUS_DINOV2BASE=8;     MEM_DINOV2BASE="28G"

MAIL_USER="mabel.wylie@gmail.com"
MAIL_TYPE="BEGIN,END"

LOCAL_DISK="20G"
export LOCAL_EMBEDDINGS=1

unset RESULTS_CSV_NAME RESULTS_DIR ACCURACIES_DIR CONFIG_CSV EMBEDDINGS_DIR SEEDS NUM_WORKERS

CONFIGS=thesis/s4_medical_domain/configs
COMMON="--parsable --tmp=$LOCAL_DISK --mail-type=$MAIL_TYPE --mail-user=$MAIL_USER"

mkdir -p logs

J1=$(sbatch $COMMON --job-name=s4_raddino_chestxray14ml --time=$TIME_RADDINO --cpus-per-task=$CPUS_RADDINO --mem=$MEM_RADDINO --output=logs/s4_raddino_chestxray14ml_%j.out.txt \
    src/run_scripts/run_pipeline_row_by_row.sh $CONFIGS/experiments_raddino_chestxray14multilabel.csv)
echo "raddino_chestxray14multilabel -> $J1"

J2=$(sbatch $COMMON --job-name=s4_raddinomaira2_chestxray14ml --time=$TIME_RADDINOMAIRA2 --cpus-per-task=$CPUS_RADDINOMAIRA2 --mem=$MEM_RADDINOMAIRA2 --output=logs/s4_raddinomaira2_chestxray14ml_%j.out.txt \
    src/run_scripts/run_pipeline_row_by_row.sh $CONFIGS/experiments_raddinomaira2_chestxray14multilabel.csv)
echo "raddinomaira2_chestxray14multilabel -> $J2"

J3=$(sbatch $COMMON --job-name=s4_dinov2base_chestxray14ml --time=$TIME_DINOV2BASE --cpus-per-task=$CPUS_DINOV2BASE --mem=$MEM_DINOV2BASE --output=logs/s4_dinov2base_chestxray14ml_%j.out.txt \
    src/run_scripts/run_pipeline_row_by_row.sh $CONFIGS/experiments_dinov2base_chestxray14multilabel.csv)
echo "dinov2base_chestxray14multilabel -> $J3"
