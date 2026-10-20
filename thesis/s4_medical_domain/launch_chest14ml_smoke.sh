#!/bin/bash
# One-row smoke tests for the multilabel ChestX-ray14 sweep (skip [(4, 6)], seeds 0 1 2), one per
# encoder, to measure time / memory / CPU before launch_chest14ml.sh. Run from the project root:
#   bash thesis/s4_medical_domain/launch_chest14ml_smoke.sh
#
# Resources are deliberately generous (the binary chestxray14 peaks were 15.6-33.0G, 3.4-5.5
# cores) so the measurement isn't capped by the request. Writes to
# thesis/s4_medical_domain/results/results_<enc>_chestxray14multilabel_smoke.csv, separate from
# the full sweep's results.
#
# Needs /cluster/.../toast/chestxray14_multilabel_clean to exist first
# (python claude_out/download_medical_datasets.py chestxray14_multilabel).

set -e

SMOKE_TIME="04:00:00"
CPUS=8
MEM="64G"

MAIL_USER="mabel.wylie@gmail.com"
MAIL_TYPE="END"

LOCAL_DISK="20G"
export LOCAL_EMBEDDINGS=1

unset RESULTS_CSV_NAME RESULTS_DIR ACCURACIES_DIR CONFIG_CSV EMBEDDINGS_DIR SEEDS NUM_WORKERS

CONFIGS=thesis/s4_medical_domain/configs
COMMON="--parsable --time=$SMOKE_TIME --cpus-per-task=$CPUS --mem=$MEM --tmp=$LOCAL_DISK --mail-type=$MAIL_TYPE --mail-user=$MAIL_USER"

mkdir -p logs

for enc in raddino raddinomaira2 dinov2base; do
    JOB=$(sbatch $COMMON --job-name=smoke_${enc}_chestxray14ml --output=logs/smoke_${enc}_chestxray14ml_%j.out.txt \
        src/run_scripts/run_pipeline_row_by_row.sh $CONFIGS/experiments_${enc}_chestxray14multilabel_smoke.csv)
    echo "smoke $enc -> $JOB"
done
