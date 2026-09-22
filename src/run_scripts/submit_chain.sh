#!/bin/bash
# Submit every experiments_*.csv in a directory as a sequential sbatch chain -- each job's
# --dependency=afterany:<prev> means the next one only starts once the previous finishes
# (success or failure), so only one job from this directory ever runs at a time.
#
# Usage: src/run_scripts/submit_chain.sh <config_dir> <results_prefix> [time_limit]
#   src/run_scripts/submit_chain.sh src/configs/translator_reuse/raddino_cross_fit_chestxray_pneumonia raddino_cross_fit_chestxray_pneumonia
#   src/run_scripts/submit_chain.sh src/configs/translator_reuse/group2 group2_translator_reuse 4:00:00
#
# Run from the project root (same place you'd run any other sbatch command from here).

set -euo pipefail

DIR="${1:?Usage: submit_chain.sh <config_dir> <results_prefix> [time_limit]}"
PREFIX="${2:?Usage: submit_chain.sh <config_dir> <results_prefix> [time_limit]}"
TIME_LIMIT="${3:-4:00:00}"

prev_dep=""
for f in "$DIR"/experiments_*.csv; do
    name=$(basename "$f" .csv)
    results="results_${PREFIX}_${name#experiments_}.csv"
    if [ -z "$prev_dep" ]; then
        jobid=$(sbatch --parsable --time="$TIME_LIMIT" \
            --output="logs/${PREFIX}_${name}_%j.out.txt" \
            src/run_scripts/run_pipeline_row_by_row.sh "$f" "$results")
    else
        jobid=$(sbatch --parsable --time="$TIME_LIMIT" --dependency="afterany:$prev_dep" \
            --output="logs/${PREFIX}_${name}_%j.out.txt" \
            src/run_scripts/run_pipeline_row_by_row.sh "$f" "$results")
    fi
    echo "$name -> job $jobid (after ${prev_dep:-none})"
    prev_dep=$jobid
done
