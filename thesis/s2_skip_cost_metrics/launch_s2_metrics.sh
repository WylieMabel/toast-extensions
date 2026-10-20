#!/bin/bash
# Stage 1 of the skip-safety metric pipeline for s2: one output_level_eval.py job per s2 combo.
# Each job evaluates exactly the spans in that combo's accuracies CSV and writes
#   thesis/s2_skip_cost_metrics/analysis/output_level_eval_s2_<alias>_<tag>.csv
# Run from the project root:
#   bash thesis/s2_skip_cost_metrics/launch_s2_metrics.sh
#
# Stages 2-4 (join with accuracies, 1%-gap safe labels, rho/AUC/IR tables + summary heatmap) are
# CPU-only and take seconds; run them once every job has finished (on the cluster or locally after
# syncing analysis/):
#   python metric_experiments/build_group_stats.py --group s2
#
# Resources are estimates, not measured: the hidden states for both 500-image calibration sets
# (test + train-split probe) are held in fp16, about 2 GB for the small models, 5 GB for dinov2-base
# and 10 GB for vit-large, and they briefly double while being concatenated.

SMALL_TIME="00:30:00"
VITLARGE_TIME="01:00:00"
CPUS=4

MEM_SMALL="16G"       # deitsmall, vitsmall
MEM_DINOV2BASE="24G"
MEM_VITLARGE="48G"

MAIL_USER="mabel.wylie@gmail.com"
MAIL_TYPE="END,FAIL"

# sbatch forwards this shell's environment. GROUP must be empty or run_output_level_metrics.sh
# switches to its group mode; every other config var is set explicitly per job below.
unset GROUP MODEL MODEL_ALIAS DATASET STEM SPANS_CSV TRUTH_CSV OUT_DIR EVAL_ONLY USE_RAW_ACCURACY

S2=thesis/s2_skip_cost_metrics
OUT_DIR=$S2/analysis

mkdir -p logs

submit() {   # submit <alias> <dataset> <time> <mem>
    local alias=$1 dataset=$2 time=$3 mem=$4
    local tag="${dataset//-/}"
    sbatch --job-name=s2_metrics_${alias}_${tag} --time=$time --mem=$mem --cpus-per-task=$CPUS \
        --mail-type=$MAIL_TYPE --mail-user=$MAIL_USER --output=logs/s2_metrics_${alias}_${tag}_%j.out.txt \
        --export=ALL,MODEL_ALIAS=$alias,DATASET=$dataset,STEM=s2_$alias,SPANS_CSV=$S2/accuracies/accuracies_results_${alias}_${tag}.csv,OUT_DIR=$OUT_DIR,EVAL_ONLY=1 \
        metric_experiments/run_output_level_metrics.sh
}

for dataset in cifar100 imagenet-1k; do
    submit deitsmall  $dataset $SMALL_TIME    $MEM_SMALL
    submit vitsmall   $dataset $SMALL_TIME    $MEM_SMALL
    submit dinov2base $dataset $SMALL_TIME    $MEM_DINOV2BASE
    submit vitlarge   $dataset $VITLARGE_TIME $MEM_VITLARGE
done
