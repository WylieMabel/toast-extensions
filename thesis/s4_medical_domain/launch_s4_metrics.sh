#!/bin/bash
# Stage 1 of the skip-safety metric pipeline for s4: one output_level_eval.py job per s4 combo.
# Each job evaluates exactly the spans in that combo's combined accuracies CSV and writes
#   thesis/s4_medical_domain/analysis/output_level_eval_s4_<alias>_<tag>.csv
# Build the combined CSVs first (it exits 1 and lists the gaps if any combo is incomplete), then
# run from the project root:
#   python thesis/s4_medical_domain/combine_s4.py
#   bash thesis/s4_medical_domain/launch_s4_metrics.sh
# Pass dataset tags to submit only those combos, e.g. just the multilabel run:
#   bash thesis/s4_medical_domain/launch_s4_metrics.sh chestxray14multilabel
#
# Stages 2-4 are CPU-only and take seconds; run them once every job has finished:
#   python metric_experiments/build_group_stats.py --group s4
#
# Resources are estimates, not measured. Every token's hidden state is kept for both 500-image
# calibration sets (test + train-split probe) in fp16, and they briefly double while being
# concatenated. rad-dino / MAIRA run at 518px (1370 tokens): ~14 GB per set, so up to ~55 GB at
# the peak. dinov2-base runs at 224px (~5 GB), vit-large / phikon-v2 at 224px (~10 GB).

SHORT_TIME="00:30:00"
LONG_TIME="01:00:00"
RADDINO_TIME="02:00:00"
CPUS=4

MEM_DINOV2BASE="24G"
MEM_RADDINO="80G"     # rad-dino and rad-dino-maira-2
MEM_VITL="48G"        # vit-large and phikon-v2

MAIL_USER="mabel.wylie@gmail.com"
MAIL_TYPE="END,FAIL"

# sbatch forwards this shell's environment. GROUP must be empty or run_output_level_metrics.sh
# switches to its group mode; every other config var is set explicitly per job below.
unset GROUP MODEL MODEL_ALIAS DATASET DATASET_TAG STEM SPANS_CSV TRUTH_CSV OUT_DIR EVAL_ONLY USE_RAW_ACCURACY

S4=thesis/s4_medical_domain
OUT_DIR=$S4/analysis

mkdir -p logs

ONLY_TAGS=" $* "   # empty = every combo

submit() {   # submit <alias> <dataset key> <file tag> <time> <mem>
    local alias=$1 dataset=$2 tag=$3 time=$4 mem=$5
    if [ -n "${ONLY_TAGS// /}" ] && [[ "$ONLY_TAGS" != *" $tag "* ]]; then
        return
    fi
    local spans=$S4/accuracies/combined/accuracies_results_${alias}_${tag}.csv
    if [ ! -f "$spans" ]; then
        echo "skip ${alias}_${tag}: no $spans (run combine_s4.py)"
        return
    fi
    sbatch --job-name=s4_metrics_${alias}_${tag} --time=$time --mem=$mem --cpus-per-task=$CPUS \
        --mail-type=$MAIL_TYPE --mail-user=$MAIL_USER --output=logs/s4_metrics_${alias}_${tag}_%j.out.txt \
        --export=ALL,MODEL_ALIAS=$alias,DATASET=$dataset,DATASET_TAG=$tag,STEM=s4_$alias,SPANS_CSV=$spans,OUT_DIR=$OUT_DIR,EVAL_ONLY=1 \
        metric_experiments/run_output_level_metrics.sh
}

# Multilabel: same images as chestxray14, so the same resources. The probe metrics are skipped
# for it (output_level_eval.py), so it is a little faster.
for ds in "chestxray14 chestxray14" "chestxray_pneumonia chestxraypneumonia" "chestxray14_multilabel chestxray14multilabel"; do
    set -- $ds
    submit dinov2base    $1 $2 $SHORT_TIME   $MEM_DINOV2BASE
    submit raddino       $1 $2 $RADDINO_TIME $MEM_RADDINO
    submit raddinomaira2 $1 $2 $RADDINO_TIME $MEM_RADDINO
done

for ds in "patchcamelyon patchcamelyon" "tcga_ut tcgaut"; do
    set -- $ds
    submit vitlarge $1 $2 $LONG_TIME $MEM_VITL
    submit phikonv2 $1 $2 $LONG_TIME $MEM_VITL
done
