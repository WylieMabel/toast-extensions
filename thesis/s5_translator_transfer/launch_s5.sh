#!/bin/bash
# Submits the whole s5 translator-transfer sweep as 10 lanes, one GPU each, all starting straight
# away; everything is estimated done in ~46h. Jobs in a lane run one after another (afterany, so a
# failed or timed-out job does not block the rest of its lane). Run from the project root:
#   bash thesis/s5_translator_transfer/launch_s5.sh
# Configs come from make_configs.py. Once everything has finished, merge the shards into one CSV
# per eval x donor combo:
#   python thesis/s5_translator_transfer/analysis/collect_accuracies.py
# To rerun a single failed job, copy its sbatch line from the lane below.
#
# Donors: the other dataset of the group (in-domain), and CIFAR-100 (out-of-domain; one CIFAR-100
# fitting row reused on both datasets of the group).
# Every row reports two heads (see make_configs.py): setup 1 retrains the head on the row's own
# embeddings (accuracy/auroc/auprc); setup 2 scores the head trained on the unskipped baseline,
# frozen (frozen_head_*). The baseline rows at the top of every config save those heads.
#
# 2026-10-16 relaunch: chest moved to chestxray14_multilabel (was binary chestxray14), every
# config gained baseline rows for both datasets, and the CIFAR-100 donor was added (it used to
# come from the older translator_reuse_cifar100.csv). First run's logs: logs/s5_old/.
#
# Defaults inherited from run_pipeline_row_by_row.sh (flags below override them per job):
#   partition gpu, 1x rtx4090, 2 CPUs, 96000 MB memory, 12:00:00, seeds 0 1 2, 500 samples
#
# Estimates from the first s5 run (binary chest, no baseline rows; 3 seeds per row):
#   rad-dino 11.9h, MAIRA 11.9h, dinov2-base 4.3h for 22 rows; pathology part1 23.5-24.4h for 24.
# Scaled to the relaunch: cx14ml rows ~9% slower than binary (measured on dinov2-base in s4) and
# 2 more rows per config. CIFAR-100 fitting rows from the smoke tests (thesis/smoke_timing/logs):
# vit-large 15.8 min, dinov2-base 9.3 min; phikon-v2 taken as vit-large (same ViT-L/16 at 224),
# rad-dino / MAIRA as ~29 min (dinov2-base x 3.1, their chestxray14 ratio; not measured).
# Estimated job lengths:
#   in-domain   chest rad-dino 13.8h, MAIRA 14.1h, dinov2-base 4.8h
#               pathology part1 25.5-26.7h, part2 23.5-24.6h
#   CIFAR-100   chest rad-dino 19.1h, MAIRA 19.4h, dinov2-base 6.5h
#               pathology part1/2 20.1-20.4h, part3 17.8-18.1h

set -e   # stop if any submission fails, so no job is left waiting on an empty dependency

# Time limits and resources are per model, the same for the in-domain and CIFAR-100 jobs.
SHORT_TIME="12:00:00"    # dinov2-base, estimated 4.8-6.5h
MEDIUM_TIME="24:00:00"   # rad-dino / MAIRA, estimated 13.8-14.1h (CIFAR-100 19.1-19.4h)
LONG_TIME="36:00:00"     # pathology, estimated 23.5-26.7h (CIFAR-100 thirds 17.8-20.4h)
# CPUs per model: the most it needed on either of its datasets in the s4 smoke tests.
# run_pipeline_row_by_row.sh sets NUM_WORKERS to CPUs - 1.
#                                     cores used
CPUS_RADDINO=6                      # 3.4
CPUS_RADDINOMAIRA2=6                # 3.5
CPUS_DINOV2BASE=8                   # 5.6
CPUS_GROUP2=4                       # ~2
# --mem per model, from how the s4 jobs actually fared: sacct MaxRSS misses short spikes, so the
# smoke-test peaks understate what a job needs. Every s5 job runs on both of its group's datasets.
MEM_RADDINO="64G"          # chestxray14 OOM'd on every row at 48G before the memfix (s4 9728291); cx14ml runs at 48G
MEM_RADDINOMAIRA2="64G"    # same as rad-dino (s4 9728292)
MEM_DINOV2BASE="28G"       # chestxray14 ran 39 rows at 28G with no OOM (s4 9728298); cx14ml runs at 28G
MEM_VITLARGE="60G"         # tcga_ut OOM'd on ~40% of seeds at 36G (s4 9728304)
MEM_PHIKONV2="60G"         # tcga_ut ran 24 rows at 60G, patchcamelyon at 48G, no OOM (s4 9728311, 9728301)

MAIL_USER="mabel.wylie@gmail.com"
MAIL_TYPE="BEGIN,END"

# Embeddings go to the node's local /scratch (home quota is only 21 GB). --tmp keeps jobs off
# nodes without local disk (gpu-biomed-02, -04).
LOCAL_DISK="20G"
export LOCAL_EMBEDDINGS=1

# sbatch forwards this shell's environment; stale values would override the per-config
# results name and the section results/ folder.
unset RESULTS_CSV_NAME RESULTS_DIR ACCURACIES_DIR CONFIG_CSV EMBEDDINGS_DIR SEEDS NUM_WORKERS HEADS_DIR

CONFIGS=thesis/s5_translator_transfer/configs
COMMON="--parsable --tmp=$LOCAL_DISK --mail-type=$MAIL_TYPE --mail-user=$MAIL_USER"

mkdir -p logs

# lane NAME "STEM|TIME|CPUS|MEM" ...: submits the configs one after another, each waiting for the
# previous one, and prints the job IDs. Job names are s5_<config stem without "experiments_">.
lane() {
    local name=$1; shift
    local prev="" ids="" spec stem time cpus mem dep
    for spec in "$@"; do
        IFS='|' read -r stem time cpus mem <<< "$spec"
        dep=""
        [ -n "$prev" ] && dep="--dependency=afterany:$prev"
        prev=$(sbatch $COMMON --job-name="s5_$stem" --time="$time" --cpus-per-task="$cpus" \
            --mem="$mem" $dep --output="logs/s5_${stem}_%j.out.txt" \
            src/run_scripts/run_pipeline_row_by_row.sh "$CONFIGS/experiments_${stem}.csv")
        ids="$ids${ids:+ -> }$prev"
    done
    echo "$name: $ids"
}

# Config stems
CX=group1_crossfit_chestxray14multilabel_from_chestxraypneumonia      # chest, in-domain
PN=group1_crossfit_chestxraypneumonia_from_chestxray14multilabel
CC=group1_crossfit_medical_from_cifar100                              # chest, CIFAR-100
TP=group2_crossfit_tcgaut_from_patchcamelyon                          # pathology, in-domain
PT=group2_crossfit_patchcamelyon_from_tcgaut
PC=group2_crossfit_medical_from_cifar100                              # pathology, CIFAR-100

VL="$CPUS_GROUP2|$MEM_VITLARGE"
PK="$CPUS_GROUP2|$MEM_PHIKONV2"
RD="$CPUS_RADDINO|$MEM_RADDINO"
MA="$CPUS_RADDINOMAIRA2|$MEM_RADDINOMAIRA2"
DB="$CPUS_DINOV2BASE|$MEM_DINOV2BASE"

#                                                                        estimated lane length
lane "lane 1"  "${TP}_vitlarge_part1|$LONG_TIME|$VL"  "${PC}_vitlarge_part3|$LONG_TIME|$VL"     # 26.5 + 17.8 = 44.3h
lane "lane 2"  "${TP}_phikonv2_part1|$LONG_TIME|$PK"  "${PC}_phikonv2_part3|$LONG_TIME|$PK"     # 26.7 + 18.1 = 44.8h
lane "lane 3"  "${PT}_vitlarge_part1|$LONG_TIME|$VL"  "${CC}_raddino|$MEDIUM_TIME|$RD"            # 26.5 + 19.1 = 45.6h
lane "lane 4"  "${PT}_phikonv2_part1|$LONG_TIME|$PK"  "${CC}_raddinomaira2|$MEDIUM_TIME|$MA"      # 26.7 + 19.4 = 46.1h
lane "lane 5"  "${TP}_vitlarge_part2|$LONG_TIME|$VL"  "${PC}_vitlarge_part1|$LONG_TIME|$VL"     # 24.5 + 20.1 = 44.6h
lane "lane 6"  "${TP}_phikonv2_part2|$LONG_TIME|$PK"  "${PC}_phikonv2_part1|$LONG_TIME|$PK"     # 24.6 + 20.4 = 45.0h
lane "lane 7"  "${PT}_vitlarge_part2|$LONG_TIME|$VL"  "${PC}_vitlarge_part2|$LONG_TIME|$VL"     # 24.5 + 20.1 = 44.6h
lane "lane 8"  "${PT}_phikonv2_part2|$LONG_TIME|$PK"  "${PC}_phikonv2_part2|$LONG_TIME|$PK"     # 24.6 + 20.4 = 45.0h
lane "lane 9"  "${CX}_raddino|$MEDIUM_TIME|$RD" "${CX}_raddinomaira2|$MEDIUM_TIME|$MA" \
               "${CX}_dinov2base|$SHORT_TIME|$DB" "${CC}_dinov2base|$SHORT_TIME|$DB"            # 32.7 + 6.5 = 39.2h
lane "lane 10" "${PN}_raddino|$MEDIUM_TIME|$RD" "${PN}_raddinomaira2|$MEDIUM_TIME|$MA" \
               "${PN}_dinov2base|$SHORT_TIME|$DB"                                               # 32.7h
