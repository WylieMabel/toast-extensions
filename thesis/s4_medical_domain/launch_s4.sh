#!/bin/bash
# Submits the s4 sweep as 6 groups. Jobs in a group run one after another (each waits for the
# previous one with afterany, so a failed or timed-out job does not block the rest of its group);
# the 6 groups run in parallel, so s4 never holds more than 6 GPUs. Run from the project root:
#   bash thesis/s4_medical_domain/launch_s4.sh
#
# Defaults inherited from run_pipeline_row_by_row.sh (flags below override them per job):
#   partition gpu, 1x rtx4090, 2 CPUs, 96000 MB memory, 12:00:00, seeds 0 1 2, 500 samples
#
# Estimated group lengths (smoke-test timings): 1 70.6h, 2 71.5h, 3 71.3h, 4 74.2h, 5 70.4h, 6 74.2h

set -e   # stop if any submission fails, so no job is left waiting on an empty dependency

# Every job is estimated to finish at least 5h before its limit.
SHORT_TIME="12:00:00"    # pneumonia jobs, estimated 2.4-5.3h
MEDIUM_TIME="24:00:00"   # estimated 10.5-19.0h
LONG_TIME="36:00:00"     # estimated 19.1-27.0h (all patchcamelyon, tcga_ut len1, chestxray14 len12)
# CPUs and --mem per model x dataset, from the smoke tests (sacct TotalCPU/Elapsed and MaxRSS).
# run_pipeline_row_by_row.sh sets NUM_WORKERS to CPUs - 1. --mem is ~1.5x the measured peak.
#                                     cores used   peak mem
CPUS_DEFAULT=4                      # ~2 cores for pneumonia, tcga_ut, patchcamelyon
CPUS_RADDINO_CHESTXRAY14=6          # 3.4
CPUS_RADDINOMAIRA2_CHESTXRAY14=6    # 3.5
CPUS_DINOV2BASE_CHESTXRAY14=8       # 5.5 (CPU-bound image loading)

MEM_RADDINO_CHESTXRAY14="48G"           # 33.0G
MEM_RADDINOMAIRA2_CHESTXRAY14="48G"     # 31.4G
MEM_DINOV2BASE_CHESTXRAY14="28G"        # 15.6G
MEM_RADDINO_CHESTXRAYPNEUMONIA="54G"    # 35.9G
MEM_RADDINOMAIRA2_CHESTXRAYPNEUMONIA="72G"  # 45.1G
MEM_DINOV2BASE_CHESTXRAYPNEUMONIA="20G"  # 9.7G
MEM_VITLARGE_TCGAUT="36G"               # 21.9G
MEM_PHIKONV2_TCGAUT="60G"               # 39.8G
MEM_VITLARGE_PATCHCAMELYON="48G"        # 30.2G
MEM_PHIKONV2_PATCHCAMELYON="48G"        # 31.5G

MAIL_USER="mabel.wylie@gmail.com"
MAIL_TYPE="BEGIN,END"

# Embeddings go to the node's local /scratch (home quota is only 21 GB). --tmp keeps jobs off
# nodes without local disk (gpu-biomed-02, -04).
LOCAL_DISK="20G"
export LOCAL_EMBEDDINGS=1

# sbatch forwards this shell's environment; stale values would override the per-config
# results name and the section results/ folder.
unset RESULTS_CSV_NAME RESULTS_DIR ACCURACIES_DIR CONFIG_CSV EMBEDDINGS_DIR SEEDS NUM_WORKERS

CONFIGS=thesis/s4_medical_domain/configs
COMMON="--parsable --tmp=$LOCAL_DISK --mail-type=$MAIL_TYPE --mail-user=$MAIL_USER"

mkdir -p logs

# Each group starts only when one of the running s2/s3 ViT-L jobs ends (afterany), so s4 takes
# over those GPUs instead of adding new ones. Heavier groups follow the jobs expected to end
# first (estimated remaining as of 2026-10-11 evening -> estimated group end):
AFTER_G1=9726496   # s2_vitlarge_imagenet1k        ~13.5h left + 70.6h
AFTER_G2=9726718   # s3_vitlarge_imagenet1k_part1  ~7.7h left  + 71.5h
AFTER_G3=9726719   # s3_vitlarge_imagenet1k_part2  ~7.7h left  + 71.3h
AFTER_G4=9726720   # s3_vitlarge_imagenet1k_part3  ~7.7h left  + 74.2h
AFTER_G5=9726492   # s2_vitlarge_cifar100          ~15.7h left + 70.4h
AFTER_G6=9726721   # s3_vitlarge_imagenet1k_part4  ~7.7h left  + 74.2h

# --- group 1 ---
G1_1=$(sbatch $COMMON --job-name=s4_g1_raddino_chestxray14_len12 --time=$LONG_TIME --cpus-per-task=$CPUS_RADDINO_CHESTXRAY14 --mem=$MEM_RADDINO_CHESTXRAY14 --dependency=afterany:$AFTER_G1 --output=logs/s4_g1_raddino_chestxray14_len12_%j.out.txt \
    src/run_scripts/run_pipeline_row_by_row.sh $CONFIGS/experiments_raddino_chestxray14_len12.csv)
echo "group 1: raddino_chestxray14_len12 -> $G1_1"
G1_2=$(sbatch $COMMON --job-name=s4_g1_raddinomaira2_chestxray14_len12 --time=$LONG_TIME --cpus-per-task=$CPUS_RADDINOMAIRA2_CHESTXRAY14 --mem=$MEM_RADDINOMAIRA2_CHESTXRAY14 --dependency=afterany:$G1_1 --output=logs/s4_g1_raddinomaira2_chestxray14_len12_%j.out.txt \
    src/run_scripts/run_pipeline_row_by_row.sh $CONFIGS/experiments_raddinomaira2_chestxray14_len12.csv)
echo "group 1: raddinomaira2_chestxray14_len12 -> $G1_2"
G1_3=$(sbatch $COMMON --job-name=s4_g1_raddinomaira2_chestxraypneumonia --time=$SHORT_TIME --cpus-per-task=$CPUS_DEFAULT --mem=$MEM_RADDINOMAIRA2_CHESTXRAYPNEUMONIA --dependency=afterany:$G1_2 --output=logs/s4_g1_raddinomaira2_chestxraypneumonia_%j.out.txt \
    src/run_scripts/run_pipeline_row_by_row.sh $CONFIGS/experiments_raddinomaira2_chestxraypneumonia.csv)
echo "group 1: raddinomaira2_chestxraypneumonia -> $G1_3"
G1_4=$(sbatch $COMMON --job-name=s4_g1_dinov2base_chestxraypneumonia --time=$SHORT_TIME --cpus-per-task=$CPUS_DEFAULT --mem=$MEM_DINOV2BASE_CHESTXRAYPNEUMONIA --dependency=afterany:$G1_3 --output=logs/s4_g1_dinov2base_chestxraypneumonia_%j.out.txt \
    src/run_scripts/run_pipeline_row_by_row.sh $CONFIGS/experiments_dinov2base_chestxraypneumonia.csv)
echo "group 1: dinov2base_chestxraypneumonia -> $G1_4"
G1_5=$(sbatch $COMMON --job-name=s4_g1_phikonv2_patchcamelyon_len2 --time=$LONG_TIME --cpus-per-task=$CPUS_DEFAULT --mem=$MEM_PHIKONV2_PATCHCAMELYON --dependency=afterany:$G1_4 --output=logs/s4_g1_phikonv2_patchcamelyon_len2_%j.out.txt \
    src/run_scripts/run_pipeline_row_by_row.sh $CONFIGS/experiments_phikonv2_patchcamelyon_len2.csv)
echo "group 1: phikonv2_patchcamelyon_len2 -> $G1_5"

# --- group 2 ---
G2_1=$(sbatch $COMMON --job-name=s4_g2_raddino_chestxray14_len34 --time=$MEDIUM_TIME --cpus-per-task=$CPUS_RADDINO_CHESTXRAY14 --mem=$MEM_RADDINO_CHESTXRAY14 --dependency=afterany:$AFTER_G2 --output=logs/s4_g2_raddino_chestxray14_len34_%j.out.txt \
    src/run_scripts/run_pipeline_row_by_row.sh $CONFIGS/experiments_raddino_chestxray14_len34.csv)
echo "group 2: raddino_chestxray14_len34 -> $G2_1"
G2_2=$(sbatch $COMMON --job-name=s4_g2_raddinomaira2_chestxray14_len34 --time=$MEDIUM_TIME --cpus-per-task=$CPUS_RADDINOMAIRA2_CHESTXRAY14 --mem=$MEM_RADDINOMAIRA2_CHESTXRAY14 --dependency=afterany:$G2_1 --output=logs/s4_g2_raddinomaira2_chestxray14_len34_%j.out.txt \
    src/run_scripts/run_pipeline_row_by_row.sh $CONFIGS/experiments_raddinomaira2_chestxray14_len34.csv)
echo "group 2: raddinomaira2_chestxray14_len34 -> $G2_2"
G2_3=$(sbatch $COMMON --job-name=s4_g2_dinov2base_chestxray14 --time=$MEDIUM_TIME --cpus-per-task=$CPUS_DINOV2BASE_CHESTXRAY14 --mem=$MEM_DINOV2BASE_CHESTXRAY14 --dependency=afterany:$G2_2 --output=logs/s4_g2_dinov2base_chestxray14_%j.out.txt \
    src/run_scripts/run_pipeline_row_by_row.sh $CONFIGS/experiments_dinov2base_chestxray14.csv)
echo "group 2: dinov2base_chestxray14 -> $G2_3"
G2_4=$(sbatch $COMMON --job-name=s4_g2_raddino_chestxraypneumonia --time=$SHORT_TIME --cpus-per-task=$CPUS_DEFAULT --mem=$MEM_RADDINO_CHESTXRAYPNEUMONIA --dependency=afterany:$G2_3 --output=logs/s4_g2_raddino_chestxraypneumonia_%j.out.txt \
    src/run_scripts/run_pipeline_row_by_row.sh $CONFIGS/experiments_raddino_chestxraypneumonia.csv)
echo "group 2: raddino_chestxraypneumonia -> $G2_4"
G2_5=$(sbatch $COMMON --job-name=s4_g2_vitlarge_patchcamelyon_len1 --time=$LONG_TIME --cpus-per-task=$CPUS_DEFAULT --mem=$MEM_VITLARGE_PATCHCAMELYON --dependency=afterany:$G2_4 --output=logs/s4_g2_vitlarge_patchcamelyon_len1_%j.out.txt \
    src/run_scripts/run_pipeline_row_by_row.sh $CONFIGS/experiments_vitlarge_patchcamelyon_len1.csv)
echo "group 2: vitlarge_patchcamelyon_len1 -> $G2_5"

# --- group 3 ---
G3_1=$(sbatch $COMMON --job-name=s4_g3_phikonv2_patchcamelyon_len1 --time=$LONG_TIME --cpus-per-task=$CPUS_DEFAULT --mem=$MEM_PHIKONV2_PATCHCAMELYON --dependency=afterany:$AFTER_G3 --output=logs/s4_g3_phikonv2_patchcamelyon_len1_%j.out.txt \
    src/run_scripts/run_pipeline_row_by_row.sh $CONFIGS/experiments_phikonv2_patchcamelyon_len1.csv)
echo "group 3: phikonv2_patchcamelyon_len1 -> $G3_1"
G3_2=$(sbatch $COMMON --job-name=s4_g3_phikonv2_patchcamelyon_len4 --time=$LONG_TIME --cpus-per-task=$CPUS_DEFAULT --mem=$MEM_PHIKONV2_PATCHCAMELYON --dependency=afterany:$G3_1 --output=logs/s4_g3_phikonv2_patchcamelyon_len4_%j.out.txt \
    src/run_scripts/run_pipeline_row_by_row.sh $CONFIGS/experiments_phikonv2_patchcamelyon_len4.csv)
echo "group 3: phikonv2_patchcamelyon_len4 -> $G3_2"
G3_3=$(sbatch $COMMON --job-name=s4_g3_vitlarge_patchcamelyon_len4 --time=$LONG_TIME --cpus-per-task=$CPUS_DEFAULT --mem=$MEM_VITLARGE_PATCHCAMELYON --dependency=afterany:$G3_2 --output=logs/s4_g3_vitlarge_patchcamelyon_len4_%j.out.txt \
    src/run_scripts/run_pipeline_row_by_row.sh $CONFIGS/experiments_vitlarge_patchcamelyon_len4.csv)
echo "group 3: vitlarge_patchcamelyon_len4 -> $G3_3"

# --- group 4 ---
G4_1=$(sbatch $COMMON --job-name=s4_g4_vitlarge_tcgaut_len1 --time=$LONG_TIME --cpus-per-task=$CPUS_DEFAULT --mem=$MEM_VITLARGE_TCGAUT --dependency=afterany:$AFTER_G4 --output=logs/s4_g4_vitlarge_tcgaut_len1_%j.out.txt \
    src/run_scripts/run_pipeline_row_by_row.sh $CONFIGS/experiments_vitlarge_tcgaut_len1.csv)
echo "group 4: vitlarge_tcgaut_len1 -> $G4_1"
G4_2=$(sbatch $COMMON --job-name=s4_g4_phikonv2_tcgaut_len2 --time=$MEDIUM_TIME --cpus-per-task=$CPUS_DEFAULT --mem=$MEM_PHIKONV2_TCGAUT --dependency=afterany:$G4_1 --output=logs/s4_g4_phikonv2_tcgaut_len2_%j.out.txt \
    src/run_scripts/run_pipeline_row_by_row.sh $CONFIGS/experiments_phikonv2_tcgaut_len2.csv)
echo "group 4: phikonv2_tcgaut_len2 -> $G4_2"
G4_3=$(sbatch $COMMON --job-name=s4_g4_phikonv2_tcgaut_len3 --time=$MEDIUM_TIME --cpus-per-task=$CPUS_DEFAULT --mem=$MEM_PHIKONV2_TCGAUT --dependency=afterany:$G4_2 --output=logs/s4_g4_phikonv2_tcgaut_len3_%j.out.txt \
    src/run_scripts/run_pipeline_row_by_row.sh $CONFIGS/experiments_phikonv2_tcgaut_len3.csv)
echo "group 4: phikonv2_tcgaut_len3 -> $G4_3"
G4_4=$(sbatch $COMMON --job-name=s4_g4_vitlarge_tcgaut_len4 --time=$MEDIUM_TIME --cpus-per-task=$CPUS_DEFAULT --mem=$MEM_VITLARGE_TCGAUT --dependency=afterany:$G4_3 --output=logs/s4_g4_vitlarge_tcgaut_len4_%j.out.txt \
    src/run_scripts/run_pipeline_row_by_row.sh $CONFIGS/experiments_vitlarge_tcgaut_len4.csv)
echo "group 4: vitlarge_tcgaut_len4 -> $G4_4"

# --- group 5 ---
G5_1=$(sbatch $COMMON --job-name=s4_g5_vitlarge_patchcamelyon_len2 --time=$LONG_TIME --cpus-per-task=$CPUS_DEFAULT --mem=$MEM_VITLARGE_PATCHCAMELYON --dependency=afterany:$AFTER_G5 --output=logs/s4_g5_vitlarge_patchcamelyon_len2_%j.out.txt \
    src/run_scripts/run_pipeline_row_by_row.sh $CONFIGS/experiments_vitlarge_patchcamelyon_len2.csv)
echo "group 5: vitlarge_patchcamelyon_len2 -> $G5_1"
G5_2=$(sbatch $COMMON --job-name=s4_g5_phikonv2_patchcamelyon_len3 --time=$LONG_TIME --cpus-per-task=$CPUS_DEFAULT --mem=$MEM_PHIKONV2_PATCHCAMELYON --dependency=afterany:$G5_1 --output=logs/s4_g5_phikonv2_patchcamelyon_len3_%j.out.txt \
    src/run_scripts/run_pipeline_row_by_row.sh $CONFIGS/experiments_phikonv2_patchcamelyon_len3.csv)
echo "group 5: phikonv2_patchcamelyon_len3 -> $G5_2"
G5_3=$(sbatch $COMMON --job-name=s4_g5_vitlarge_patchcamelyon_len3 --time=$LONG_TIME --cpus-per-task=$CPUS_DEFAULT --mem=$MEM_VITLARGE_PATCHCAMELYON --dependency=afterany:$G5_2 --output=logs/s4_g5_vitlarge_patchcamelyon_len3_%j.out.txt \
    src/run_scripts/run_pipeline_row_by_row.sh $CONFIGS/experiments_vitlarge_patchcamelyon_len3.csv)
echo "group 5: vitlarge_patchcamelyon_len3 -> $G5_3"

# --- group 6 ---
G6_1=$(sbatch $COMMON --job-name=s4_g6_phikonv2_tcgaut_len1 --time=$LONG_TIME --cpus-per-task=$CPUS_DEFAULT --mem=$MEM_PHIKONV2_TCGAUT --dependency=afterany:$AFTER_G6 --output=logs/s4_g6_phikonv2_tcgaut_len1_%j.out.txt \
    src/run_scripts/run_pipeline_row_by_row.sh $CONFIGS/experiments_phikonv2_tcgaut_len1.csv)
echo "group 6: phikonv2_tcgaut_len1 -> $G6_1"
G6_2=$(sbatch $COMMON --job-name=s4_g6_vitlarge_tcgaut_len2 --time=$MEDIUM_TIME --cpus-per-task=$CPUS_DEFAULT --mem=$MEM_VITLARGE_TCGAUT --dependency=afterany:$G6_1 --output=logs/s4_g6_vitlarge_tcgaut_len2_%j.out.txt \
    src/run_scripts/run_pipeline_row_by_row.sh $CONFIGS/experiments_vitlarge_tcgaut_len2.csv)
echo "group 6: vitlarge_tcgaut_len2 -> $G6_2"
G6_3=$(sbatch $COMMON --job-name=s4_g6_vitlarge_tcgaut_len3 --time=$MEDIUM_TIME --cpus-per-task=$CPUS_DEFAULT --mem=$MEM_VITLARGE_TCGAUT --dependency=afterany:$G6_2 --output=logs/s4_g6_vitlarge_tcgaut_len3_%j.out.txt \
    src/run_scripts/run_pipeline_row_by_row.sh $CONFIGS/experiments_vitlarge_tcgaut_len3.csv)
echo "group 6: vitlarge_tcgaut_len3 -> $G6_3"
G6_4=$(sbatch $COMMON --job-name=s4_g6_phikonv2_tcgaut_len4 --time=$MEDIUM_TIME --cpus-per-task=$CPUS_DEFAULT --mem=$MEM_PHIKONV2_TCGAUT --dependency=afterany:$G6_3 --output=logs/s4_g6_phikonv2_tcgaut_len4_%j.out.txt \
    src/run_scripts/run_pipeline_row_by_row.sh $CONFIGS/experiments_phikonv2_tcgaut_len4.csv)
echo "group 6: phikonv2_tcgaut_len4 -> $G6_4"
