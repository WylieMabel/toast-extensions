#!/bin/bash
#SBATCH -p gpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=16000
#SBATCH --time=00:30:00
#SBATCH --output=logs/class_balance_report.out.txt

# Report per-split class balance and majority-class rate for every dataset in
# DATASET2LOCAL_PATH (or a chosen subset), reading straight from the local on-disk copies.
#
#   sbatch src/run_scripts/run_class_balance_report.sh
#   sbatch src/run_scripts/run_class_balance_report.sh chestxray14 chestxray_pneumonia patchcamelyon tcga_ut

PROJECT_DIR="/cluster/home/mwylie/toast-extensions"
WRITABLE_CACHE="/cluster/home/mwylie/hf_writable"
mkdir -p $WRITABLE_CACHE logs

source $GPU_ENV/bin/activate
pip install -e .

export HF_HOME="/cluster/customapps/biomed/vogtlab/users/mwylie/toast/hf_cache"
export HF_DATASETS_CACHE="$WRITABLE_CACHE/datasets"
export TRANSFORMERS_CACHE="$WRITABLE_CACHE/transformers"
export HF_MODULES_CACHE="$WRITABLE_CACHE/modules"
export HF_DATASETS_OFFLINE=1
export PYTHONPATH=$PYTHONPATH:$PROJECT_DIR/src

cd $PROJECT_DIR

python src/toast/scripts/report_class_balance.py "$@"
