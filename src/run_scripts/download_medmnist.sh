#!/bin/bash
#SBATCH -p gpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=16000
#SBATCH --time=2:00:00
#SBATCH --output=logs/download_medmnist.out.txt

# Build the MedMNIST arrow datasets the medical experiments read from disk.
#
#   sbatch src/run_scripts/download_medmnist.sh
#   sbatch src/run_scripts/download_medmnist.sh pathmnist        # just one
#
# No GPU work, but it writes to the shared project directory so it runs as a job rather than
# on the login node.
#
# The downloader prints each split's class distribution and majority-class rate. Read them:
# the majority rate is the accuracy a constant predictor gets for free, so it is the number a
# trained probe has to beat before any result on that dataset means anything.
#
# It also refuses to truncate multi-label variants. chestmnist has 14 binary labels, and the
# old code silently kept only label 0 -- which is why every chestmnist number in this repo
# sits at its 0.892 majority rate. chestmnist now downloads with all 14 labels intact, but the
# eval path is still CrossEntropy + argmax, so it is NOT runnable end-to-end yet; it needs
# BCEWithLogitsLoss and macro-AUC first.

VARIANTS="${@:-pathmnist pneumoniamnist dermamnist}"

PROJECT_DIR="/cluster/home/mwylie/toast-extensions"
WRITABLE_CACHE="/cluster/home/mwylie/hf_writable"
mkdir -p $WRITABLE_CACHE logs

source $GPU_ENV/bin/activate
pip install -e .

export HF_HOME="/cluster/customapps/biomed/vogtlab/users/mwylie/toast/hf_cache"
export HF_DATASETS_CACHE="$WRITABLE_CACHE/datasets"
export TRANSFORMERS_CACHE="$WRITABLE_CACHE/transformers"
export HF_MODULES_CACHE="$WRITABLE_CACHE/modules"
export PYTHONPATH=$PYTHONPATH:$PROJECT_DIR/src

cd $PROJECT_DIR

echo "Downloading: $VARIANTS"
python claude_out/download_medmnist.py $VARIANTS
STATUS=$?

echo ""
echo "Datasets on disk:"
ls -d /cluster/customapps/biomed/vogtlab/users/mwylie/toast/medmnist_*_clean 2>/dev/null \
    || echo "  none found"

exit $STATUS
