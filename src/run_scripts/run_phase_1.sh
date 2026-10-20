#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:rtx4090:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12000
#SBATCH --time=2:00:00
#SBATCH --output=logs/phaseone.out.txt

# 1. Path Setup
BASE_DIR="/cluster/customapps/biomed/vogtlab/users/mwylie/toast"
PROJECT_DIR="/cluster/home/mwylie/toast-extensions"
# A writable directory for lock files
WRITABLE_CACHE="/cluster/home/mwylie/hf_writable"
mkdir -p $WRITABLE_CACHE

# 2. Activate the venv
source $GPU_ENV/bin/activate
pip install -e .
# 3. CRITICAL Environment Variable Setup
# Point this to your existing data on customapps
export HF_HOME="/cluster/customapps/biomed/vogtlab/users/mwylie/toast/hf_cache"

# Redirect lock files and temporary downloads to home (writable)
# This prevents the "Read-only file system" error
export HF_DATASETS_CACHE="$WRITABLE_CACHE/datasets"
export TRANSFORMERS_CACHE="$WRITABLE_CACHE/transformers"
export HF_MODULES_CACHE="$WRITABLE_CACHE/modules"

# 4. Lockdown mode
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# 5. Fix Python imports
export PYTHONPATH=$PYTHONPATH:$PROJECT_DIR/src

# 6. Execute
cd $PROJECT_DIR

# IMPORTANT: We pass the local path to the encoder here if 
# you want to avoid the "Model configuration not found" error, 
# OR ensure the MODEL2CONFIGS dict is updated in your python code.
bash src/toast/scripts/encode_vision.sh
