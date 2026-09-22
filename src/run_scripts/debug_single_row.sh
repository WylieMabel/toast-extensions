#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:rtx4090:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=128000
#SBATCH --time=03:00:00
#SBATCH --output=logs/debug_single_row_%j.out.txt
#
# Reproduces exactly ONE (dataset, encoder, skip, seed) row outside the big sweep, with:
#   - generous --mem (128G vs the sweep's 48G) so a run that succeeds here but failed in the
#     sweep tells you the 48G cap really was the problem; a run that still dies here tells you
#     it is NOT simply insufficient --mem, and you need the memlog trace below instead.
#   - TOAST_MEMLOG=1, which turns on the [memlog] host-RSS/GPU-mem prints added to
#     extract_representations() (src/toast/utils/utils.py) and the model/dataset/loader-build
#     steps (src/toast/utils/encode_vision_full.py). Look for the last [memlog] line before a
#     "Killed" to see exactly which phase the OOM-killer caught it in.
#   - /usr/bin/time -v wrapping the python call, so you get Maximum resident set size even on
#     a SIGKILL (the process's own prints stop, but /usr/bin/time is the parent and still
#     reports on the child it wrapped).
#   - the full log always kept (unlike run_pipeline_row_by_row.sh, which deletes it on
#     success) -- this script is for reading afterward, not for driving a long unattended sweep.
#
# Usage (all via env, matching src/configs/health/*/experiments*.csv column names):
#   DATASET=tcga_ut ENCODER="owkin/phikon-v2" SKIP="[(5, 9)]" SEED=0 \
#     sbatch src/run_scripts/debug_single_row.sh
#
# Defaults below reproduce the first row from the pasted failure log (tcga_ut / phikon-v2 /
# [(5, 9)] / seed 0) -- override to point at any other still-missing config from
# src/configs/health/rerun_missing/.
#
# Phase 2 (training) is skipped by default -- phase 1 (encode) is what's been failing. Set
# RUN_PHASE2=1 to also run it (only reached if phase 1 succeeds).

set -u

DATASET="${DATASET:-tcga_ut}"
ENCODER="${ENCODER:-owkin/phikon-v2}"
SKIP="${SKIP:-[(5, 9)]}"
SEED="${SEED:-0}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-1}"
RUN_PHASE2="${RUN_PHASE2:-0}"
SAMPLES=500

# 1. Path setup
PROJECT_DIR="/cluster/home/mwylie/toast-extensions"
WRITABLE_CACHE="/cluster/home/mwylie/hf_writable"
mkdir -p "$WRITABLE_CACHE" logs

# 2. Activate venv
source "$GPU_ENV/bin/activate"
pip install -e .

# 3. Environment variables
export HF_HOME="/cluster/customapps/biomed/vogtlab/users/mwylie/toast/hf_cache"
export HF_DATASETS_CACHE="$WRITABLE_CACHE/datasets"
export TRANSFORMERS_CACHE="$WRITABLE_CACHE/transformers"
export HF_MODULES_CACHE="$WRITABLE_CACHE/modules"
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH="${PYTHONPATH:-}:$PROJECT_DIR/src"
export PYTHONUNBUFFERED=1

# This run's whole point is to see memory instrumentation, so it stays on regardless of what
# the sweep normally sets.
export TOAST_MEMLOG=1

cd "$PROJECT_DIR"

# 4. Build the one-row config CSV. Same column order make_health_skip_grid.py writes.
TEMP_CSV=$(mktemp /tmp/debug_row_XXXXXX.csv)
cat > "$TEMP_CSV" <<CSV
"dataset","encoder","skip","mlp_skip","attn_skip","head_dict","skip_translator","mlp_mode","attn_mode","fit_dataset"
"$DATASET","$ENCODER","$SKIP","[]","[]","{}","linear","identity","identity",""
CSV

echo "=================================================="
echo " Debug row"
echo "=================================================="
cat "$TEMP_CSV"
echo "SEED=$SEED BATCH_SIZE=$BATCH_SIZE NUM_WORKERS=$NUM_WORKERS  (--mem=128G, sweep uses 48G)"
echo ""

# Delete any leftover embeddings for this exact config so phase 1 actually re-encodes rather
# than skipping (encode_vision_full.py skips outright if the embedding dir already has a
# dataset_dict.json).
EMB_DIR=$(python src/toast/scripts/get_embed_dir.py "$TEMP_CSV" "$SAMPLES" 2>/dev/null)
if [ -n "$EMB_DIR" ]; then
    [ -d "$EMB_DIR" ] && rm -rf "$EMB_DIR"
    [ -d "${EMB_DIR}_temp" ] && rm -rf "${EMB_DIR}_temp"
fi

echo "GPU memory before:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
echo "Host memory before:"
free -h || true
echo ""

TIME_CMD=""
command -v /usr/bin/time >/dev/null 2>&1 && TIME_CMD="/usr/bin/time -v"

echo "--- phase 1: encode ---"
$TIME_CMD python -u src/toast/utils/encode_vision_full.py \
    --translator_name=identity \
    --seed="$SEED" \
    --config_csv="$TEMP_CSV" \
    --batch_size="$BATCH_SIZE" \
    --num_workers="$NUM_WORKERS" \
    --samples_to_extract="$SAMPLES"
PHASE1_STATUS=$?

echo ""
echo "GPU memory after phase 1:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
echo "Host memory after phase 1:"
free -h || true
echo "phase 1 exit status: $PHASE1_STATUS"

if [ "$PHASE1_STATUS" -ne 0 ]; then
    echo ""
    echo "=================================================="
    echo " Phase 1 failed (status $PHASE1_STATUS). Check the last [memlog] line above and"
    echo " 'Maximum resident set size' from /usr/bin/time -v. Leaving $TEMP_CSV in place for"
    echo " inspection (not deleted, unlike the normal sweep loop)."
    echo "=================================================="
    exit "$PHASE1_STATUS"
fi

if [ "$RUN_PHASE2" = "1" ]; then
    echo ""
    echo "--- phase 2: train ---"
    SEEDS="$SEED" CONFIG_CSV="$TEMP_CSV" $TIME_CMD bash src/toast/scripts/train_skipped_full.sh
    PHASE2_STATUS=$?
    echo "phase 2 exit status: $PHASE2_STATUS"
fi

# Clean up this debug row's embeddings so it doesn't eat disk quota.
[ -n "$EMB_DIR" ] && [ -d "$EMB_DIR" ] && rm -rf "$EMB_DIR"
rm -f "$TEMP_CSV"

echo ""
echo "Done."
