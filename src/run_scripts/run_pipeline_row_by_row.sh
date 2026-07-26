#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:rtx4090:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=48000
#SBATCH --time=48:00:00
#SBATCH --output=logs/pipeline_row_by_row.out.txt

CONFIG_CSV="${CONFIG_CSV:-src/configs/experiments.csv}"
SAMPLES=250

# 1. Path Setup
BASE_DIR="/cluster/customapps/biomed/vogtlab/users/mwylie/toast"
PROJECT_DIR="/cluster/home/mwylie/toast-extensions"
WRITABLE_CACHE="/cluster/home/mwylie/hf_writable"
mkdir -p $WRITABLE_CACHE logs

# 2. Activate the venv
source $GPU_ENV/bin/activate
pip install -e .

# 3. Environment variables
export HF_HOME="/cluster/customapps/biomed/vogtlab/users/mwylie/toast/hf_cache"
export HF_DATASETS_CACHE="$WRITABLE_CACHE/datasets"
export TRANSFORMERS_CACHE="$WRITABLE_CACHE/transformers"
export HF_MODULES_CACHE="$WRITABLE_CACHE/modules"
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Quiet noisy libraries (progress bars / load reports) — we only want config + accuracy
export TQDM_DISABLE=1
export TRANSFORMERS_VERBOSITY=error
export HF_HUB_DISABLE_PROGRESS_BARS=1
export DATASETS_VERBOSITY=error

# 4. Fix Python imports
export PYTHONPATH=$PYTHONPATH:$PROJECT_DIR/src

cd $PROJECT_DIR

# 5. Row-by-row loop
HEADER=$(head -1 "$CONFIG_CSV")
NUM_ROWS=$(tail -n +2 "$CONFIG_CSV" | grep -c .)
echo "Running $NUM_ROWS rows from $CONFIG_CSV"

for i in $(seq 1 "$NUM_ROWS"); do
    ROW=$(sed -n "$((i + 1))p" "$CONFIG_CSV")
    [ -z "$(echo "$ROW" | tr -d '[:space:],')" ] && continue

    TEMP_CSV=$(mktemp /tmp/exp_row_XXXXXX.csv)
    printf '%s\n%s\n' "$HEADER" "$ROW" > "$TEMP_CSV"

    echo ""
    echo "=== Row $i / $NUM_ROWS ==="
    echo "$ROW"

    LOG=$(mktemp /tmp/exp_row_log_XXXXXX)

    # Phase 1: encode (silent unless it fails)
    CONFIG_CSV="$TEMP_CSV" bash src/toast/scripts/encode_vision_full.sh > "$LOG" 2>&1
    if [ $? -ne 0 ]; then
        echo "ERROR: phase 1 failed for row $i:"
        cat "$LOG"
        rm -f "$TEMP_CSV" "$LOG"
        continue
    fi

    # Phase 2a: train — print the params-saved line and the per-seed accuracy strings
    CONFIG_CSV="$TEMP_CSV" bash src/toast/scripts/train_skipped_full.sh > "$LOG" 2>&1
    if [ $? -ne 0 ]; then
        echo "ERROR: phase 2 failed for row $i:"
        cat "$LOG"
    else
        grep -E '^[[:space:]]*Params saved:' "$LOG"
        if ! grep -oE 'Accuracy: [0-9.]+' "$LOG"; then
            echo "(no result — check row config)"
        fi
    fi
    rm -f "$LOG"

    # Delete embeddings for this row (silent)
    EMB_DIR=$(python src/toast/scripts/get_embed_dir.py "$TEMP_CSV" "$SAMPLES" 2>/dev/null)
    [ -d "$EMB_DIR" ] && rm -rf "$EMB_DIR"
    rm -f "$TEMP_CSV"
done

echo "Done."
