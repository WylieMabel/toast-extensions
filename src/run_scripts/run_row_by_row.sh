#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:rtx4090:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24000
#SBATCH --time=48:00:00
#SBATCH --output=logs/run_row_by_row_%j.out.txt

# ── User config ────────────────────────────────────────────────
CONFIG_CSV="${CONFIG_CSV:-src/configs/experiments.csv}"
SAMPLES=500
SEEDS="1 2 3"
CLASSIFIER=linear
TRANSLATOR=identity

# ── Paths ──────────────────────────────────────────────────────
PROJECT_DIR="/cluster/home/mwylie/toast-extensions"
BASE_DIR="/cluster/customapps/biomed/vogtlab/users/mwylie/toast"
WRITABLE_CACHE="/cluster/home/mwylie/hf_writable"

mkdir -p "$WRITABLE_CACHE" logs

source "${BASE_DIR}/gpu_env/bin/activate"
pip install -q -e .

export HF_HOME="${BASE_DIR}/hf_cache"
export HF_DATASETS_CACHE="$WRITABLE_CACHE/datasets"
export TRANSFORMERS_CACHE="$WRITABLE_CACHE/transformers"
export HF_MODULES_CACHE="$WRITABLE_CACHE/modules"
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$PYTHONPATH:$PROJECT_DIR/src"

cd "$PROJECT_DIR"

# ── Row-by-row loop ────────────────────────────────────────────
HEADER=$(head -1 "$CONFIG_CSV")
NUM_ROWS=$(tail -n +2 "$CONFIG_CSV" | grep -c .)

echo "Processing $NUM_ROWS rows from $CONFIG_CSV"

for i in $(seq 1 "$NUM_ROWS"); do
    ROW=$(sed -n "$((i + 1))p" "$CONFIG_CSV")

    # Skip blank lines
    [ -z "$(echo "$ROW" | tr -d '[:space:]')" ] && continue

    TEMP_CSV=$(mktemp /tmp/exp_row_XXXXXX.csv)
    printf '%s\n%s\n' "$HEADER" "$ROW" > "$TEMP_CSV"

    echo ""
    echo "══════════════════════════════════════════════"
    echo "  Row $i / $NUM_ROWS"
    echo "  $ROW"
    echo "══════════════════════════════════════════════"

    # ── Phase 1: encode ────────────────────────────────────────
    echo "→ Phase 1: encoding"
    python src/toast/utils/encode_vision_full.py \
        --translator_name="$TRANSLATOR" \
        --seed=0 \
        --config_csv="$TEMP_CSV" \
        --samples_to_extract="$SAMPLES"

    if [ $? -ne 0 ]; then
        echo "  ERROR: encoding failed for row $i. Skipping."
        rm -f "$TEMP_CSV"
        continue
    fi

    # ── Phase 2a: train (all seeds) ────────────────────────────
    echo "→ Phase 2a: training"
    TRAIN_FAILED=0
    for seed in $SEEDS; do
        echo "  seed=$seed"
        python src/toast/utils/train_skipped_full.py \
            --seed="$seed" \
            --classifier_type="$CLASSIFIER" \
            --translator_name="$TRANSLATOR" \
            --samples_to_extract="$SAMPLES" \
            --config_csv="$TEMP_CSV"

        if [ $? -ne 0 ]; then
            echo "  ERROR: training failed for row $i seed $seed."
            TRAIN_FAILED=1
        fi
    done

    # ── Delete embeddings ──────────────────────────────────────
    EMB_DIR=$(python src/toast/scripts/get_embed_dir.py "$TEMP_CSV" "$SAMPLES")
    if [ -d "$EMB_DIR" ]; then
        echo "→ Deleting embeddings: $EMB_DIR"
        rm -rf "$EMB_DIR"
    else
        echo "  WARNING: embedding dir not found at $EMB_DIR"
    fi

    rm -f "$TEMP_CSV"

    if [ $TRAIN_FAILED -ne 0 ]; then
        echo "  WARNING: one or more training seeds failed for row $i."
    else
        echo "  Row $i complete."
    fi
done

echo ""
echo "All rows complete."
