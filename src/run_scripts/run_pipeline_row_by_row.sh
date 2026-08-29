#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:rtx4090:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=48000
#SBATCH --time=72:00:00
# Log naming: use LOG_PREFIX if provided, otherwise %j keeps concurrent submissions from
# writing over each other's logs. This script is routinely run several times at once,
# one per config shard.
#SBATCH --output=logs/pipeline_row_by_row_%j.out.txt

usage() {
    cat <<'USAGE'
Usage: [sbatch] run_pipeline_row_by_row.sh [CONFIG_CSV] [RESULTS_CSV_NAME]

  CONFIG_CSV         sweep to run; path relative to the project root. Default:
                     src/configs/experiments_table3_medical_missing.csv
  RESULTS_CSV_NAME   basename written under results/. Default: the config's own name with
                     experiments_ swapped for results_, so concurrent shards land in separate
                     files without having to be told to.

Both are also readable from the environment under the same names; a positional argument wins
over the environment. Concurrent jobs MUST end up with different RESULTS_CSV_NAMEs -- every
row rewrites that file in full, so two jobs sharing one lose each other's rows.

  sbatch src/run_scripts/run_pipeline_row_by_row.sh src/configs/experiments_skip_grid_part1.csv

To customize the log file name, pass --output to sbatch:
  CONFIG_CSV=src/configs/experiments_transfer_learning_part1.csv \
  RESULTS_CSV_NAME=results_transfer_learning_part1.csv \
    sbatch --output=logs/transfer_learning_part1.out.txt \
      src/run_scripts/run_pipeline_row_by_row.sh
USAGE
}

case "$1" in
    -h|--help) usage; exit 0 ;;
esac

if [ "$#" -gt 2 ]; then
    echo "ERROR: expected at most 2 arguments, got $#." >&2
    usage >&2
    exit 2
fi

CONFIG_CSV="${1:-${CONFIG_CSV:-src/configs/experiments_table3_medical_missing.csv}}"
RESULTS_CSV_NAME="${2:-${RESULTS_CSV_NAME:-}}"

# Derive the results name from the config when it was not given. experiments_foo.csv ->
# results_foo.csv, which reproduces the old hard-coded pair for the default config and, more
# usefully, makes a per-shard results file the automatic outcome of passing a per-shard config.
stem=$(basename "$CONFIG_CSV" .csv)
DERIVED_RESULTS="results_${stem#experiments_}.csv"
if [ -z "$RESULTS_CSV_NAME" ]; then
    RESULTS_CSV_NAME="$DERIVED_RESULTS"
elif [ -n "$1" ] && [ -z "$2" ] && [ "$RESULTS_CSV_NAME" != "$DERIVED_RESULTS" ]; then
    # sbatch forwards the submitting shell's environment, so a RESULTS_CSV_NAME exported for
    # an earlier run silently attaches itself to a config passed positionally here. Left
    # unnoticed across concurrent shards that is precisely how rows get overwritten.
    echo "WARNING: config given as an argument but RESULTS_CSV_NAME=$RESULTS_CSV_NAME comes" >&2
    echo "         from the environment (would otherwise be $DERIVED_RESULTS). Unset it to" >&2
    echo "         derive the name from the config." >&2
fi
export RESULTS_CSV_NAME
SAMPLES=500

# Images now vary by seed (encode_vision_full.py shuffles the train split using the seed's
# RNG state), so phase 1 must be re-run per seed rather than once per row. Test images stay
# fixed across seeds (see encode_vision_full.py) so accuracy deltas between seeds reflect
# training noise, not an easier/harder eval sample.
SEEDS="${SEEDS:-0 1 2}"

# The loop below runs each row in its own process with a single-row temp CSV, so that process
# cannot see whether a *later* row transfers from the translator it is about to fit. Exporting
# the whole sweep lets encode_vision_full.py answer that and save only what will be read back.
export SWEEP_CSV="$CONFIG_CSV"

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

# Checked here rather than at the top: CONFIG_CSV is conventionally relative to the project
# root, which only becomes the working directory now. A missing file otherwise reaches the
# loop as zero rows and the job "succeeds" having run nothing.
if [ ! -f "$CONFIG_CSV" ]; then
    echo "ERROR: config CSV not found: $CONFIG_CSV (looked in $PROJECT_DIR)" >&2
    exit 2
fi

# 5. Row-by-row loop
HEADER=$(head -1 "$CONFIG_CSV")
NUM_ROWS=$(tail -n +2 "$CONFIG_CSV" | grep -c .)
echo "Config : $CONFIG_CSV"
echo "Results: results/$RESULTS_CSV_NAME"
echo "Running $NUM_ROWS rows from $CONFIG_CSV"

FAILED_ROWS=""

# Delete this row's embeddings (~11GB for imagenet-1k + vit-large). Has to run on the failure
# paths too, not just the happy one: a row that dies partway through save_to_disk can leave a
# *_temp directory behind, and on a quota-limited home those accumulate until nothing runs.
cleanup_row_embeddings() {
    local emb_dir
    emb_dir=$(python src/toast/scripts/get_embed_dir.py "$1" "$SAMPLES" 2>/dev/null)
    [ -n "$emb_dir" ] && [ -d "$emb_dir" ] && rm -rf "$emb_dir"
    [ -n "$emb_dir" ] && [ -d "${emb_dir}_temp" ] && rm -rf "${emb_dir}_temp"
    return 0
}

for i in $(seq 1 "$NUM_ROWS"); do
    ROW=$(sed -n "$((i + 1))p" "$CONFIG_CSV")
    [ -z "$(echo "$ROW" | tr -d '[:space:],')" ] && continue

    TEMP_CSV=$(mktemp /tmp/exp_row_XXXXXX.csv)
    printf '%s\n%s\n' "$HEADER" "$ROW" > "$TEMP_CSV"

    echo ""
    echo "=== Row $i / $NUM_ROWS ==="
    echo "$ROW"

    ROW_FAILED=0
    for seed in $SEEDS; do
        echo "  -- seed $seed --"

        # Force a fresh encode: phase 1 skips outright if this row's embedding dir already
        # has a dataset_dict.json (see encode_vision_full.py), which would otherwise reuse
        # the previous seed's images.
        cleanup_row_embeddings "$TEMP_CSV"

        LOG=$(mktemp /tmp/exp_row_log_XXXXXX)

        # Phase 1: encode (silent unless it fails)
        SEED="$seed" CONFIG_CSV="$TEMP_CSV" bash src/toast/scripts/encode_vision_full.sh > "$LOG" 2>&1
        if [ $? -ne 0 ]; then
            echo "ERROR: phase 1 failed for row $i, seed $seed:"
            cat "$LOG"
            FAILED_ROWS="$FAILED_ROWS $i"
            ROW_FAILED=1
            cleanup_row_embeddings "$TEMP_CSV"
            rm -f "$LOG"
            break
        fi

        # Phase 2: train — single seed per call, print params-saved + accuracy
        SEEDS="$seed" CONFIG_CSV="$TEMP_CSV" bash src/toast/scripts/train_skipped_full.sh > "$LOG" 2>&1
        if [ $? -ne 0 ]; then
            echo "ERROR: phase 2 failed for row $i, seed $seed:"
            cat "$LOG"
            FAILED_ROWS="$FAILED_ROWS $i"
            ROW_FAILED=1
        else
            grep -E '^[[:space:]]*Params saved:' "$LOG"
            if ! grep -oE 'Accuracy: [0-9.]+' "$LOG"; then
                # Phase 2 produced no accuracy despite exiting 0. Keep the log rather than
                # deleting it -- this is exactly the case where the cause is in the text and
                # discarding it leaves only the symptom.
                KEPT="logs/failed_row_${i}_seed_${seed}.log"
                cp "$LOG" "$KEPT"
                echo "(no result — full log kept at $KEPT)"
                FAILED_ROWS="$FAILED_ROWS $i"
                ROW_FAILED=1
            fi
        fi
        rm -f "$LOG"

        cleanup_row_embeddings "$TEMP_CSV"
        [ "$ROW_FAILED" -eq 1 ] && break
    done

    rm -f "$TEMP_CSV"
done

# Wipe the translators this sweep created, pass or fail. Scoped to this sweep's own keys --
# the store is shared, so a blanket rm would take out maps another sweep still needs. Sweeps
# with no transfer rows saved nothing and this is a no-op.
SAVED_TRANSLATORS=$(python src/toast/scripts/sweep_translators.py "$SWEEP_CSV" "$SAMPLES" 2>/dev/null)
if [ -n "$SAVED_TRANSLATORS" ]; then
    echo ""
    echo "Removing $(echo "$SAVED_TRANSLATORS" | wc -l | tr -d ' ') translator(s) saved by this sweep:"
    echo "$SAVED_TRANSLATORS" | while IFS= read -r d; do
        [ -n "$d" ] && [ -d "$d" ] && du -sh "$d" && rm -rf "$d"
    done
fi

# Package the per-seed results into one mean/std-per-config summary, same aggregation
# skipping_heads/calculate_accuracies.py already uses elsewhere. Runs even on a partial/failed
# sweep -- whatever rows did complete are still worth summarizing rather than only ever
# available as a manual follow-up step.
SUMMARY_NAME="accuracies_${RESULTS_CSV_NAME}"
if [ -f "results/$RESULTS_CSV_NAME" ]; then
    python skipping_heads/calculate_accuracies.py \
        --input "results/$RESULTS_CSV_NAME" \
        --output "results/$SUMMARY_NAME"
    echo "Summary (mean/std per config): results/$SUMMARY_NAME"
fi

if [ -n "$FAILED_ROWS" ]; then
    echo ""
    echo "=================================================="
    echo "INCOMPLETE: $(echo $FAILED_ROWS | wc -w) of $NUM_ROWS rows failed:$FAILED_ROWS"
    echo "results/$RESULTS_CSV_NAME is missing those configs."
    echo "=================================================="
    exit 1
fi

echo "Done. All $NUM_ROWS rows completed."
