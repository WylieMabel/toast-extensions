#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:rtx4090:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=48000
#SBATCH --time=72:00:00
#SBATCH --output=logs/layer_pipeline.out.txt

# =============================================================================
# One-shot layer-redundancy pipeline. For every ENCODER x DATASET combo:
#   1. layer_priority.py   -> redundancy score files (src/layers/outputs)
#   2. analysis_plots.py   -> saves every plot into PLOTS_DIR/<stem>/
#   3. recommend_runs.py   -> appends candidate rows into EXPERIMENTS_CSV
# then runs every row (encode + train over the seeds) via the existing
# run_pipeline_row_by_row.sh, and finally aggregates mean/std per unique run.
#
# EDIT THE CONFIG BLOCK BELOW. Seeds are set in train_skipped_full.sh (1..5).
# =============================================================================

# -------------------------------- CONFIG -------------------------------------
RESULTS_PATH="results/results_pipeline.csv"          # per-seed rows land here
EXPERIMENTS_CSV="src/configs/experiments_pipeline.csv"  # assembled candidate rows
DATASETS=(chestmnist dermamnist cifar100)            # datasets to analyse
ENCODERS=(                                           # HF encoder ids to analyse
    facebook/deit-small-patch16-224
    facebook/deit-base-patch16-224
    facebook/dinov2-base
    google/vit-base-patch16-224
    google/vit-large-patch16-224
    microsoft/rad-dino
)
NUM_SAMPLES=5000                                     # images for the analysis pass
OUTPUTS_DIR="src/layers/outputs"                     # score files (.csv/.npy)
PLOTS_DIR="src/layers/plots"                         # per-run plot folders
FORCE_LAYER_SCORES=0                                 # 1 = recompute scores even if they exist
# -----------------------------------------------------------------------------

# 1. Path setup (mirrors run_pipeline_row_by_row.sh)
BASE_DIR="/cluster/customapps/biomed/vogtlab/users/mwylie/toast"
PROJECT_DIR="/cluster/home/mwylie/toast-extensions"
WRITABLE_CACHE="/cluster/home/mwylie/hf_writable"
mkdir -p "$WRITABLE_CACHE" logs

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
export TQDM_DISABLE=1
export TRANSFORMERS_VERBOSITY=error
export HF_HUB_DISABLE_PROGRESS_BARS=1
export DATASETS_VERBOSITY=error

# 4. Fix Python imports
export PYTHONPATH=$PYTHONPATH:$PROJECT_DIR/src
cd $PROJECT_DIR

# 5. Tell train_skipped_full.py which results file to write (basename under results/)
export RESULTS_CSV_NAME="$(basename "$RESULTS_PATH")"

# 6. Per-combo analysis -> plots -> recommended runs
rm -f "$EXPERIMENTS_CSV"          # reset at start; accumulate within this run
first=1
for enc in "${ENCODERS[@]}"; do
    for ds in "${DATASETS[@]}"; do
        stem="$(echo "${enc//\//_}_${ds}" | tr '[:upper:]' '[:lower:]')"
        echo ""
        echo "############################################################"
        echo "  Combo: ${enc}  x  ${ds}   (stem: ${stem})"
        echo "############################################################"

        # 1) redundancy scores (skips if score files already exist unless forced)
        force_flag=""
        [ "$FORCE_LAYER_SCORES" -eq 1 ] && force_flag="--force"
        python src/layers/layer_priority.py \
            --model "$enc" --dataset "$ds" \
            --num-samples "$NUM_SAMPLES" --output-dir "$OUTPUTS_DIR" $force_flag

        # 2) save every analysis plot into PLOTS_DIR/<stem>/
        python src/layers/analysis_plots.py \
            --stem "$stem" --output-dir "$OUTPUTS_DIR" \
            --plot-dir "$PLOTS_DIR/$stem"

        # 3) append recommended runs (header only on the first combo)
        append_flag=""
        [ "$first" -eq 0 ] && append_flag="--append"
        python src/layers/recommend_runs.py \
            --stem "$stem" --dataset "$ds" --encoder "$enc" \
            --output-dir "$OUTPUTS_DIR" --out-csv "$EXPERIMENTS_CSV" $append_flag

        first=0
    done
done

# 7. Run every assembled row (encode + train over all seeds, memory-safe cleanup)
echo ""
echo "############################################################"
echo "  Running all rows from $EXPERIMENTS_CSV"
echo "############################################################"
CONFIG_CSV="$EXPERIMENTS_CSV" bash src/run_scripts/run_pipeline_row_by_row.sh

# 8. Mean + std of accuracy per unique run
echo ""
echo "############################################################"
echo "  Aggregating mean/std -> results/accuracies_pipeline.csv"
echo "############################################################"
python skipping_heads/calculate_accuracies.py \
    --input "$RESULTS_PATH" --output "results/accuracies_pipeline.csv"

echo "Done."
