#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:rtx4090:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=48000
#SBATCH --time=12:00:00
#SBATCH --output=logs/transfer.out.txt

# Translator transferability: fit the closed-form translator on one dataset, evaluate it on
# another. If it transfers, the translator captures structure intrinsic to the model rather
# than statistics of the dataset it was fitted on.
#
#   sbatch src/run_scripts/run_transfer.sh
#
# ROW ORDER MATTERS. Donor rows (fit_dataset == dataset) fit and save a translator; transfer
# rows load it. experiments_transfer.csv is generated with the donors first, so do not sort or
# shuffle it. A transfer row whose donor has not run raises FileNotFoundError naming the key.
#
# Fitted translators persist in data/translators/ between rows and between jobs -- unlike
# embeddings, which the row-by-row runner deletes per row. So a re-run reuses existing
# translators, and deleting that directory forces a clean refit.

CONFIG_CSV="${CONFIG_CSV:-src/configs/experiments_transfer.csv}"
RESULTS_CSV_NAME="${RESULTS_CSV_NAME:-results_transfer.csv}"
export CONFIG_CSV RESULTS_CSV_NAME
[ -n "$SEEDS" ] && export SEEDS

PROJECT_DIR="/cluster/home/mwylie/toast-extensions"
cd $PROJECT_DIR

# run_pipeline_row_by_row.sh activates the venv, but it runs in a subshell so that activation
# does not reach the aggregation step below. Activate here too.
WRITABLE_CACHE="/cluster/home/mwylie/hf_writable"
mkdir -p $WRITABLE_CACHE logs
source $GPU_ENV/bin/activate
export HF_HOME="/cluster/customapps/biomed/vogtlab/users/mwylie/toast/hf_cache"
export HF_DATASETS_CACHE="$WRITABLE_CACHE/datasets"
export TRANSFORMERS_CACHE="$WRITABLE_CACHE/transformers"
export HF_MODULES_CACHE="$WRITABLE_CACHE/modules"
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH=$PYTHONPATH:$PROJECT_DIR/src

bash src/run_scripts/run_pipeline_row_by_row.sh
STATUS=$?

echo ""
echo "============================================================"
echo "  Translators saved"
echo "============================================================"
ls -1 data/translators 2>/dev/null || echo "  (none -- every row used identity, or all rows failed)"

echo ""
echo "============================================================"
echo "  Aggregating"
echo "============================================================"
python skipping_heads/calculate_accuracies.py \
    --input "results/${RESULTS_CSV_NAME}" \
    --output "results/accuracies_${RESULTS_CSV_NAME}"

echo ""
echo "Results: results/${RESULTS_CSV_NAME}"
echo ""
echo "The check that decides whether any transfer number is meaningful:"
echo "  the fit_dataset == dataset control must reproduce the accuracy of the same config"
echo "  run with a blank fit_dataset. If it does not, the translator is not being reloaded"
echo "  the way you think it is."

exit $STATUS
