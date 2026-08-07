#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:rtx4090:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=48000
#SBATCH --time=48:00:00
#SBATCH --output=logs/table3_medical.out.txt

# Table 3 replicated on medical data: whole-block skip vs MLP-only vs attention-only removal
# at matched parameter savings, for dermamnist, pneumoniamnist and pathmnist.
#
#   sbatch src/run_scripts/run_table3_medical.sh
#
# The spans are held fixed at the values from the prior paper rather than re-derived per
# dataset, because that IS the experiment: do blocks that were redundant on natural images
# stay redundant on medical images? rad-dino reuses DINOv2-base's spans -- same architecture,
# different pretraining domain -- which makes that pair a controlled comparison.
#
# 219 rows. The config is dataset-major and contiguous, so cancelling partway leaves complete
# results for the datasets it reached (dermamnist, then pneumoniamnist, then pathmnist) rather
# than holes across all three.
#
# Requires medmnist_{dermamnist,pneumoniamnist,pathmnist}_clean on disk -- see
# src/run_scripts/download_medmnist.sh.

CONFIG_CSV="${CONFIG_CSV:-src/configs/experiments_table3_medical.csv}"
RESULTS_CSV_NAME="${RESULTS_CSV_NAME:-results_table3_medical.csv}"
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
echo "  Aggregating"
echo "============================================================"
python skipping_heads/calculate_accuracies.py \
    --input "results/${RESULTS_CSV_NAME}" \
    --output "results/accuracies_${RESULTS_CSV_NAME}"

echo ""
echo "============================================================"
echo "  Collapse check"
echo "============================================================"
# A probe that scores the majority-class rate has learned nothing. This is what went unnoticed
# on chestmnist for 520 rows, so it is checked automatically here rather than by eye.
python - <<'PYEOF'
import os
import pandas as pd
p = f"results/{os.environ['RESULTS_CSV_NAME']}"
try:
    df = pd.read_csv(p)
except Exception as e:
    print(f"  could not read {p}: {e}"); raise SystemExit(0)
if "majority_class_rate" not in df.columns:
    print("  no majority_class_rate column"); raise SystemExit(0)
d = df.dropna(subset=["majority_class_rate"])
bad = d[d["accuracy"] <= d["majority_class_rate"] + 1e-6]
for ds, g in d.groupby("dataset"):
    mr = g["majority_class_rate"].iloc[0]
    base = g[g["approx_layer"] == "[]"]["accuracy"].mean()
    flag = "  <-- COLLAPSED, results meaningless" if base <= mr + 1e-6 else ""
    print(f"  {ds:16} majority {mr:.4f} | baseline probe {base:.4f}{flag}")
if len(bad):
    print(f"\n  {len(bad)}/{len(d)} rows at or below the majority rate.")
    print("  Those configs learned nothing -- do not read accuracy differences into them.")
else:
    print("\n  No row collapsed to the majority class.")
PYEOF

echo ""
echo "Results: results/${RESULTS_CSV_NAME}"

exit $STATUS
