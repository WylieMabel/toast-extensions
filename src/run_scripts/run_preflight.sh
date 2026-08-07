#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:rtx4090:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=48000
#SBATCH --time=1:00:00
#SBATCH --output=logs/preflight.out.txt

# Cheap checks to run BEFORE queueing any long sweep. Roughly 10 minutes total.
#
#   sbatch src/run_scripts/run_preflight.sh
#
# 1. Verifies the low-rank translators work against the real latentis classes (registry, fit,
#    transform, save/load round-trip, parameter accounting). This is the part that cannot be
#    checked without latentis installed, so it is the most likely thing to be broken.
# 2. Reports reconstruction error vs translator rank on real activations, so the rank grid can
#    be trimmed before spending GPU hours on ranks whose curve has already flattened.
#
# Exits non-zero if the smoke test fails -- if that happens, do not queue anything else.

# 1. Path setup
PROJECT_DIR="/cluster/home/mwylie/toast-extensions"
WRITABLE_CACHE="/cluster/home/mwylie/hf_writable"
mkdir -p $WRITABLE_CACHE logs

# 2. Activate venv
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

# 4. Threading + crash diagnostics
#
# torch, numpy and scipy are often linked against different OpenMP runtimes (libgomp vs
# libiomp5). When that happens the first threaded LAPACK call segfaults -- no Python
# traceback, just "Segmentation fault (core dumped)". Serialising the threading avoids it, and
# costs nothing here: these are D x D decompositions on a 64-dim problem, not the bottleneck.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_THREADING_LAYER=GNU

# Print the Python stack if a native crash does happen, and never buffer stdout -- otherwise a
# segfault discards everything printed so far and the log looks empty.
export PYTHONFAULTHANDLER=1
export PYTHONUNBUFFERED=1

# 5. Fix Python imports
export PYTHONPATH=$PYTHONPATH:$PROJECT_DIR/src

cd $PROJECT_DIR

echo "============================================================"
echo "  environment"
echo "============================================================"
echo "  python : $(which python)"
python -c "import torch, numpy; print(f'  torch  : {torch.__version__}'); print(f'  numpy  : {numpy.__version__}')"
python -c "import latentis; print(f'  latentis: {latentis.__file__}')" 2>&1 | tail -1
echo "  OMP_NUM_THREADS=$OMP_NUM_THREADS  MKL_THREADING_LAYER=$MKL_THREADING_LAYER"

MODEL="${MODEL:-dinobase}"
DATASET="${DATASET:-imagenet-1k}"
SPANS="${SPANS:-[(5,6),(2,4),(0,1)]}"

echo ""
echo "============================================================"
echo "  1/2  Low-rank translator smoke test"
echo "============================================================"
python -u src/toast/scripts/verify_lowrank.py
STATUS=$?

if [ $STATUS -ne 0 ]; then
    echo ""
    echo "------------------------------------------------------------"
    if [ $STATUS -gt 128 ]; then
        SIG=$((STATUS - 128))
        echo "  CRASHED with signal $SIG (139 = segfault), not a failed assertion."
        echo ""
        echo "  The last '...' line above names the operation that died."
        echo ""
        echo "  If it died in section 0 (matmul / lstsq / svd / eigh), the fault is"
        echo "  environmental rather than a bug in the translators -- those calls are plain"
        echo "  torch. This job already sets OMP_NUM_THREADS=1 and MKL_THREADING_LAYER=GNU,"
        echo "  so if it still crashes the venv's torch/numpy/scipy builds are incompatible."
        echo "  Try, in the venv:"
        echo "      pip install --force-reinstall --no-cache-dir numpy"
        echo "  or rebuild the venv with matching torch/numpy wheels."
        echo ""
        echo "  If it died in section 0b or later, the crash is in latentis or toast code"
        echo "  and the faulthandler stack above pinpoints the line."
    else
        echo "  A CHECK FAILED (exit $STATUS). The 'FAIL' line above names it."
    fi
    echo ""
    echo "  Do not queue the sweep until this passes."
    echo "------------------------------------------------------------"
    exit 1
fi

echo ""
echo "============================================================"
echo "  2/2  Rank pre-check: ${MODEL} x ${DATASET}"
echo "============================================================"
python src/layers/rank_precheck.py \
    --model "$MODEL" \
    --dataset "$DATASET" \
    --spans "$SPANS" \
    --out "src/layers/outputs/rank_precheck_${MODEL}_${DATASET}.csv"

echo ""
echo "Pre-flight complete."
echo "If rel_error has flattened by r=64, regenerate a smaller grid before the sweep:"
echo "  python src/configs/make_lowrank_sweep.py --ranks 8 16 32 64 > src/configs/experiments_lowrank.csv"
