# Transfer Learning Workflow

## Complete Workflow (3 Commands)

### 1. Submit all experiments (runs for ~1-2 days)

**Option A: Run directly on login node**
```bash
bash src/run_scripts/run_transfer_learning.sh
```

**Option B: Submit as a Slurm job**
```bash
sbatch src/run_scripts/run_transfer_learning.sh
```

Both do the same thing. Choose based on your preference:
- **Option A**: Direct control, can Ctrl+C to stop
- **Option B**: Runs as background job, can disconnect

This script:
- Submits 6 patterns in parallel
- Waits for batch to finish (~1-2 hours)
- Submits next 6
- Repeats for all 70 patterns
- Progress shown in real-time

### 2. Consolidate results (takes ~1 minute)

```bash
bash consolidate_transfer_learning.sh
```

Creates: `results_transfer_learning_all.csv` (all 1,092 experiment results combined)

### 3. Analyze results (takes ~1 minute)

```bash
python analyze_transfer_results.py \
  --results results_transfer_learning_all.csv \
  --ranking source_rankings.csv \
  --transfer-losses transfer_losses.csv
```

Creates:
- `source_rankings.csv` — Which datasets transfer best to chestmnist/pneumoniamnist
- `transfer_losses.csv` — Per-pattern transfer efficiency

### 4. View results

```bash
head -20 source_rankings.csv | column -t -s','
```

---

## Timeline

| Step | Time |
|------|------|
| Step 1: Run all experiments | ~24-36 hours (1-2 days) |
| Step 2: Consolidate | ~1 minute |
| Step 3: Analyze | ~1 minute |
| **Total** | **~1-2 days** |

---

## Storage

- **During experiments**: ~100MB per batch (cleaned up automatically)
- **After complete**: ~70KB (just results)
- **Peak**: Never exceeds ~100MB

---

## File Organization

```
src/configs/transfer_learning_patterns/     (70 config files - experiment definitions)
results/transfer_learning/                   (results stored here, created during run)
logs/                                         (job logs)
results_transfer_learning_all.csv            (created by step 2)
source_rankings.csv                          (created by step 3)
transfer_losses.csv                          (created by step 3)
```

---

## Monitor Progress (While Running)

```bash
# In another terminal:
watch -n 30 'ls -1 results/transfer_learning/results*.csv 2>/dev/null | wc -l'
```

---

## What Each Script Does

| Script | Purpose |
|--------|---------|
| `run_transfer_learning.sh` | Submit 6 patterns in parallel, wait for each batch |
| `consolidate_transfer_learning.sh` | Combine all 70 result files into one CSV |
| `analyze_transfer_results.py` | Rank datasets by transfer efficiency |

---

## If Something Fails

**Check logs:**
```bash
tail -20 logs/transfer_learning_pattern_*.out.txt
```

**Rerun one pattern:**
```bash
CONFIG_CSV="src/configs/transfer_learning_patterns/experiments_transfer_learning_pattern_pattern15.csv" \
RESULTS_CSV_NAME="results/transfer_learning/results_transfer_learning_pattern_pattern15.csv" \
  sbatch --output="logs/transfer_learning_pattern_pattern15.out.txt" \
    src/run_scripts/run_pipeline_row_by_row.sh
```

**Manually consolidate and analyze:**
```bash
bash consolidate_transfer_learning.sh
python analyze_transfer_results.py --results results_transfer_learning_all.csv --ranking source_rankings.csv
```

---

## That's it!

3 commands, ~1-2 days, done.
