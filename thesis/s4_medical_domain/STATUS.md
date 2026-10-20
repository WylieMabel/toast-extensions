# s4 status — 2026-10-12

## What s4 is
Contiguous skips of length 1–4 plus a baseline (same skip list as s2), linear translator, seeds 0 1 2.
- Chest: chestxray14 / chestxray_pneumonia × rad-dino / rad-dino-maira-2 / dinov2-base
- Hist: tcga_ut / patchcamelyon × vit-large / phikon-v2

There are 24 configs in `configs/`. Long combos are split by skip length (`_len1`–`_len4`; rad-dino and MAIRA chestxray14 are split into `_len12` / `_len34`). Only the first part of each split has the baseline row, so `delta_acc` is NaN in later parts; compute it from part 1's baseline in analysis.

## How it was launched
`launch_s4.sh` submitted 6 groups. Within a group, jobs are chained with `afterany`; each group's first job waited on one of the s2/s3 ViT-L jobs. At most 6 GPUs are in use at once, and each group is about 71h by the smoke-test estimates. Job IDs run 9728291–9728314 in launcher order:

| Group | Jobs (in order) |
|---|---|
| 1 | 9728291 raddino_cx14_len12 ✗, 9728292 maira_cx14_len12 ✗, 9728293 maira_pneu, 9728294 dinov2_pneu, 9728295 phikon_pcam_len2 |
| 2 | 9728296 raddino_cx14_len34 ✗, 9728297 maira_cx14_len34 ✗, 9728298 dinov2_cx14, 9728299 raddino_pneu, 9728300 vitL_pcam_len1 |
| 3 | 9728301 phikon_pcam_len1, 9728302 phikon_pcam_len4, 9728303 vitL_pcam_len4 |
| 4 | 9728304 vitL_tcga_len1 (some rows ✗), 9728305 phikon_tcga_len2, 9728306 phikon_tcga_len3, 9728307 vitL_tcga_len4 |
| 5 | 9728308 vitL_pcam_len2, 9728309 phikon_pcam_len3, 9728310 vitL_pcam_len3 |
| 6 | 9728311 phikon_tcga_len1, 9728312 vitL_tcga_len2, 9728313 vitL_tcga_len3, 9728314 phikon_tcga_len4 |

## What went wrong and what was fixed
- **Out-of-memory failures.** The one-row smoke tests' sacct MaxRSS missed short memory spikes, so some requests were too low.
  - rad-dino / MAIRA chestxray14 (48G): every row was killed about 40s in, during the 500-sample capture concat.
  - vitL tcga_ut len1 (36G): about 40% of seeds were killed at the end of the full encode. Failed rows: 1, 5, 6, 7, 8 (seed 2 only), 10–14, as of row 14.
- **Fix (no effect on results).** Two changes free memory that was no longer needed:
  - `utils.py` `extract_representations` frees each layer's batch pieces once that layer is concatenated.
  - `encode_vision_full.py` drops the capture once the translator is fitted.
  - Each row runs as a fresh Python process, so running jobs picked the fix up row by row once it was synced.
- **Verified.** The memfix smoke test passed for rad-dino chestxray14 at 48G and vitL tcga_ut at 36G. Accuracies matched the original smoke runs exactly (`../smoke_timing/launch_memfix_test.sh`). No vitL tcga rows have failed since the sync.
- **Memory raised on pending jobs as a precaution** (scontrol takes MB): 9728299 → 73728, and 9728307 / 9728312 / 9728313 → 61440.

## Still to do
- [ ] Check the requests above took effect: `squeue -u $USER -o "%A %j %m %T"`.
- [x] Ran `requeue_chest14.sh`: the four failed chestxray14 jobs are resubmitted at the end of groups 1 and 2 (after 9728295 and 9728300), with 6 CPUs and 48G. New job IDs are in the script's output and in `squeue -u $USER`.
- [x] vitL_tcga_len1 (9728304) finished with 11 of 24 rows failed: 1 5 6 7 8 10 11 12 13 14 15 (one OOM'd seed each; the whole row is dropped). Rerun config: `configs/experiments_vitlarge_tcgaut_len1_rerun.csv` (includes the baseline). It writes to `results_vitlarge_tcgaut_len1_rerun.csv`; merge it with the original in analysis.
- [ ] Run `requeue_vitlarge_tcgaut_len1.sh` from the project root. It appends the rerun to the end of group 4 (after 9728307) at 36G, 4 CPUs, 24h; estimated ~9.5h.
- [ ] When everything has finished, scan all logs for `ERROR: phase` / `Killed` / `INCOMPLETE` and collect any other rows to rerun.

## Combining and metrics (2026-10-15)
All jobs finished; every log ends `Done. All N rows completed.` (the first raddino/MAIRA chestxray14 attempts failed, and their requeues completed).
- **Results overwritten.** Six results CSVs were overwritten on the cluster by copies taken mid-run on 2026-10-13 around 18:52, pushed back up after the jobs had finished: phikonv2 pcam_len2/len3/len4, phikonv2 tcga_len3, vitL pcam_len1, vitL tcga_len2. Their per-seed rows are partly lost. The per-job accuracies CSVs are intact. Each was written from the full results at the end of its job, and all of them match the per-seed accuracies in the job logs (all 3 seeds for every span). Exclude `results/` and `accuracies/` when pushing to the cluster.
- `python thesis/s4_medical_domain/combine_s4.py` writes one `results/combined/results_<alias>_<tag>.csv` and one `accuracies/combined/accuracies_results_<alias>_<tag>.csv` per combo. Accuracies are recomputed from the per-seed rows where all 3 seeds exist. The 50 spans whose rows were lost are taken from the per-job accuracies CSVs. All 10 combos are complete, and every row matches the per-job files.
- [ ] Stage 1: `bash thesis/s4_medical_domain/launch_s4_metrics.sh` (10 jobs; resources are estimates). Then `python metric_experiments/build_group_stats.py --group s4`, and set `USE_TEMP = False` in `analysis/s4_metric_analysis.ipynb`.

## Timing
At the last check no running job was at risk of timing out. The tightest was phikon_pcam_len1, at about 27.7h of 36h. Rows are running 2–17% slower than in the smoke tests. Users can't extend time limits (only admins can); if a job times out, only its in-progress row is lost, so add it to a rerun config.

## Resource requests
These are set per model × dataset in `launch_s4.sh`, from sacct and sstat measurements: 4 CPUs by default, 6 for rad-dino / MAIRA chestxray14 and 8 for dinov2 chestxray14; 20–72G memory. This follows the cluster admin's email about over-requesting (96G / 8 CPUs before).
