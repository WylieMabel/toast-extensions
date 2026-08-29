# Experiment status report — for supervisor meeting 2026-08-19

Exhaustive scan of the repo, organized by experiment. For each: purpose, what was
actually run, results (real numbers pulled from the CSVs/notebooks, not
approximations), why it worked or didn't, and a one-line conclusion. Data-quality
issues are flagged inline rather than smoothed over — several are worth deciding
on before they're quoted anywhere.

A "possible stories" section is at the end, since that's the actual decision to
make tomorrow.

---

## 1. Core TOAST pipeline reproduction

**Purpose.** Establish the baseline claim before extending it: skip a contiguous
run of transformer blocks, bridge the gap with a translator (identity or a
fitted linear map), see how much accuracy survives with no retraining.

**Method.** `results/accuracies_pipeline.csv` / `dedup_results_pipeline.csv`.
5 encoders (deit-small, dinov2-base, vit-base, vit-large, rad-dino) × 3 datasets
(chestmnist, dermamnist, cifar100) × ~17 contiguous skip spans, identity or
linear bridge, seeds 1–5 (1275 rows deduped).

**Results.** Baselines: ~0.89 chestmnist, 0.72–0.76 dermamnist, 0.71–0.87
cifar100 — except rad-dino on cifar100, which collapses to 0.29–0.33 (rad-dino
is a radiology-only encoder; a natural-image task is out of its domain, not a
bug). Identity-mode deltas average +0.01, up to +0.47 in places — many skips
cost almost nothing on these small/easy evals.

**Why.** Confirms the core premise (some blocks are near-redundant) cheaply
before scaling up. The generous "improves accuracy" deltas are most likely
eval-set noise (num_samples 250–500) rather than a real effect.

**Conclusion.** Foundation holds. **Flag:** every row tagged `translator=linear`
in `dedup_results_pipeline.csv` (130 rows) has `original_accuracy=0.0` and
`delta_acc=0.0` — these look like placeholder/broken rows, not real linear
evaluations. Don't cite linear-vs-identity comparisons from this specific file.

---

## 2. Skip grid — scaled-up reproduction

**Purpose.** Same question as #1, at the scale the rest of the thesis is built
on.

**Method.** `results/accuracies_skip_grid.csv`, 1068 rows: 6 encoders (adds
deit-base) × 4 datasets (cifar100, dermamnist, pneumoniamnist, **imagenet-1k**)
× contiguous skip spans.

**Results/Why.** This is the sweep that made multi-seed + ImageNet-1k support
real (commit `6c95ed4`); it's the data source for the block-distance and
skip-candidate-predictor work in §5–6, not an endpoint in itself.

**Conclusion.** Finished, and it's the load-bearing dataset for later analysis
— worth naming explicitly as "where the numbers come from" in any story.

---

## 3. Linear-approximation window sweep (single span)

**Purpose.** For every model, sweep every contiguous skip window bridged with a
**fitted linear** translator, to map accuracy retention against layer position
and span size — the headline "how much can we cut" chart.

**Method.** `results_window_grid_combined.csv` (2034 rows) → `thesis_analysis.ipynb`
→ `linear_approximations_sorted.csv` (660 rows). 6 encoders × 3 datasets
(cifar100, imagenet-1k, pneumoniamnist), seeds 1–3, single span per config.

**Results.** Mean retention 88.6%, median 97.3%; 58% of configs (380/660) keep
≥95% of baseline accuracy. Best: rad-dino, imagenet-1k, span `(1,3)` —
**107.4%** retention (i.e. above baseline) at 16.0% params saved. Worst:
vit-large, imagenet-1k, span `(6,21)` — 22.1% retention at 62.2% params saved.
Best compression at ≥95% retention: vit-large/pneumoniamnist span `(6,23)`,
99.4% retention at **70.5%** params saved.

**Why.** Late-network spans and the easier task (pneumoniamnist) tolerate large
cuts; mid-network spans on the deepest model (vit-large) on the hardest task
(imagenet-1k) are the failure mode — consistent with the "later blocks are more
redundant" pattern seen everywhere else in the repo (§5).

**Conclusion.** This is the cleanest "TOAST works, here's the tradeoff curve"
result in the whole repo — strong candidate for the headline chart.
**Flag:** the figures this feeds (`layer_approximation_impact.png`,
`accuracy_vs_params.png`, `accuracy_loss_distribution.png`) are only
reproducible by re-running `thesis_analysis.ipynb` — no saved cell outputs.

---

## 4. Alternating window sweep (multi-span)

**Purpose.** Does splitting a skip budget into several smaller, non-contiguous
spans beat one big contiguous cut?

**Method.** `results_alternating_window_grid_combined.csv` →
`linear_approximations_alternating_sorted.csv` (237 rows), 2–9 spans per
config, same models/datasets as #3.

**Results.** Mean retention 95.4% (std 6.5) vs. 88.6% (std 17.7) for single-span
— tighter and higher. Degrades smoothly with span count: 96.9% at 2 spans,
89.0% at 9. Best: rad-dino `[(0,1),(2,3)]`, 105.4% retention, 15.3% params
saved. Worst: vit-base/imagenet-1k 3-span config, 65.2% retention.

**Why.** Spreading the same total cut across smaller spans avoids the sharp
mid-network failure mode seen in #3 — smaller local perturbations are each
easier for a linear map to bridge, even if the total removed is the same.

**Conclusion.** Real, useful result: "many small skips" is a strictly safer
compression strategy than "one big skip" at matched budget.
**Flag — needs attention before the meeting:** `thesis_analysis_alternating.ipynb`
is **corrupted** (malformed JSON, nested `cell_type` keys from cell 5 on) and
won't currently open in Jupyter. The CSV numbers above are recoverable and
solid; the notebook narrative/plots need hand-repair or a re-run first.

---

## 5. Block distance, CKA, and when a translator is needed

*(Full write-up already exists: `docs/cka_skip_distance_findings.md`.)*

**Purpose.** The pipeline currently auto-picks `identity` vs `linear` by
thresholding CKA between the two endpoints of a skip span. Is that rule
actually correct?

**Method.** 670-run position × span sweep on dinov2-base/imagenet-1k, every
span run with both bridges.

**Results.** The rule is **0 for 3** everywhere it fires (spans where CKA ≥
0.90): mean accuracy lost by trusting identity there is **20.2 points**. CKA
correlates at −0.146 with identity's damage (no signal) but −0.765 with
linear's damage (strong signal). Distance is the strongest predictor of
unbridged damage (+0.739); depth is negatively correlated (later blocks
cheaper to drop).

**Why.** CKA is invariant to invertible linear maps — high CKA says "these
spaces are related by *some* linear map," not "they're equal." That's exactly
the question `linear` answers and `identity` doesn't. The rule is asking CKA
something it structurally cannot answer.

**Conclusion.** Genuine bug in current logic, not a tuning issue — no threshold
fixes it. Recommended fix already written up: **choose spans by distance/depth,
choose the translator by CKA**, and drop or invert the identity branch in
`recommend_runs.py`. This is a concrete, actionable fix to walk in with.

---

## 6. Skip candidate predictor

*(Full write-up already exists: `docs/skip_candidate_predictor_findings.md`.)*

**Purpose.** Two questions: (a) can you predict the damage from skipping an
arbitrary combination of blocks *without running it*? (b) does CKA structure
— and skip-safety — transfer across datasets?

**Results.**
- **Disjoint pairs** (two separate single-block skips): near-additive damage,
  a 3-feature regression predicts it almost exactly out-of-sample (LOO-CV
  R² = 0.951–0.998 across 3 model/dataset combos), and beats random selection
  significantly (p = 0.022–0.043 on 2 of 3).
- **Contiguous 2-block spans**: don't compose additively; a plain regression is
  a mixed bag (8/24 fits good, 10/24 negative). Screening out spans that cross
  a CKA "cliff" before fitting rescues it dramatically on 2 of 4 datasets
  (cifar100: −0.562 → **+0.938**; dermamnist: −0.401 → **+0.924**), but makes
  small-n fits worse, so it's a selective tool, not a default.
- **Cross-dataset**: CKA's internal structure is dataset-invariant (correlation
  0.946–1.000 across every model/dataset pair). Skip-*safety* is not: of 36
  model/dataset-pair correlations, 9 are statistically significant (permutation
  test, 20000 trials) — 5 positive, 4 negative, all 4 negative ones on deit
  models with cifar100 involved. vit-large shows no cross-dataset consistency
  either way, at its highest statistical power.

**Why.** The disjoint case works because two independent single-block removals
really do act independently on this architecture. The contiguous case doesn't,
because a long-enough span can hit a genuine representational
floor/saturation, not just "more of the same damage." The cross-dataset split
suggests CKA is measuring something architectural (stable), while what's
*safe to remove* also depends on what the task actually needs (not stable).

**Why.** — **Conclusion.** Positive, quantified answer to "can we predict skip
damage" for the disjoint case; a usable-but-conditional recipe for the
contiguous case; and a real, non-obvious nuance for the cross-dataset claim
(CKA transfers, safety doesn't — worth stating explicitly rather than treating
as one finding). **Known gap, already flagged in the doc:** both headline
results (disjoint-pair regression, the well-powered null test) only exist
because imagenet-1k happens to have disjoint-pair runs and vit-large happens
to have 24 layers — replicating on the other 3 datasets is the natural next
step, not yet done.

---

## 7. Redundant attention heads / threshold dropping

**Purpose.** Instead of skipping a whole block, can individual attention
*heads* within a layer be identified as redundant (by pairwise similarity) and
dropped independently?

**Method.** `src/heads/head_priority.py`: pairwise Jensen–Shannon divergence
between every head's attention distribution within a layer, converted to a
similarity score; a threshold sweep (0.05–0.95) greedily drops a head if it's
similar enough to one already kept. Only one model/dataset was actually run:
deit-small on CIFAR-100, 72 heads total (12 layers × 6 heads), 3 seeds.

**Results.** Baseline 71.25%. Clear knee: up to 8/72 heads (11%) droppable for
<2pp loss. Past ~20/72 (28%), damage steepens sharply (10+ points), and by
40–50/72 dropped it plateaus around 37–48% rather than collapsing to
chance (~1% for 100 classes) — some residual function survives even
aggressive head removal.

**Why.** Consistent with the "late layers are redundant" pattern seen
throughout — head redundancy is a finer-grained version of block redundancy,
and there's a real, exploitable free region before the knee.

**Conclusion.** The head-level idea works and has a genuine positive result at
small scale — but it's **early, not generalized**: one model, one dataset,
where the block-skip side of the project already spans 3–5 models and multiple
datasets. **Flags to resolve before presenting this:**
- `heads_dropping_analysis.ipynb` has a syntax error in its plotting cell (an
  unclosed paren) — it doesn't currently execute as committed.
- That same cell uses a per-head parameter-count constant (98,496) that
  **contradicts** the CSV's own `parameters_saved` column (148,224/head) —
  unresolved, pick one and fix the other.
- A related but separate line of work — `results_full_table1/2/3.csv` and
  `accuracies_advanced.csv` — despite living in the same folder, is actually
  whole-block/MLP/attention-*layer* skipping (all `head_dict` empty), not
  head-level. Table1→2→3 is a real progression (deit-small/imagenet-1k only →
  + dinov2-base/vit-large → + cifar100). The "advanced" file tests combining
  skip types together and shows they don't compose safely: one deit-small
  combo nudges *above* baseline (71.39% vs 71.25%), but a dinov2-base
  block+attention combo drops from 87.2% to 65.96% — much worse than either
  cut alone. Worth its own line in the story: **combined approximations don't
  compose predictably**, which is exactly what motivated §6's composability
  work.
- `accuracies_table3_medical_completed.csv`: despite the filename, deit-small
  has only 63/78 rows (21 of 26 configs) vs. dinov2-base and rad-dino's full
  78 — not actually complete. Medical-result variance is also much higher
  (mean std 1.2pp, max 5.7pp) than the CIFAR/ImageNet tables (~0.2–0.3pp) —
  treat these numbers as noisier.
- `attention_heads/` (16 PDFs of per-head CKA heatmaps, vit/dinov2 ×
  CIFAR10/100/FashionMNIST/MNIST) is exploratory visualization using a
  *different* similarity metric (CKA, not JSD) and isn't wired into the
  accuracy pipeline — good supporting visuals, not itself a result.

---

## 8. Low-rank translator

**Purpose.** §5 shows the linear translator does "enormous work" recovering
accuracy. This asks how much of that 768×768 matrix is actually needed — can a
low-rank version match it at a fraction of the parameters?

**Method.** Three estimators (`lowrank_translator.py`): `lowrank_r` (SVD-
truncate the already-fit full matrix), `rrr_r` (reduced-rank regression, the
provably-optimal rank-r fit), `lora_r` (same objective via Adam, as a sanity
check on `rrr`). Ranks 8–256, dinov2-base/imagenet-1k, 3 spans chosen by how
much the linear translator gains over identity: `(5,6)` gain 0.437, `(2,4)`
gain 0.590 (hardest), `(0,1)` gain 0.084 (meant as a "rank shouldn't matter"
control).

**Results.** `rrr` beats `lowrank` at every matched rank, sometimes hugely
(span `(5,6)`, rank 256: 0.718 vs 0.607) — truncating post-hoc wastes rank on
directions the data doesn't need. `rrr_256` (393k params, a 33% saving vs.
full) recovers 98% of the linear translator's accuracy gain on the two hard
spans, but the knee is sharp, not gradual: only 52% recovered at rank 64, 87%
at rank 128. **Span `(0,1)`, the control, inverted the hypothesis**: `rrr_64`
(0.316) and `rrr_128` (0.618) both score *below* doing nothing at all
(identity, 0.654) — an underpowered low-rank map actively hurts on a span
where the full linear map barely helps.

**Why.** No sharp low-effective-rank structure exists here to exploit — the
translator needs most of its rank, not a small subspace of it. The `(0,1)`
result suggests a partial-rank map can actively distort a representation that
was already fine, rather than just failing to help.

**Conclusion.** Answers the motivating question, and the answer is a genuine
negative worth stating plainly: **big compression via low rank isn't there**
for this model/dataset — the translator is close to full-rank necessary.
**Flags:** single model/dataset pair, first-pass only; 2 of 8 configured ranks
(4, 384) were silently never run (a config-generator/registry mismatch, not
missing data — worth fixing before trusting the sweep is complete); medical
spans are stubbed but not filled in.

---

## 9. MedMNIST medical applications

**Purpose.** Does the block-skip/translator/MLP-linearization approach hold up
on medical imaging, and does a domain-specific medical encoder (rad-dino)
behave differently from general-purpose ones under the same treatment?

**Method.** `med_applications/runs/dermamnist.csv`, 4 encoders (deit-small,
dinov2-base, vit-large, rad-dino) on DermaMNIST, seeds 1–3, varying skip spans,
translator, MLP/attention linearization.

**Results.** vit-large: 0.7466–0.7746 (best config uses aggressive MLP
linearization on late layers, not the plain baseline). dinov2-base:
0.6906–0.7433, widest spread and highest variance (std up to 0.057 on
aggressive early double-skips — unstable there). deit-small: 0.7145–0.7287,
tight and stable. **rad-dino: 0.6964–0.7169 — the lowest ceiling of all four,
despite being the domain-specific medical model.**

**Why.** rad-dino's low ceiling looks like a classifier-head/probe-fit issue
rather than a skip-config issue — all its configs cluster tightly near 0.70
regardless of what's skipped, unlike the other models where config choice
visibly moves the number. vit-large's larger capacity gives MLP linearization
more room to actually help.

**Conclusion.** A real, complete (if small — one dataset) grid. The
counterintuitive result — **domain-specific pretraining doesn't automatically
win here** — is worth a line in the story, but should be framed as "the probe
setup may be underselling rad-dino" rather than "medical pretraining doesn't
help," since nothing here isolates that. `med.ipynb` has zero markdown
commentary, so this interpretation is mine from the data, not inherited from
prior analysis — worth double-checking before presenting as settled.

---

## 10. Cross-model / cross-dataset transfer

**Purpose.** If a skip/translator config is fit (calibrated) on one dataset,
does it transfer to a different target dataset without refitting? Rad-dino
tests whether a medical-only encoder transfers differently than general ones.

**Method.** Two separate sweep generations exist on disk (see flag below).
The newer, workflow-documented one: 70 "pattern" files
(`results/results/transfer_learning/results_transfer_learning_pattern_*.csv`),
6 encoders, targets {pneumoniamnist, cifar100, dermamnist, imagenet-1k},
sources = those four plus chestmnist, 250 samples/eval, seeds 1–3. **2739 raw
rows on disk** (913 unique configs × up to 3 seeds) — the workflow doc's
figure of "1,092 experiment results" doesn't match what's actually there;
worth reconciling which number is current before it goes in front of anyone.

**Results** (mean accuracy, in-domain-fit vs. best cross-domain fit):
- pneumoniamnist: in-domain 0.8325 vs. chestmnist-fit 0.8254 (Δ 0.7pt) — nearly
  free transfer between two medical binary tasks.
- dermamnist: in-domain 0.7588 vs. chestmnist-fit 0.7528 (Δ 0.6pt) — small
  loss.
- cifar100: in-domain 0.8055 vs. best medical-fit 0.7571 (Δ ~5pt) — moderate
  loss, medical → natural images.
- imagenet-1k: in-domain 0.6873 vs. best medical-fit 0.6012 (Δ ~9–10pt) —
  largest loss, medical-fit configs don't generalize to the much harder
  1000-way task.
- rad-dino baseline accuracy is domain-locked: 0.888 (pneumoniamnist) / 0.701
  (dermamnist) but only 0.290 (cifar100) / 0.075 (imagenet-1k) — confirms it's
  unsuitable as a general-purpose transfer source, as expected.

**Why.** Transfer loss tracks target-task difficulty/domain distance, not just
medical-vs-not: medical→medical is nearly free, medical→natural-image
degrades progressively with how hard the natural-image task is. This is a
clean, reportable pattern.

**Conclusion — the pipeline here is genuinely incomplete, not just
"unfinished analysis":**
- `source_rankings.csv` and `transfer_losses.csv` (workflow step 3's output)
  **don't exist anywhere in the repo** — never actually produced.
- `analyze_transfer_results.py`, which would produce them, expects columns
  `encoder`, `skip`, `test_accuracy`, `fit_dataset` — neither sweep generation
  actually has those column names (`model`, `approx_layer`, `accuracy` instead)
  — so as written, running it against the real data would silently produce
  nothing. The numbers above were computed directly from the raw pattern files
  for this report, not from the pipeline's own analysis step.
- `combine_transfer_learning_results.py`'s filename regex only matches the
  *older* `_skipNN.csv` sweep, not the newer `_pattern_patternN.csv` files —
  the two sweep generations were never merged, and `transfer_learning_experiments/*.csv`
  reflects only the older, superseded run.
- chestmnist is used only as a *source* in the newer sweep, never as an
  evaluation *target* — so the workflow doc's stated goal ("rank which source
  transfers best to chestmnist") can't actually be answered from the data
  that exists.

This is worth raising directly: the transfer-learning results that do exist
are good and worth keeping, but the analysis pipeline supporting them needs a
column-mapping fix and a re-merge before its own headline claim can be
produced from real data.

---

## 11. Token-mixing translator (design only — not implemented)

**Purpose.** The linear translator (§5, §8) flattens every token into one pile
and fits a single per-token matrix — token 5 is transformed without ever
seeing token 6. For an MLP sub-block that's the right shape of tool (MLPs are
per-token already); for self-attention it structurally cannot represent what
attention does, since moving information between tokens is the entire
operation. This is a structural argument, not yet an empirical one.

**Status.** Nothing built yet. `docs/token_mixing_design.md` is a design note:
proposes a dense N×N mixer (MLP-Mixer style, cheap, ~66k params) or a
depthwise conv (fewer params, resolution-flexible), scopes the CLS-token
handling problem, and flags that the one existing scaffold
(`conv_translator.py`) is dimensionally broken and unregistered — a rewrite,
not a starting point. Recommends starting with the dense mixer as the whole-
block bridge (cheapest test, ~1 day) before touching the harder isolated-
attention-sublayer path (~3 days, requires threading a translator parameter
into `AttentionLinearisedEncoder`, which currently bypasses the translator
registry entirely).

**Conclusion.** Not a result — a scoped next step with a clear, cheap first
experiment already designed. Worth naming as "here's the next thing," not
folding into the "what did we find" section.

---

## Data-quality issues worth deciding on before the meeting

These don't change the science, but they'll come up if anyone asks to
reproduce a number:

1. `dedup_results_pipeline.csv` — all `linear` rows are zeroed/broken (§1).
2. `thesis_analysis_alternating.ipynb` — corrupted JSON, won't open (§4).
3. `heads_dropping_analysis.ipynb` — plotting cell has a syntax error, and
   uses a parameter-count constant that contradicts the CSV it plots (§7).
4. `accuracies_table3_medical_completed.csv` — deit-small is 5 configs short
   despite the filename (§7).
5. Low-rank sweep — 2 of 8 configured ranks silently never ran due to a
   config/registry mismatch (§8).
6. Transfer-learning analysis script's column names don't match either sweep's
   actual CSV schema, and the two sweep generations were never merged — the
   workflow's own headline output was never produced (§10).

None of these are hard to fix, but each is a "wait, where did that number come
from" risk if raised cold in a meeting.

---

## Possible stories for tomorrow

Three defensible framings, not mutually exclusive:

- **"We found and fixed a real bug in the core method."** §5 (CKA rule is
  backwards) is the tightest, most surprising, most actionable single result
  in the repo — a falsifiable claim, tested, wrong, with a concrete fix
  proposed. Good if the meeting wants one crisp result to anchor on.

- **"We built a training-free damage predictor."** §6's disjoint-pair
  regression (R² up to 0.998, beats random selection significantly) plus §3/§4
  (the actual compression-vs-accuracy tradeoff curves, including the "many
  small skips beat one big skip" finding) is the most complete positive
  story, with a named, honest gap (needs replication beyond imagenet-1k /
  vit-large) rather than an overclaim.

- **"We mapped where the method does and doesn't generalize."** Head-level
  redundancy (§7) works but is one-model deep; low-rank compression (§8) is a
  real negative result; medical transfer (§9, §10) shows domain-specific
  pretraining isn't a free win and cross-dataset skip-safety isn't stable
  (§6). This is the honest "here's the map, here's what's solid vs.
  preliminary" framing — probably closest to what a progress meeting wants,
  and sets up §11 (token-mixing) as a well-motivated next step rather than a
  loose end.

My inclination is the third, with the CKA-rule fix (§5) pulled out as the
opening hook — it's concrete enough to open with and everything else slots in
as "here's what else we now know."
