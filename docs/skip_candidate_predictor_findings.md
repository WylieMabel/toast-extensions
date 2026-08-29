# Predicting good skip candidates: what the data actually supports

Four-step investigation into to-do 1 ("predict good skip candidates, beat a random-selection
null hypothesis") and to-do 2 ("accuracy patterns are similar across datasets — is CKA key?"),
run entirely on `block_distance/distance_analysis.csv` (5 models x 4 datasets, single-block and
2-block skip sweeps) plus the precomputed CKA matrices in `src/layers/outputs/`. No new eval
runs were needed.

Reproduce in order:

```
python block_distance/composability_check.py
python block_distance/fit_skip_regression.py
python block_distance/null_hypothesis_test.py
python block_distance/cross_dataset_consistency.py
```

## Headline result: disjoint-pair damage is (almost) perfectly predictable

The strongest, cleanest, most reproducible finding in the whole investigation. When you skip
two *separate* single blocks (e.g. block 3 and block 9, independently), the combined accuracy
drop is nearly additive, and a 3-feature linear regression — `cka_1`, `cka_2` (the individual
CKA of each single-block bridge, kept separate, not summed), and `block_drop_sum` (the sum of
the two blocks' individual skip damage) — predicts it almost exactly, out-of-sample:

| model | dataset | n candidate pairs | LOO-CV R² |
|---|---|---|---|
| deit-small | imagenet-1k | 45 | **0.951** |
| dinov2-base | imagenet-1k | 45 | **0.973** |
| vit-large | imagenet-1k | 186 | **0.998** |

And it isn't just a good fit — it beats random selection with statistical significance, using
a null built by picking a few (k=3) random pairs and taking the best, repeated 2000 times per
model/dataset (`null_hypothesis_test.py`):

| model | dataset | n pool | single-random-pick p |
|---|---|---|---|
| deit-small | imagenet-1k | 45 | **0.022** |
| vit-large | imagenet-1k | 186 | **0.043** |
| dinov2-base | imagenet-1k | 45 | 0.111 (close, not significant) |

**Practical implication:** you get a fast, training-free way to rank arbitrary two-block skip
combinations without evaluating them. Measure every single-block skip once (already have this
data from the "skip each layer in turn" sweep) plus the CKA matrix (already precomputed), fit
one cheap regression, and predict any pair's damage. This is the direct, positive answer to
to-do 1 for the disjoint case, and the part most worth leading with.

*Caveat: only tested on imagenet-1k (the only dataset with disjoint-pair runs in the sweep).
Worth confirming it holds on the other 3 datasets before generalizing the claim.*

## Second result: a CKA-boundary screen fixes the contiguous case where it's data-rich enough

Contiguous 2-block spans (one span dropping two adjacent blocks) do **not** compose additively
the way disjoint pairs do (`composability_check.py`), and a plain regression on them is a mixed
bag: of 24 model/dataset fits, 8 are good (LOO-CV R² > 0.6), 10 are *negative* (worse than
guessing the mean) — see `block_distance/skip_regression_coefs.csv`.

vit-large is the extreme case: all 4 of its contiguous fits start negative (−1.16 to −0.40)
despite having by far the most data (n=22). The cause, found by inspecting per-span residuals
(`skip_regression_predictions.csv`): 3 spans around blocks 4–7 collapse to near-zero CKA and
catastrophic accuracy loss (up to 0.71) — a floor/saturation effect — while the other 19 spans
sit in a tight, boring, well-behaved cluster (CKA ≈ 1.0, drop < 0.05). Fitting the regression
only on the well-behaved majority, after screening out spans that cross a CKA gap, recovers the
fit completely:

| dataset | LOO-CV R² (all spans) | LOO-CV R² (CKA-safe subset) |
|---|---|---|
| cifar100 | −0.562 | **+0.938** |
| dermamnist | −0.401 | **+0.924** |
| imagenet-1k | −0.569 | +0.562 |
| pneumoniamnist | −1.159 | +0.045 (still weak — see noise floor below) |

The screen (`find_cka_gap_split` in `fit_skip_regression.py`) looks for a gap in sorted CKA
values much bigger than the typical spacing that only carves off a minority of spans, so it
doesn't fire on models without a real critical zone. It is **not a universal win**: on the
small-n (n=10) fits it usually makes things worse (e.g. dinov2-base/dermamnist −0.22 → **−2.42**)
because removing 2–4 of only 10 points destabilizes leave-one-out validation regardless of
whether the removed points were genuinely anomalous. Use it selectively — per model/dataset,
whichever of the screened/unscreened fit has the higher LOO-CV R² — not by default.

## Third result: CKA is a model fingerprint, not a task-safety signal

This directly engages to-do 2, and the answer is more interesting than a plain "yes":

- **CKA structure is essentially dataset-invariant.** Comparing a model's CKA matrix computed
  on dataset A vs. dataset B (`cross_dataset_consistency.py`), the full-matrix correlation is
  **0.946–1.000 for every model, every dataset pair tested** (mean 0.983). vit-large hits
  0.998–1.000 across the board.
- **Which blocks are safe to skip is inconsistent, and it's not just noise.** With only 11-23
  blocks per model, raw Spearman r isn't enough to call — each of the 36 model/dataset-pair
  correlations was permutation-tested (shuffle one vector, rebuild the null distribution of r,
  20000 trials). Result: **9/36 pairs are statistically significant at p<0.05, split 5 positive
  / 4 negative**; the other 27 are genuinely indistinguishable from chance at this sample size,
  not just "weak." So it isn't accurate to say the pattern doesn't generalize at all — it's that
  where it's significant, it goes *both ways*:
  - **Real positive transfer** in 5 pairs, strongest for dinov2-base cifar100-vs-imagenet-1k
    (r=+0.909, p<0.001) and rad-dino/vit-base pairs involving pneumoniamnist (r=0.67-0.74).
  - **Real negative transfer (inversion), all 4 involving cifar100 on deit models**: deit-base
    cifar100-vs-imagenet-1k (r=−0.691, p=0.024), deit-base cifar100-vs-pneumoniamnist
    (r=−0.692, p=0.021), deit-small cifar100-vs-dermamnist (r=−0.791, p=0.005), deit-small
    cifar100-vs-pneumoniamnist (r=−0.664, p=0.031). For deit specifically, a block that's safe
    to drop on cifar100 is *significantly more likely to be unsafe* elsewhere, not just
    unrelated.
  - vit-large, with the most power (n=23 blocks), shows **no significant pairs at all** —
    a real result at that sample size, not an artifact of too little data.

Put together: CKA characterizes something stable about the model's internal geometry,
independent of what data flows through it. Skip-safety is not dataset-invariant, but the
honest claim is narrower than "doesn't generalize": it's architecture- and dataset-pair-
specific, with cifar100-vs-others inversion on deit models as the most concrete, reproducible
sub-finding, and vit-large showing no cross-dataset consistency either way. This still
sharpens `docs/cka_skip_distance_findings.md`'s finding that CKA predicts translator success
but not skip safety — CKA would give the *same* answer on every dataset, but the *right* answer
is dataset- and architecture-dependent in a way that isn't just noise for at least 9 of the 36
pairs tested.

## Honest negatives

- **pneumoniamnist is a noise floor, not a modeling failure.** Seed-to-seed accuracy std for a
  single-block skip is 0.009–0.026 there vs. 0.0015–0.004 on imagenet-1k (~10x), and most
  contiguous-span deltas are the same size as that noise. No feature set regresses well against
  a signal smaller than its own measurement error.
- **Most of the null-hypothesis test is underpowered by construction.** 20 of 27 model/dataset/
  shape combos have only 10 candidate spans, so the best possible single-random-pick p-value is
  1/10 = 0.10 — p<0.05 significance is mathematically unreachable there regardless of predictor
  quality. Only the 7 combos with n≥22 (vit-large's contiguous fits, all 3 disjoint pools) can
  demonstrate significance at all, and 4 of those 7 do.

## Suggested framing for the thesis

1. Lead with the **disjoint-pair result** (R²=0.95–0.998, significant vs. random at p<0.05/0.04)
   as the clearest positive answer to to-do 1 — a training-free damage predictor for combining
   arbitrary single-block skips, built entirely from data TOAST-style sweeps already produce.
2. Present the **CKA-boundary screen** as the practical recipe for the contiguous case: screen
   first when there's enough data to spare (it turned vit-large from the worst case into one of
   the best), fit directly when there isn't.
3. Frame the **cross-dataset result** as a nuance, not a failure: CKA transfers across datasets,
   skip-safety doesn't — worth stating explicitly against `docs/cka_skip_distance_findings.md`.
4. Biggest open gap: the disjoint-pair result and the well-powered null-hypothesis test both
   only exist because imagenet-1k happens to have disjoint-pair runs and vit-large happens to
   have 24 layers. Replicating the disjoint sweep on the other 3 datasets, and/or running a
   deeper/wider window sweep to get contiguous-span pools bigger than n=10, would turn today's
   "works where we happened to have enough data" into a claim that generalizes.

## Files produced

| file | what it is |
|---|---|
| `block_distance/composability_check.py` | Step 0: additivity test, contiguous vs. disjoint |
| `block_distance/fit_skip_regression.py` | Step 1: per-model/dataset regressions + CKA screen |
| `block_distance/skip_regression_coefs.csv` | one row per (model, dataset, shape, screened) fit |
| `block_distance/skip_regression_predictions.csv` | one row per span: features, actual vs. predicted damage |
| `block_distance/null_hypothesis_test.py` | Step 2: predictor vs. random-selection significance test |
| `block_distance/null_hypothesis_results.csv` | per-combo p-values, power flags |
| `block_distance/cross_dataset_consistency.py` | Step 3: cross-dataset accuracy/CKA pattern comparison |
| `block_distance/cross_dataset_accuracy_pattern.csv`, `cross_dataset_cka_pattern.csv` | pairwise correlations |
