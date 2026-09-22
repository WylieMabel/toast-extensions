# Chapter 5 — predicting multi-skip collapse from single skips

Core question: can we predict the accuracy damage of skipping multiple blocks from
the damage of skipping each block individually?

Three predictors of multi-skip damage, all built from single-block drops, each
scored by **Pearson r** (does it track the magnitude of damage?) and **Spearman
rank r** (does it correctly rank skips from safest to most damaging?) against the
actual measured multi-skip damage:

- **naive** — sum of single-block drops, no fitting: `predicted = drop(a) + drop(b)`
- **cka** — naive sum + a learned CKA-similarity correction term, fit per (model, dataset)
- **ang_dist** — naive sum + a learned angular-distance correction term, fit per (model, dataset)

Full per-(model, dataset) numbers for all three predictors, both shapes:
`predictor_comparison_naive_cka_angdist.csv`. Source files: `naive_sumdrop_baseline.csv`
(naive), `skip_regression_coefs.csv` / `skip_regression_predictions.csv` (cka),
`skip_regression_coefs_angdist.csv` / `skip_regression_predictions_angdist.csv` (ang_dist).

## Continuous (contiguous 2-block spans, e.g. `[(0,2)]`)

Median across all (model, dataset) combos:

| predictor | median pearson r | median spearman r |
|---|---|---|
| naive | 0.802 | 0.812 |
| cka | 0.595 | 0.529 |
| ang_dist | 0.671 | 0.571 |

Naive summation is the strongest predictor on median, not the correction terms.
The regression models are pulled down by a few models where the fitted correction
actively hurts LOO-CV prediction — dinov2-base and vit-base go strongly negative on
dermamnist/pneumoniamnist (e.g. dinov2-base+pneumoniamnist: naive n/a, cka pearson
−0.74, ang_dist pearson −0.80) — the correction term overfits on small, noisy
single-model spans rather than adding real signal. deit-small and deit-base are the
exception: correction terms match or slightly beat naive there. So contiguous
additivity is architecture-dependent, and "fit a correction term" is not a free win.

## Disjoint (two separate single-block skips, e.g. `[(0,1),(2,3)]`)

Same three predictors, restricted to the 3 combos we have (all imagenet-1k:
deit-small, dinov2-base, vit-large):

| predictor | median pearson r | median spearman r |
|---|---|---|
| naive | 0.989 | 0.944 |
| cka | 0.987 | 0.950 |
| ang_dist | 0.987 | 0.952 |

All three predictors are essentially equivalent and all strong (~0.95-0.99).
Disjoint damage is close to additive on its own; the correction terms neither help
nor hurt meaningfully because there's almost no residual signal left to capture.

## Stratify by distance between the two skips — `disjoint_additivity_by_gap.csv`

Bin the disjoint pairs by gap (near/mid/far) and re-check pearson r / R² per bin,
for the same 3 imagenet-1k combos. Additivity gets more exact as the gap widens for
all three models (e.g. vit-large: R² 0.996 → 0.998 → 1.000) — the farther apart the
two skipped blocks are, the less they interact, and the closer naive summation gets
to exact.

## Story for the chapter

1. Contiguous spans: naive summation is actually the more robust predictor overall;
   a learned cka/ang_dist correction term helps for some backbones (deit family)
   but overfits and hurts for others (dinov2-base, vit-base) — no universal win.
2. Disjoint pairs: naive summation already ranks and predicts skips well; cka and
   ang_dist correction terms are statistically indistinguishable from it.
3. Stratifying disjoint pairs by gap shows why disjoint is closer to additive than
   contiguous in the first place — additivity strengthens monotonically with
   distance, i.e. interaction between skipped blocks decays with separation.
