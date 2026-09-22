# Metric Experiments — Summary

**Question:** given a candidate layer-skip span in a ViT-family model, which cheap metric (computed without a full retrain/eval loop) best predicts whether that skip is "safe" (accuracy within 1% of the best span found)?

**Setup:** 4 backbones (DeiT-S, ViT-S, ViT-L, DINOv2-B) × 4 datasets (CIFAR-10, CIFAR-100, PathMNIST, ImageNet-1k) = 15 model/dataset combos, 51–66 contiguous skip spans each. "Safe" = accuracy gap to the top span ≤ 1% of the top span's mean (effect-size threshold, not a significance test, n=3 seeds).

## Metric families tested
- **Local** (only at the skip boundary): `local_cka`, `angular_dist`
- **Global/headless** (final embedding, no task head needed): `cosine_sim`, `mse`, `cka_embed`, `knn_overlap@5/10`, `participation_ratio_change`
- **Task/frozen-head** (needs a matching ImageNet head — only 3/15 combos: DeiT-S, ViT-S, ViT-L on ImageNet-1k): `kl_div`, `top1_flip_pct`, `entropy_delta`, `orig_head_accuracy`

## Headline result: `orig_head_accuracy` wins by a wide margin
Feeding the skip-modified representation through the **frozen original head** and checking if it still predicts correctly: mean AUC **0.998** for the safe/not-safe split, rho **0.97–0.98** vs true accuracy, significant in all 3 combos it could be computed on. This is the strongest signal by far — but it only applies where the checkpoint's original head matches the eval task (3/15 combos), so it's not a general-purpose metric, just a near-perfect one when available.

## Best generally-applicable metrics (all 15 combos)
Ranked by mean AUC for predicting "safe" (from `safe_prediction_auc_summary.csv`):

| metric | mean AUC | significant combos |
|---|---|---|
| angular_dist (local) | 0.837 | 12/15 |
| local_cka (local) | 0.812 | 12/15 |
| mse (global) | 0.794 | 9/15 |
| cka_embed (global) | 0.767 | 11/15 |
| cosine_sim (global) | 0.760 | 10/15 |
| participation_ratio_change (global) | 0.754 | 10/15 |
| knn_overlap@5 (global) | 0.743 | 8/15 |
| knn_overlap@10 (global) | 0.737 | 9/15 |

**The two local, boundary-only metrics beat every global/headless metric**, consistently, across backbones. `angular_dist` is also the strongest and most stable Spearman correlate of true accuracy (|rho| up to 0.97 on CIFAR-10/100 combos), whereas global metrics are noisier and occasionally flip sign or lose significance (e.g. `cosine_sim`/`mse` non-significant on `dinov2base_pathmnist`).

## Most interesting/close-to-novel finding
**DINOv2-B breaks the global metrics but not the local ones.** On `dinov2base_cifar10`, `cosine_sim`/`cka_embed`/`knn_overlap` drop to near-chance (rho 0.13–0.26, AUC 0.66–0.70), while `local_cka` and `angular_dist` stay strong (rho 0.81/0.90, AUC 0.93/0.97). Interpretation: DINOv2's final embedding is robust/redundant enough that a locally-disruptive skip doesn't show up in global embedding similarity — global "does the output still look the same" metrics are the wrong lens for skip-safety on this backbone, but a boundary-local probe still catches it. That's a fairly concrete, checkable claim (not just "DINOv2 is different") and worth a sentence in the thesis if it holds up under a second look.

## Weakest spot: PathMNIST
Every metric's rho drops sharply on PathMNIST relative to natural-image datasets (e.g. deitsmall: cosine_sim 0.88→0.30, angular_dist -0.97→-0.28). Consistent across all 4 backbones, so likely a property of the task (simpler decision boundary, more redundant features) rather than noise — but worth flagging as "these metrics don't generalize to easy/low-diversity tasks" rather than claiming a universal ranking.

## Cross-dataset transfer of "which spans are safe"
Spearman correlation of `true_accuracy` across dataset pairs, same backbone (`cross_dataset_true_accuracy_correlations.csv`):
- Within natural-image datasets: strong and significant everywhere (cifar10 vs cifar100: rho 0.90–0.99; imagenet vs cifar: rho 0.71–0.98).
- **PathMNIST vs everything else is a cliff** for DeiT-S/ViT-S (rho 0.03–0.24, not significant) — good skip spans for natural images don't transfer to the medical-image task on these two backbones.
- DINOv2-B and ViT-L are the exception: PathMNIST still correlates significantly with the others (rho 0.32–0.80). Consistent with the metric-robustness finding above — DINOv2 (and to a lesser extent ViT-L) may just generalize its notion of a "safe" skip better across task domains.

## Caveats before treating any of this as a result
- `orig_head_accuracy`/`kl_div`/`entropy_delta` numbers rest on only 3 combos — striking but not yet a trend.
- "Safe" label depends on an arbitrary 1%-of-top-mean tolerance and n=3 seeds; worth a sensitivity check at 0.5%/2% before leaning on exact AUC values.
- Many p<0.05 claims above with no multiple-comparison correction (15 combos × 8–12 metrics) — treat the *ranking* of metrics as the finding, not any individual p-value.
- The DINOv2 local-vs-global gap is currently a 3-dataset observation (cifar10/100/pathmnist, no imagenet head) — real, but a single backbone's worth of evidence for what's framed above as a general claim about DINOv2.
