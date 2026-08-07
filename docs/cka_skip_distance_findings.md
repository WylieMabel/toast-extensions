# Skip distance, CKA, and when a translator is needed

Findings from `block_distance/results_imgnet_dinov2.csv` — a systematic position × span sweep
on DINOv2-base / ImageNet-1k, 670 runs, each span evaluated with both an `identity` bridge and
a `linear` translator. No-skip baseline: **0.7431**.

Reproduce with `python block_distance/validate_cka_rule.py`.

## Headline: the CKA rule in the pipeline is backwards

`recommend_runs.py:92-94` currently auto-selects a translator like this:

```python
return "identity" if cka[skip_from + 1, skip_to + 1] >= CKA_IDENTITY else "linear"
```

That is a falsifiable prediction, and the sweep contains the runs to test it. It does not hold.

| | |
|---|---|
| Spans where identity is safe (within 2pp of baseline) | **0 of 21** |
| Times the rule recommended identity | 3 |
| Times that was correct | **0 of 3** |
| Mean accuracy lost by following it | **20.2 points** |

The three spans it fired on were `(8,9)`, `(8,10)`, `(9,10)` — CKA 0.990–0.997, essentially
identical representations by that measure — and dropping the translator there cost 13.7, 26.6
and 20.3 points respectively. The result is not sensitive to the tolerance: the rule is 0-for-3
at every threshold from 0.5% to 5%.

## Why — CKA is measuring the wrong equivalence

This is not a miscalibrated threshold. There is no threshold that fixes it, because identity is
never safe on this model. The rule asks CKA a question CKA cannot answer.

CKA is *invariant to invertible linear transformations*. That invariance is the whole reason it
is a good tool for comparing representations. But it means a high CKA says:

> these two spaces are related by a linear map

whereas dropping the translator asks:

> are these two spaces **equal**

Those are different claims, and CKA cannot distinguish them. The correlations say exactly this:

| | correlation with CKA | |
|---|---|---|
| Accuracy lost using **identity** | **−0.146** | what the rule assumes CKA predicts |
| Accuracy lost using **linear** | **−0.765** | what CKA actually predicts |

Controlling for span length (distance-1 spans only, n=11) sharpens it further: **+0.121** for
identity versus **−0.703** for linear. CKA carries essentially no signal about whether identity
will work, and strong signal about whether a *linear translator* will work.

So the rule should be inverted in meaning: **high CKA is evidence that a linear translator will
succeed, not evidence that you can skip having one.** Some of the best results in the sweep are
high-CKA spans bridged linearly — `(2,3)` at CKA 0.83 loses 0.98pp, `(5,6)` at 0.85 loses
1.17pp, `(4,5)` at 0.87 loses 1.28pp — while those same spans lose 25.1, 44.9 and 20.5 points
with identity.

## The translator is doing enormous work

Worth stating separately, because it reframes what TOAST's linear map is for. Across the sweep
the linear translator is often nearly free while identity is catastrophic on the same span:

| span | CKA | identity drop | linear drop |
|---|---|---|---|
| (2, 3) | 0.83 | 25.1 pp | **0.98 pp** |
| (5, 6) | 0.85 | 44.9 pp | **1.17 pp** |
| (4, 5) | 0.87 | 20.5 pp | **1.28 pp** |
| (3, 4) | 0.90 | 11.0 pp | **1.37 pp** |
| (2, 4) | 0.70 | 64.7 pp | **5.65 pp** |

`(2,4)` is the extreme case: dropping two blocks costs 64.7 points unbridged and 5.7 points with
a 768×768 matrix. The translator is not a refinement on top of block-dropping — it is the thing
that makes block-dropping viable at all.

This is also what motivates the low-rank work: if a single linear map recovers that much, the
natural question is how much of it is really needed. (See the rank sweep — `rrr_*` versus
`lowrank_*` versus `lora_*`.)

## What does predict the damage

Neither position nor span alone; both, with span dominating for identity.

| predictor | vs identity drop | vs linear drop |
|---|---|---|
| skip distance | **+0.739** | +0.463 |
| CKA | −0.146 | **−0.765** |
| depth (`skip_from`) | −0.525 | — |

Distance is the strongest predictor of unbridged damage (+0.739): drop two adjacent blocks
instead of one and the loss roughly doubles (mean 16.9pp → 34.2pp). Depth is negatively
correlated, i.e. later blocks are cheaper to remove — consistent with the usual finding that
late transformer blocks are more redundant, and with the low block-influence scores
`layer_priority` assigns them.

The practical selection rule this suggests: **choose spans by distance and depth, choose the
translator by CKA** — rather than using CKA for both, which is what happens now.

## Caveats

- One model, one dataset (DINOv2-base / ImageNet-1k), 21 single-span configs with both
  translators. The mechanism argument is general; the specific numbers are not yet.
- The rule fired only 3 times here, because CKA ≥ 0.90 is rare on this model. That is a small
  sample for the headline claim, even though it is 0-for-3 and the correlation evidence across
  all 21 points the same way.
- Multi-span configs are excluded, since attributing a drop to one span is ambiguous.
- Worth re-running on the medical datasets once rad-dino's layer_priority scores exist — if
  CKA's failure as an identity criterion reproduces there, the claim is much stronger.

## Recommended change

Do not just retune `CKA_IDENTITY`. Either drop the identity branch from
`recommend_runs.translator_for_span` entirely (identity was never the right call on any span
tested), or invert its meaning so high CKA selects `linear` with confidence and low CKA flags a
span as hard to bridge at all — the `(10,11)`, `(7,8)`, `(9,11)` and `(6,8)` cases, where CKA is
0.16–0.35 and *both* bridges fail.
