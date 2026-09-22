# Novelty Assessment — Skip-Span Predictor Metrics

Verdict up front: **your two best general metrics are both published prior art, and you should stop framing them as findings.** But the *evaluation* you've built is not published, and two of your negative results are genuinely open. The publishable contribution is a benchmark + stress test + cheap fix, not a new metric.

---

## 1. What is already covered (do not claim as novel)

| Your finding | Prior art | How close |
|---|---|---|
| `angular_dist` (boundary input↔output angle) is the best general predictor | [Gromov et al. 2024, *The Unreasonable Ineffectiveness of the Deeper Layers*](https://arxiv.org/abs/2403.17887) | **Identical metric.** They define d = (1/π)·arccos(cos(x_ℓ, x_{ℓ+n})) over *contiguous n-block spans* and pick the minimum. This is exactly your `angular_dist` over exactly your candidate set. |
| `cosine_sim` between skip input/output predicts safety | [ShortGPT / Block Influence, Men et al. 2024](https://arxiv.org/abs/2403.03853) | **Identical up to sign.** BI = 1 − cosine(input, output). |
| CKA-based selection (`cka_embed`, `local_cka`) | [Pons et al. 2024, *Effective Layer Pruning Through Similarity Metric Perspective*](https://arxiv.org/abs/2405.17081) | Same idea: CKA between unpruned and candidate-pruned representations as the pruning criterion. |
| Combining several similarity metrics helps | [*Layer Pruning with Consensus: A Triple-Win Solution*](https://arxiv.org/html/2411.14345v1) | Rank-sum ensemble of similarity metrics; explicitly argues single metrics are inconsistent across benchmarks. **This is a baseline you currently don't have.** |
| Deeper layers more skippable; never skip the last block | Gromov et al.; ShortGPT; [GM-Skip](https://arxiv.org/abs/2508.18227) | Well established, including the reverse-order-deletion result for vision/VLM. |
| Representation similarity ≠ functional equivalence | [Klabunde et al. survey](https://arxiv.org/abs/2305.06329), [ReSi benchmark, ICLR 2025](https://arxiv.org/abs/2408.00531), [*No Free Swap*](https://arxiv.org/pdf/2605.16234) | The *conceptual* point is thoroughly made. ReSi even benchmarks 24 similarity measures across vision/language/graph on whether they track functional behaviour. |
| Task-level/calibration objective beats structural proxies | [*Rethinking Layer Redundancy: Calibration Matters More Than Search*](https://arxiv.org/html/2604.24938v1); GM-Skip's greedy accuracy-guided selection | Same conclusion, but **LLM-only** (LLaMA3/Qwen3 8B) and VLM-only respectively. |
| Rank-correlation methodology for cheap proxies | Zero-cost NAS proxy literature (ParZC, AZ-NAS, GreenFactory) | Your Spearman/precision@k protocol is the standard evaluation template there. Cite it as method, not contribution. |

**Blunt implication for `orig_head_accuracy`:** on ImageNet-1k this is close to circular — you are using task accuracy under a frozen head to predict task accuracy under a retrained linear head. A reviewer will say "of course accuracy predicts accuracy." AUC 0.998 is not a finding *as currently framed*. It only becomes a finding if you turn it into a general method (see §3).

---

## 2. What is genuinely open

These are the gaps I could not find covered anywhere:

**(a) No exhaustive-span metric benchmark exists for ViTs.**
Gromov et al. are LLM-only and — this is the key sentence for your thesis — explicitly decline to compare metrics: *"we do not expect our choice of angular distance – in lieu of any other reasonable metric – to be particularly significant."* They never measure correlation with downstream accuracy across candidate spans; they only greedily pick the argmin. Nobody has tested that assumption. You have 15 model×dataset combos × 51–66 exhaustively enumerated contiguous spans × 3 seeds of real ground truth. **That asset is the contribution.**

**(b) Self-supervised backbones break global metrics but not local ones.**
Your DINOv2 result (global metrics → near chance, ρ 0.13–0.26; local metrics hold, ρ 0.81–0.90) has no counterpart in the literature I can find. Searches for SSL-vs-supervised layer redundancy return nothing on pruning-metric reliability. This is your strongest genuinely-novel nugget — but it is currently n=1 backbone.

**(c) Domain shift collapses metric predictiveness *and* skip transferability.**
PathMNIST kills every metric (ρ 0.88→0.30 on DeiT-S) *and* breaks cross-dataset rank transfer (ρ 0.03–0.24, n.s. for DeiT-S/ViT-S). The closest prior work ([Straightforward Layer-wise Pruning](https://arxiv.org/abs/2407.14330)) notes deeper layers carry more redundancy under distribution shift, but nobody has shown that the *selection metrics themselves stop working* out of domain. The pairing of both effects in one experiment is new.

**(d) Cross-dataset transfer of skip-span choice for ViTs.** Gromov et al. never tested transfer; cross-task pruning transfer exists for LLMs only. Your 4-dataset × 4-backbone grid is a clean design for this.

---

## 3. The contribution I'd actually pitch

Frame it as **benchmark → stress test → cheap fix**, in that order.

> *LLM layer-pruning heuristics don't survive contact with vision transformers. We build the first exhaustive skip-span benchmark for ViTs, show that the standard similarity metrics inherited from LLM pruning degrade to near-chance under domain shift and on self-supervised backbones, and show that a 500-image linear probe — costing less than the metrics it replaces — dominates all of them.*

That framing is defensible because every individual metric being prior art is *fine* — you're the evaluation, not the method.

**The one experiment that unlocks it:** generalise `orig_head_accuracy` into a **500-image linear-probe proxy** computable on *all 15 combos*, not just the 3 with a matching ImageNet head. You already extract the embeddings; fitting a logistic regression on 500 cached embeddings is seconds of CPU. If probe accuracy holds ~0.95+ AUC on PathMNIST and DINOv2 where every similarity metric fails, you have a complete story: *diagnosis (similarity metrics fail out-of-domain) + cure (task probe doesn't) + it's cheap.* Without this, you only have a negative result on 15 combos and a near-circular positive on 3.

**Other experiments reviewers will demand:**
- **Consensus/ensemble baseline** — rank-sum of your 8 metrics. The Consensus paper claims this beats singles; if your probe beats the ensemble too, that's a much stronger claim.
- **Regret, not just AUC.** "Pick top-1 span by metric X → what accuracy do you lose vs the oracle span?" This is what a pruning audience cares about, and your precision@3 numbers (mostly 0.0!) suggest the metrics look far worse under this framing than under ρ/AUC. That contrast — *good rank correlation, terrible top-k selection* — is itself a publishable observation and it's already sitting in your `_stats.csv` files.
- **A second SSL backbone** (MAE, DINOv1, or CLIP) to take the DINOv2 claim from n=1 to a pattern.
- Threshold sensitivity (0.5%/1%/2%) and Benjamini–Hochberg correction across the 15×12 test grid.

**Risk to watch:** the field is moving on this. [*Rethinking Depth Pruning for Vision Transformers*](https://arxiv.org/pdf/2607.03784) (HetDPT) is recent and ViT-native, and *Rethinking Layer Redundancy* already makes the "calibration objective beats structural proxy" argument for LLMs. Your defensible territory is specifically **ViTs + exhaustive spans + domain shift + SSL backbones**. Lead with those, and position the probe result as confirming-and-extending the LLM finding into vision rather than discovering it.

---

## 4. Reframing your current summary

| Current framing | Replace with |
|---|---|
| "angular_dist is the best general metric" | "We confirm Gromov et al.'s untested assumption that angular distance is a reasonable choice — *and* quantify where it fails (AUC 0.45–0.99 across combos)." |
| "orig_head_accuracy wins (AUC 0.998)" | "A task-level probe dominates all representation metrics; we generalise it beyond the ImageNet-head case to a 500-image linear probe." |
| "local beats global metrics" | Keep, but explain it: global embedding metrics are washed out by the residual stream's redundancy; the effect is strongest exactly where the backbone is most over-complete (DINOv2). |
| "PathMNIST is a weak spot" | "Metric reliability is domain-conditional — the primary negative result, not a caveat." |

## Sources
- [The Unreasonable Ineffectiveness of the Deeper Layers (Gromov et al., 2024)](https://arxiv.org/abs/2403.17887) · [HTML](https://arxiv.org/html/2403.17887v1)
- [ShortGPT: Layers in LLMs are More Redundant Than You Expect](https://arxiv.org/abs/2403.03853)
- [Effective Layer Pruning Through Similarity Metric Perspective](https://arxiv.org/abs/2405.17081)
- [Layer Pruning with Consensus: A Triple-Win Solution](https://arxiv.org/html/2411.14345v1)
- [ReSi: A Comprehensive Benchmark for Representational Similarity Measures (ICLR 2025)](https://arxiv.org/abs/2408.00531)
- [Similarity of Neural Network Models: A Survey (Klabunde et al.)](https://arxiv.org/abs/2305.06329)
- [No Free Swap: Protocol-Dependent Layer Redundancy in Transformers](https://arxiv.org/pdf/2605.16234)
- [Rethinking Layer Redundancy in LLMs: Calibration Objectives and Search](https://arxiv.org/html/2604.24938v1)
- [GM-Skip: Metric-Guided Transformer Block Skipping for VLMs](https://arxiv.org/abs/2508.18227)
- [Rethinking Depth Pruning for Vision Transformers (HetDPT)](https://arxiv.org/pdf/2607.03784)
- [Straightforward Layer-wise Pruning for More Efficient Visual Adaptation](https://arxiv.org/abs/2407.14330)
- [Similarity of Neural Network Representations Revisited (CKA, Kornblith et al. 2019)](http://proceedings.mlr.press/v97/kornblith19a/kornblith19a.pdf)
