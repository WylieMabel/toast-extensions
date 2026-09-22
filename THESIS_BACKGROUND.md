# Thesis Background — Reading Checklist & Section Structure

## Framing

The contribution is an **analysis**, not a benchmark and not a new metric. Given TOAST:

1. Which units (attention heads / MLP blocks / attention sublayers / contiguous layer spans) are redundant?
2. Which cheap similarity signal best anticipates the damage of removing/linearising them?
3. Does either finding survive a move to the medical domain — including translators fit on public data and applied to a private target?

The background's job: (a) establish that redundancy exists and is *structured*, (b) survey the removal-cost signals people already use, (c) explain what those signals do and don't measure, (d) show the domain-transfer question is open.

`[have]` = already on the list / in `metric_experiments/NOVELTY.md`; `[add]` = missing.

---

## 1. Paper review checklist

### A. Redundancy & over-parameterisation in transformers

- [ ] `[have]` **TOAST** — Cannistraci et al., TMLR 2026 ([2410.04941](https://arxiv.org/abs/2410.04941)). The method. Extract: what a "translator" is, which blocks are replaced, that it is training-free, the datasets/backbones — the thesis inherits all of this.
- [ ] `[add]` **The Tunnel Effect** — Masarczyk et al., NeurIPS 2024 ([2305.19753](https://arxiv.org/abs/2305.19753)). Extract: deep layers stop building new representations → why redundancy is concentrated in depth and is depth-dependent.
- [ ] `[add]` **Are Sixteen Heads Really Better than One?** — Michel et al., NeurIPS 2019. Extract: many heads removable at test time; importance is head-specific — the canonical head-granularity result.
- [ ] `[add]` **Analyzing Multi-Head Self-Attention** — Voita et al., ACL 2019. Extract: heads specialise; a prunable majority. Pair with Michel for the head-level analysis.
- [ ] `[have]` **On Attention Redundancy: A Comprehensive Study**. Extract: attention-map redundancy across layers/heads; how it is quantified.
- [ ] `[add]` **Adaptive MLP Pruning for Large Vision Transformers**. Extract: MLP-block redundancy is real and separable from attention — motivates treating MLP vs attention skips differently.

### B. Structured depth reduction (the removal operation)

- [ ] `[have]` **The Unreasonable Ineffectiveness of the Deeper Layers** — Gromov et al., ICLR 2025 ([2403.17887](https://arxiv.org/abs/2403.17887)). **Central.** Extract: contiguous-span dropping, angular-distance selection, "never drop the last block," and their explicit statement that they do not defend the metric choice — the question the thesis takes up.
- [ ] `[have]` **ShortGPT / Block Influence** — Men et al. 2024 ([2403.03853](https://arxiv.org/abs/2403.03853)). Extract: BI = 1 − cos(in, out) as the removal-cost score.
- [ ] `[have]` **Uncovering the Redundancy in Transformers via a Unified Study of Layer Dropping**. Extract: taxonomy of what gets dropped (module type, position).
- [ ] `[have]` **What Layers When: Learning to Skip Compute in LLMs with Residual Gates**. Extract: learned/dynamic skipping — contrast to the fixed-span static analysis here.
- [ ] `[have]` **Attention Residuals**. Extract: attention sublayers as near-identity in parts of the network — supports skipping attention separately from MLP.
- [ ] `[add]` **GM-Skip** — metric-guided block skipping for VLMs, 2025 ([2508.18227](https://arxiv.org/abs/2508.18227)). Extract: greedy accuracy-guided selection in a vision-adjacent setting — the "just use task signal" position to compare against.
- [ ] `[add]` **Effective Layer Pruning Through Similarity Metric Perspective** — Pons et al. 2024 ([2405.17081](https://arxiv.org/pdf/2405.17081)). Extract: CKA-between-representations as the pruning criterion — direct precedent for the CKA metrics.

### C. What the removal-cost signals are, and what they measure

- [ ] `[add]` **Similarity of Neural Network Representations Revisited (CKA)** — Kornblith et al., ICML 2019. Extract: definitions of linear CKA / CCA; the metric family behind `local_cka`, `cka_embed`.
- [ ] `[have]` **Similarity of Neural Network Models: A Survey** — Klabunde et al. 2023 ([2305.06329](https://arxiv.org/abs/2305.06329)). Extract: the functional-vs-representational split — the backbone of the "global / local / output / task-level" framing. Cite where the metric families are introduced.
- [ ] `[add]` **Deconfounded Representation Similarity** — Cui et al., NeurIPS 2022. Extract: CKA/RSA inflated by input population structure → why global embedding similarity can miss a locally disruptive skip (the DINOv2 observation).
- [ ] `[have]` **ReSi** — ICLR 2025 ([2408.00531](https://arxiv.org/abs/2408.00531)). Extract: even careful similarity measures do not reliably track behaviour — supports checking rather than assuming.
- [ ] `[add]` **Do Vision Transformers See Like CNNs?** — Raghu et al., NeurIPS 2021. Extract: layer-wise CKA structure of ViTs (early/late split, skip-connection role) — grounds the local-boundary vs final-embedding distinction.
- [ ] `[have]` **Tracing Representation Progression: Analyzing and Enhancing Layer-Wise Similarity**. Extract: how inter-layer similarity grows with depth in ViTs.
- [ ] `[have]` **Block-Recurrent Dynamics in ViT**. Extract: mid-network blocks act like a repeated fixed-point map — a mechanistic account of why contiguous mid spans are skippable.

### D. Training-free approximation with simple maps (the translator story)

- [ ] `[add]` **Relative representations enable zero-shot latent space communication** — Moschella et al., ICLR 2023 ([2209.15430](https://arxiv.org/abs/2209.15430)). Extract: latent spaces related by near-isometries; simple maps align them without training. Justifies a linear translator functional form.
- [ ] `[add]` **Bootstrapping Parallel Anchors for Relative Representations** — Cannistraci et al. 2023. Extract: alignment from few anchors — precedent for fitting the translator on a small public sample.
- [ ] `[add]` **Understanding image representations by measuring equivariance and equivalence** — Lenc & Vedaldi, CVPR 2015. Extract: the original "learn one linear layer to stitch two networks" result.
- [ ] `[add]` **LoRA** — Hu et al., ICLR 2022. Extract: low-rank updates suffice — the prior for the `lowrank_translator` sweep.

### E. Domain-specificity and transfer of pruning decisions

- [ ] `[add]` **MedMNIST v2** — Yang et al., Scientific Data 2023. Extract: PathMNIST / ChestMNIST / PneumoniaMNIST specs — must cite as the benchmark.
- [ ] `[add]` **From CNNs to Vision Transformers: Domain Shift and Adaptation in Medical Image Analysis** — Springer 2026 ([10.1007/s11831-026-10702-8](https://link.springer.com/article/10.1007/s11831-026-10702-8)). Extract: the natural→medical context gap; ViT adaptation behaviour under shift.
- [ ] `[have]` **Straightforward Layer-wise Pruning for More Efficient Visual Adaptation** ([2407.14330](https://arxiv.org/abs/2407.14330)). Extract: redundancy pattern shifts under distribution change — closest precedent for "skip choice may not transfer."
- [ ] `[add]` **The Lighter The Better: Rethinking Transformers in Medical Image Segmentation Through Adaptive Pruning** ([2206.14413](https://arxiv.org/pdf/2206.14413)). Extract: ImageNet-derived importance does not carry cleanly to medical models.
- [ ] `[add]` **RAPID: Layer-Wise Redundancy-Aware Pruning for Efficient ViT**, 2026 ([2606.08156](https://arxiv.org/html/2606.08156)). Extract: current ViT-native redundancy pruning — situates the work against vision, not just LLM, work.

### F. Optional / concurrent (cite briefly; verify arXiv IDs first)

- [ ] `[have]` **Rethinking Layer Redundancy: Calibration Matters More Than Search** ([2604.24938](https://arxiv.org/html/2604.24938v3)) — "task objective beats structural proxy," LLM-only. One sentence.
- [ ] `[have]` **Layer Pruning with Consensus** ([2411.14345](https://arxiv.org/html/2411.14345v1)) — ensembling similarity metrics. One sentence, as an alternative to picking one.
- [ ] `[add]` **HetDPT / Rethinking Depth Pruning for ViTs** ([2607.03784](https://arxiv.org/pdf/2607.03784)) — recent ViT depth pruning; concurrent-work paragraph.

---

## 2. ICLR-style background section structure

Target ~1.5–2 pages, 6 subsections. Each ends with a sentence that hands off to the next.

### 2. Background and Related Work

**2.1 Redundancy in pretrained transformers.**
Over-parameterisation; evidence that whole components contribute little. Heads (Michel; Voita; *On Attention Redundancy*), MLP blocks (Adaptive MLP Pruning), and depth (Tunnel Effect). Establishes: redundancy is real *and structured by unit type and position* — which is why a per-granularity analysis is worth doing.
→ *"Exploiting this redundancy requires (i) a way to remove or replace a unit and (ii) a way to decide which."*

**2.2 Removing depth: layer dropping, skipping, and block approximation.**
The removal operation. LLM lineage: Gromov et al. (contiguous spans), ShortGPT (block influence), *Unified Study of Layer Dropping*, *What Layers When* (dynamic gating). Then the training-free branch: TOAST — replace a block with a linear map or identity, no fine-tuning — and where it leaves the selection problem open. State plainly that most of this evidence is from LLMs and that vision transformers differ in depth dynamics (forward ref to 2.4/2.5).
→ *"Given a removal operation, the open question is how to predict its cost cheaply."*

**2.3 Predicting the cost of removal.**
Survey the signals: input–output similarity at the boundary (angular distance, cosine / block influence), CKA-based criteria (Pons et al.), and task-level signals (KL / accuracy under the frozen head; GM-Skip's greedy accuracy). Organise them explicitly into the four families — **local (boundary), global (final embedding), output-distribution, task-level** — and note that the literature rarely compares them: Gromov et al. state outright that they do not defend angular distance over alternatives. This is the gap the analysis addresses — *which family gives the best cheap prediction*, not a new metric.
→ *"Whether these signals track the actual functional damage is itself contested."*

**2.4 What representation similarity measures — and what it misses.**
Klabunde et al.'s functional-vs-representational distinction as the organising frame. CKA definitions (Kornblith). Deconfounded CKA (Cui et al.) and ReSi: high representational similarity need not mean preserved behaviour, and global similarity is sensitive to input population structure. This motivates why a *local* boundary probe might outperform a *global* embedding-similarity metric — and why the answer could depend on the backbone (e.g. self-supervised).

**2.5 How ViT representations evolve with depth.**
Why redundancy exists and is depth-dependent, ViT-specifically. Raghu et al. (layer-wise CKA structure, skip connections), *Tracing Representation Progression*, *Block-Recurrent Dynamics*. Grounds the expectation that mid-network contiguous spans are the skippable ones and that natural-image ViT structure may not hold in a new domain.

**2.6 Simple maps between latent spaces, and domain transfer.**
Two threads the thesis extends. (a) *Linear maps suffice to relate latent spaces*: relative representations (Moschella), parallel anchors (Cannistraci), model stitching (Lenc & Vedaldi), low-rank sufficiency (LoRA) — the basis for fitting a translator on a small, public sample. (b) *Pruning decisions may not transfer*: MedMNIST as the target domain; ViT medical domain-shift review; Straightforward Layer-wise Pruning and *The Lighter The Better* showing ImageNet-derived importance degrades in medical models; RAPID / HetDPT as concurrent ViT-native work.
→ *"We therefore ask whether the redundancy structure and the best predictive signal identified on natural images persist under a domain shift, and whether a translator estimated on public data transfers to a private target."*

**Positioning paragraph** (end of 2.6, or its own short 2.7).
One paragraph, three sentences: prior work removes layers in LLMs using one similarity heuristic each and rarely compares them; TOAST shows blocks can be replaced by trivial maps in vision but does not say which; no one has checked whether the choice of unit, the predictive signal, or a fitted translator survives a domain shift. That is the thesis.
