# Layer / sublayer / head redundancy analysis

`layer_priority.py` runs 500–5000 samples through an encoder **once** and produces cheap
redundancy scores that rank what you can drop, *before* you spend compute verifying candidates
through the full encode → train → `delta_acc` pipeline. It is the block/sublayer analogue of
`src/heads/head_priority.py` (which only ranks attention heads), and it also recomputes the head
metric so one run covers all four granularities.

It only writes score files. Choosing configs and plotting are done separately.

```bash
python src/layers/layer_priority.py --model deitsmall --dataset dermamnist --num-samples 500
# or on the cluster:
sbatch src/layers/run_layer_priority.sh
```

`--model` takes an alias (`deitsmall`, `dinobase`, `vitlarge`, `raddino`, …) or a raw HF id.
`--dataset` is any key in `DATASET2LOCAL_PATH` (`dermamnist`, `chestmnist`, `mnist`, `cifar100`,
`imagenet-1k`, …). The architecture is navigated via `MODEL2CONFIGS`, so any ViT-family HF model
already registered there works.

## The metrics and the papers behind them

Let `h_i` be the residual stream entering block `i`, and inside a block
`h1 = h0 + Attn(LN(h0))`, `h2 = h1 + MLP(LN(h1))`. Cosines are averaged over all tokens/samples.

**Block Influence (BI)** — *ShortGPT: Layers in LLMs are More Redundant Than You Expect* (Men et
al., 2024). `BI = 1 − E[cos(input, output)]`. A block/sub-block with low BI barely rotates the
residual stream, so removing it changes the representation little.
- `bi_block[i] = 1 − cos(h0, h2)` — rank whole blocks for the `skip` column.
- `bi_attn[i]  = 1 − cos(h0, h1)` — rank attention sub-blocks for the `attn_skip` column.
- `bi_mlp[i]   = 1 − cos(h1, h2)` — rank MLP sub-blocks for the `mlp_skip` column.
- `relnorm_attn`, `relnorm_mlp = E[‖Δ‖ / ‖residual‖]` — a norm-based second opinion on the same
  question (how large is the sub-block's contribution relative to the stream).

**Angular distance over spans** — *The Unreasonable Ineffectiveness of the Deeper Layers* (Gromov
et al., 2024). `d(i, j) = (1/π)·arccos(E[cos(h_i, h_j)])`. A skip `(skip_from, skip_to)` bridges
`out(skip_from) → out(skip_to)`, i.e. hidden states `skip_from+1 → skip_to+1`; small
`d(skip_from+1, skip_to+1)` means that whole run of blocks can be replaced by a single mapping. The
`best_span` table in the block-scores file gives, for each number of dropped blocks, the
lowest-distance `(skip_from, skip_to)` — exactly a `skip = [(skip_from, skip_to)]` config candidate.

**Linear CKA** — *Similarity of Neural Network Representations Revisited* (Kornblith et al., 2019).
Representational similarity between two layers. High `cka[skip_from+1, skip_to+1]` for a span you
plan to skip suggests an `identity` `skip_translator` will hold up; low CKA means fit a `linear` one.

**Head JSD** — *Are Sixteen Heads Really Better Than One?* (Michel et al., 2019) and *Analyzing
Multi-Head Self-Attention* (Voita et al., 2019). Pairwise Jensen-Shannon divergence between the
attention maps of heads within a layer; `head_similarity[l, i, j]` near 1 means heads `i` and `j`
attend almost identically, so one is redundant. Same computation as `head_priority.py`.

## Output files (in `src/layers/outputs/`, stem `{model}_{dataset}`)

| File | Contents | Feeds config column |
|------|----------|---------------------|
| `{stem}_block_scores.csv` | per-block `bi_block`, `angdist_in_out`; then a `blocks_dropped / skip_from / skip_to` best-placement table | `skip` |
| `{stem}_sublayer_scores.csv` | per-layer `bi_attn`, `relnorm_attn`, `bi_mlp`, `relnorm_mlp` | `attn_skip`, `mlp_skip` |
| `{stem}_angdist.npy` | full `[L+1, L+1]` angular-distance matrix | `skip` (span choice) |
| `{stem}_cka.npy` | full `[L+1, L+1]` linear-CKA matrix | `skip_translator` (identity vs linear) |
| `{stem}_head_similarity.npy` | `[L, H, H]` JSD similarity matrix | `head_dict` |

Indexing: hidden states are `h_0 … h_L` (length `L+1`); block `i` maps `h_i → h_{i+1}`, and its
output `out(i) == h_{i+1}`. The `skip` convention (from `SkipModel`) uses **layer-output indices**
`0 … L-1`: a pair `(skip_from, skip_to)` keeps block `skip_from`, drops blocks
`skip_from+1 … skip_to`, and bridges `out(skip_from) → out(skip_to)` with a translator. Both indices
must be `≤ L-1`, so e.g. `(8, 12)` is illegal on a 12-block model. To drop a single block `j`, write
`(j-1, j)`.

## How to turn scores into a config

1. **Whole blocks** — sort `bi_block` ascending; the lowest are the safest single blocks. Drop block
   `j` with `skip = [(j-1, j)]`. For multi-block skips, read the best-placement table (or
   `angdist.npy`) for the lowest-distance run of the length you want → `skip = [(skip_from, skip_to)]`.
2. **MLP vs attention** — sort `bi_mlp` / `bi_attn` ascending; take the lowest indices into
   `mlp_skip` / `attn_skip`. `relnorm_*` should agree; where they disagree, trust the one with the
   larger gap to its neighbours.
3. **Translator** — check `cka[skip_from+1, skip_to+1]` for the span: high → try
   `skip_translator = identity` first; low → `linear`.
4. **Heads** — threshold `head_similarity` per layer (as `head_priority.py` does) to build
   `head_dict`.
5. **Verify** — put the baseline (empty skip) plus your top few candidates into an
   `experiments_*.csv` and run them through `run_pipeline_row_by_row.sh`; confirm the low-score
   candidates give small `delta_acc` in `results/results_full_heads.csv`.

Sanity check that matches the papers: `bi_block` should trend **downward with depth** — the deeper
blocks are the most redundant and the first you should try to drop.
