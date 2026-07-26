"""
Layer / sublayer / head redundancy analyser.

Runs N samples through a vision encoder ONCE and, from the per-layer activations,
computes cheap redundancy signals that rank what can be dropped:

  - Block Influence (ShortGPT, Men et al. 2024):  BI_i = 1 - E_tokens cos(h_in, h_out).
    Low BI = the block barely moves the residual stream = safe to drop. Computed per
    whole block, and per sub-block (attention vs MLP) so the two are ranked separately.
  - Angular distance over spans (Gromov et al. 2024): d(i,j) = (1/pi) arccos(E cos(h_i,h_j)).
    The contiguous span with the smallest distance is the best (start, end) to skip.
  - Linear CKA (Kornblith et al. 2019): representational similarity between layer pairs;
    high CKA predicts an `identity` skip translator suffices, low predicts you need `linear`.
  - Head JSD (Michel et al. 2019, Voita et al. 2019): pairwise Jensen-Shannon divergence
    between attention heads within a layer; high similarity = redundant head.

It ONLY writes score files. Choosing configs and plotting are done separately.

Usage:
    python layer_priority.py --model deitsmall --dataset dermamnist --num-samples 500
    python layer_priority.py --model facebook/dinov2-base --dataset cifar100 --num-samples 5000
"""

import argparse
import csv
import gc
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from datasets import load_from_disk
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoImageProcessor, AutoModel

from toast.utils.dictionaries import (
    DATASET2INPUT_COLUMN,
    DATASET2LABEL_COLUMN,
    DATASET2LOCAL_PATH,
    MODEL2CONFIGS,
)

# Friendly aliases -> HF id (a raw HF id may also be passed to --model directly).
MODEL_ALIASES = {
    "deitsmall": "facebook/deit-small-patch16-224",
    "deitbase": "facebook/deit-base-patch16-224",
    "dinosmall": "facebook/dinov2-small",
    "dinobase": "facebook/dinov2-base",
    "raddino": "microsoft/rad-dino",
    "vittiny": "WinKawaks/vit-tiny-patch16-224",
    "vitsmall": "WinKawaks/vit-small-patch16-224",
    "vitbase": "google/vit-base-patch16-224",
    "vitlarge": "google/vit-large-patch16-224",
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------------------------------------------------------- #
# Data / model loading (mirrors src/heads/head_priority.py)
# ----------------------------------------------------------------------------- #
def _collate(samples, processor, image_col, label_col):
    images = [s[image_col].convert("RGB") for s in samples]
    pixel_values = torch.cat(
        [processor(images=img, return_tensors="pt")["pixel_values"] for img in images],
        dim=0,
    )
    labels = torch.tensor([s[label_col] for s in samples])
    return pixel_values, labels


def _load_dataset(dataset_name: str, num_samples: int, hf_id: str):
    if dataset_name not in DATASET2LOCAL_PATH:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. Known: {sorted(DATASET2LOCAL_PATH)}"
        )
    raw = load_from_disk(DATASET2LOCAL_PATH[dataset_name])
    split = "validation" if "validation" in raw else "test"
    ds = raw[split]
    n = min(num_samples, len(ds))
    ds = ds.select(range(n))

    processor = AutoImageProcessor.from_pretrained(hf_id)
    image_col = DATASET2INPUT_COLUMN[dataset_name]
    label_col = DATASET2LABEL_COLUMN[dataset_name]
    collate_fn = lambda batch: _collate(batch, processor, image_col, label_col)

    return DataLoader(ds, batch_size=8, shuffle=False, num_workers=1, collate_fn=collate_fn), n


def _resolve_hf_id(model_arg: str) -> str:
    key = model_arg.lower().strip()
    return MODEL_ALIASES.get(key, model_arg)


def _load_model(hf_id: str):
    enc_config = AutoConfig.from_pretrained(
        hf_id, output_attentions=True, output_hidden_states=True, return_dict=True
    )
    model = AutoModel.from_pretrained(hf_id, config=enc_config, attn_implementation="eager")
    model.to(device).eval()
    num_layers = model.config.num_hidden_layers
    num_heads = getattr(model.config, "num_attention_heads", None)
    return model, num_layers, num_heads


def _resolve_layers(model, hf_id: str):
    """Locate the ModuleList of transformer blocks via MODEL2CONFIGS navigation."""
    cfg = MODEL2CONFIGS.get(hf_id)
    if cfg is None:
        raise ValueError(
            f"'{hf_id}' not in MODEL2CONFIGS; add its layer-navigation entry there."
        )
    parent = model
    for part in cfg["layers_parent_path"].split("."):
        if part:
            parent = getattr(parent, part)
    return getattr(parent, cfg["layers_attribute_name"])


def _sublayer_modules(layer):
    """
    Return (attn_module, mlp_out_module) whose forward OUTPUT is the residual delta
    added to the stream: h1 = h0 + attn_out, h2 = h1 + mlp_out.

      DINOv2 / rad-dino: attn = layer.layer_scale1, mlp = layer.layer_scale2
          (a hybrid: has .attention AND .mlp, and wraps each sub-block in LayerScale,
           so the LayerScale output is the true residual delta.)
      HF ViT/DeiT:       attn = layer.attention (output[0]), mlp = layer.output.dense
      CLIP / timm:       attn = layer.self_attn|attn (output[0]), mlp = layer.mlp
    """
    if hasattr(layer, "layer_scale1") and hasattr(layer, "layer_scale2"):
        return layer.layer_scale1, layer.layer_scale2
    if hasattr(layer, "intermediate") and hasattr(layer, "output"):
        mlp_out = getattr(layer.output, "dense", None)
        return getattr(layer, "attention", None), mlp_out
    if hasattr(layer, "mlp"):
        attn = getattr(layer, "self_attn", None) or getattr(layer, "attn", None)
        return attn, layer.mlp
    return None, None


# ----------------------------------------------------------------------------- #
# Metric helpers
# ----------------------------------------------------------------------------- #
def _flatten(t):
    """[B, S, D] (or [T, D]) -> [T, D] float32 on device."""
    if t.dim() == 3:
        t = t.reshape(-1, t.shape[-1])
    return t.float()


def _hook_output(store, idx):
    def hook(_module, _inp, out):
        store[idx] = out[0] if isinstance(out, tuple) else out
    return hook


def _linear_cka(X, Y):
    """Feature-space linear CKA on centered features (only D x D matrices formed)."""
    Xc = X - X.mean(0, keepdim=True)
    Yc = Y - Y.mean(0, keepdim=True)
    hsic_xy = (Xc.t() @ Yc).pow(2).sum()
    hsic_xx = (Xc.t() @ Xc).pow(2).sum()
    hsic_yy = (Yc.t() @ Yc).pow(2).sum()
    denom = (hsic_xx.sqrt() * hsic_yy.sqrt()).clamp_min(1e-12)
    return (hsic_xy / denom).item()


def _accumulate_head_jsd(attentions, total_jsd, num_layers, num_heads):
    """Port of head_priority._accumulate_jsd for one batch. attentions: tuple[L] [B,H,S,S].

    Memory-lean form for long sequences (e.g. rad-dino at 518px, S~1370): iterate the
    attention tuple directly instead of torch.stack-ing it (avoids duplicating all L
    attention maps), and loop over the query head so the largest intermediate is [H,S,S]
    rather than the [H,H,S,S] broadcast the vectorised version built. Numerically identical.
    """
    bs = attentions[0].shape[0]
    for l in range(num_layers):
        a_l = attentions[l]                                          # [B, H, S, S] (no copy)
        for b in range(bs):
            heads_c = torch.clamp(a_l[b], min=1e-10)                 # [H, S, S]
            log_h = torch.log2(heads_c)
            rows = []
            for i in range(num_heads):
                Pi = heads_c[i].unsqueeze(0)                         # [1, S, S]
                M = torch.clamp(0.5 * (Pi + heads_c), min=1e-10)     # [H, S, S]
                log_M = torch.log2(M)
                kl_pm = torch.sum(Pi * (log_h[i].unsqueeze(0) - log_M), dim=-1)  # [H, S]
                kl_qm = torch.sum(heads_c * (log_h - log_M), dim=-1)             # [H, S]
                rows.append((0.5 * kl_pm + 0.5 * kl_qm).mean(dim=-1))            # [H]
            total_jsd[l] += torch.stack(rows).cpu().double()         # [H, H]
    return bs


# ----------------------------------------------------------------------------- #
# Main capture pass
# ----------------------------------------------------------------------------- #
@torch.no_grad()
def analyse(model, loader, num_layers, num_heads, layers, token_cap):
    n_states = num_layers + 1  # hidden_states length

    # BI / relnorm accumulators (streaming, full tokens)
    sum_cos_block = np.zeros(num_layers, dtype=np.float64)
    sum_cos_attn = np.zeros(num_layers, dtype=np.float64)
    sum_cos_mlp = np.zeros(num_layers, dtype=np.float64)
    sum_relnorm_attn = np.zeros(num_layers, dtype=np.float64)
    sum_relnorm_mlp = np.zeros(num_layers, dtype=np.float64)
    tok_count = np.zeros(num_layers, dtype=np.float64)

    # angular-distance accumulator (streaming): sum of per-token cosine between every
    # pair of hidden states, plus total token count.
    sum_cos_pair = torch.zeros((n_states, n_states), dtype=torch.float64)
    pair_tokens = 0

    # capped token store per hidden state, for CKA at the end (float16 to save memory)
    cka_store = [[] for _ in range(n_states)]
    cka_tokens = 0

    # head JSD
    do_heads = num_heads is not None
    total_jsd = torch.zeros((num_layers, num_heads, num_heads), dtype=torch.float64) if do_heads else None
    n_head_imgs = 0

    # register sublayer output hooks once
    attn_out, mlp_out = {}, {}
    hooks = []
    for idx, layer in enumerate(layers):
        a_mod, m_mod = _sublayer_modules(layer)
        if a_mod is not None:
            hooks.append(a_mod.register_forward_hook(_hook_output(attn_out, idx)))
        if m_mod is not None:
            hooks.append(m_mod.register_forward_hook(_hook_output(mlp_out, idx)))

    try:
        for images, _ in tqdm.tqdm(loader, desc="  Analysing"):
            images = images.to(device)
            attn_out.clear()
            mlp_out.clear()
            outputs = model(images)
            hs = outputs.hidden_states  # tuple len n_states, each [B, S, D]

            # ----- block + sublayer BI / relnorm -----
            for i in range(num_layers):
                h0 = _flatten(hs[i])
                h2 = _flatten(hs[i + 1])
                t = h0.shape[0]
                sum_cos_block[i] += F.cosine_similarity(h0, h2, dim=-1).sum().item()
                tok_count[i] += t

                if i in attn_out:
                    d_attn = _flatten(attn_out[i])
                    h1 = h0 + d_attn
                    sum_cos_attn[i] += F.cosine_similarity(h0, h1, dim=-1).sum().item()
                    sum_cos_mlp[i] += F.cosine_similarity(h1, h2, dim=-1).sum().item()
                    sum_relnorm_attn[i] += (
                        d_attn.norm(dim=-1) / h1.norm(dim=-1).clamp_min(1e-12)
                    ).sum().item()
                    if i in mlp_out:
                        d_mlp = _flatten(mlp_out[i])
                        sum_relnorm_mlp[i] += (
                            d_mlp.norm(dim=-1) / h2.norm(dim=-1).clamp_min(1e-12)
                        ).sum().item()

            # ----- angular distance: per-token cosine between all hidden-state pairs -----
            U = torch.stack([F.normalize(_flatten(h), dim=-1) for h in hs])  # [n_states, T, D]
            sum_cos_pair += torch.einsum("itd,jtd->ij", U, U).double().cpu()
            pair_tokens += U.shape[1]

            # ----- capped token store for CKA -----
            if cka_tokens < token_cap:
                take = min(hs[0].reshape(-1, hs[0].shape[-1]).shape[0], token_cap - cka_tokens)
                for s in range(n_states):
                    cka_store[s].append(_flatten(hs[s])[:take].half().cpu())
                cka_tokens += take

            # ----- head JSD -----
            if do_heads and outputs.attentions is not None:
                n_head_imgs += _accumulate_head_jsd(
                    outputs.attentions, total_jsd, num_layers, num_heads
                )

            del outputs, hs, U
            if device.type == "cuda":
                torch.cuda.empty_cache()
            gc.collect()
    finally:
        for h in hooks:
            h.remove()

    have_sub = tok_count.sum() > 0 and sum_cos_attn.any()

    results = {
        "bi_block": 1.0 - sum_cos_block / np.clip(tok_count, 1, None),
        "bi_attn": 1.0 - sum_cos_attn / np.clip(tok_count, 1, None) if have_sub else None,
        "bi_mlp": 1.0 - sum_cos_mlp / np.clip(tok_count, 1, None) if have_sub else None,
        "relnorm_attn": sum_relnorm_attn / np.clip(tok_count, 1, None) if have_sub else None,
        "relnorm_mlp": sum_relnorm_mlp / np.clip(tok_count, 1, None) if have_sub else None,
    }

    # angular-distance matrix [n_states, n_states]
    mean_cos_pair = (sum_cos_pair / max(pair_tokens, 1)).clamp(-1.0, 1.0)
    angdist = (torch.arccos(mean_cos_pair) / torch.pi).numpy()
    results["angdist"] = angdist

    # CKA matrix [n_states, n_states]
    reps = [torch.cat(chunks).float().to(device) for chunks in cka_store]
    cka = np.eye(n_states, dtype=np.float64)
    for a in range(n_states):
        for b in range(a + 1, n_states):
            cka[a, b] = cka[b, a] = _linear_cka(reps[a], reps[b])
    results["cka"] = cka

    # head similarity [L, H, H]
    if do_heads and n_head_imgs > 0:
        results["head_similarity"] = (1.0 - total_jsd / n_head_imgs).numpy()
    else:
        results["head_similarity"] = None

    return results


# ----------------------------------------------------------------------------- #
# Output writers
# ----------------------------------------------------------------------------- #
def _best_spans(angdist, num_layers, max_span):
    """
    Best placement for dropping k contiguous blocks, in the SkipModel `skip`
    convention. A pair (skip_from, skip_to) keeps block `skip_from`, DROPS blocks
    skip_from+1 .. skip_to, and bridges out(skip_from) -> out(skip_to) with a
    translator. Both indices are layer-output keys 0 .. num_layers-1, so skip_to
    is capped at num_layers-1 (this is what makes (8, 12) illegal for a 12-block model).

    out(i) == hidden_state[i+1], so the distance the bridge must cover for
    (skip_from, skip_to) is angdist[skip_from+1, skip_to+1]. For each k = skip_to -
    skip_from (number of blocks dropped) we return the min-distance placement as
    (blocks_dropped, skip_from, skip_to, distance).
    """
    rows = []
    max_k = min(max_span, num_layers - 1)
    for k in range(1, max_k + 1):
        best = None
        for a in range(0, num_layers - k):          # skip_from=a, skip_to=a+k <= num_layers-1
            d = angdist[a + 1, a + k + 1]           # out(a) -> out(a+k)
            if best is None or d < best[2]:
                best = (a, a + k, d)
        rows.append((k, best[0], best[1], best[2]))
    return rows


def _save_block_scores(results, num_layers, max_span, path):
    bi = results["bi_block"]
    ang = results["angdist"]
    spans = {k: (s, e, d) for (k, s, e, d) in _best_spans(ang, num_layers, max_span)}
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        # bi_block[i] == angular-ish redundancy of block i; drop block i via skip=(i-1, i).
        w.writerow(["block", "bi_block", "angdist_in_out"])
        for i in range(num_layers):
            w.writerow([i, f"{bi[i]:.6f}", f"{ang[i, i + 1]:.6f}"])
        w.writerow([])
        # skip = [(skip_from, skip_to)] drops `blocks_dropped` (= skip_to - skip_from) blocks.
        w.writerow(["blocks_dropped", "skip_from", "skip_to", "angdist"])
        for k in sorted(spans):
            s, e, d = spans[k]
            w.writerow([k, s, e, f"{d:.6f}"])
    print(f"  block scores      -> {path}")


def _save_sublayer_scores(results, num_layers, path):
    if results["bi_attn"] is None:
        print("  (no sublayer split available for this architecture; skipping)")
        return
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["layer", "bi_attn", "relnorm_attn", "bi_mlp", "relnorm_mlp"])
        for i in range(num_layers):
            w.writerow([
                i,
                f"{results['bi_attn'][i]:.6f}",
                f"{results['relnorm_attn'][i]:.6f}",
                f"{results['bi_mlp'][i]:.6f}",
                f"{results['relnorm_mlp'][i]:.6f}",
            ])
    print(f"  sublayer scores   -> {path}")


def run(model_arg, dataset_name, num_samples, max_span, token_cap, output_dir, force=False):
    hf_id = _resolve_hf_id(model_arg)
    ds_key = dataset_name.lower().strip()
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{model_arg.lower().strip().replace('/', '_')}_{ds_key}"

    # Skip the (expensive, GPU) analysis pass if the core score files already exist.
    # Gate on the block-scores CSV + both matrices so a partial/interrupted run still
    # recomputes; head-similarity is model-dependent so it is not part of the check.
    expected = [
        out_dir / f"{stem}_block_scores.csv",
        out_dir / f"{stem}_angdist.npy",
        out_dir / f"{stem}_cka.npy",
    ]
    if not force and all(p.exists() for p in expected):
        print(f"  Scores already exist for {stem}; skipping (pass --force to recompute).")
        return

    print(f"\n{'=' * 60}")
    print(f"  Model   : {model_arg}  ({hf_id})")
    print(f"  Dataset : {ds_key}")
    print(f"  Samples : {num_samples}")
    print(f"  Device  : {device}")
    print(f"{'=' * 60}\n")

    print("[1/4] Loading model...")
    model, num_layers, num_heads = _load_model(hf_id)
    layers = _resolve_layers(model, hf_id)
    print(f"      {num_layers} blocks, {num_heads} heads/layer")

    print("[2/4] Loading dataset...")
    loader, actual_n = _load_dataset(ds_key, num_samples, hf_id)
    print(f"      Using {actual_n} samples")

    print("[3/4] Analysing...")
    results = analyse(model, loader, num_layers, num_heads, layers, token_cap)

    print("[4/4] Saving outputs...")
    _save_block_scores(results, num_layers, min(max_span, num_layers), out_dir / f"{stem}_block_scores.csv")
    _save_sublayer_scores(results, num_layers, out_dir / f"{stem}_sublayer_scores.csv")
    np.save(out_dir / f"{stem}_angdist.npy", results["angdist"])
    print(f"  angular distance  -> {out_dir / f'{stem}_angdist.npy'}")
    np.save(out_dir / f"{stem}_cka.npy", results["cka"])
    print(f"  CKA matrix        -> {out_dir / f'{stem}_cka.npy'}")
    if results["head_similarity"] is not None:
        np.save(out_dir / f"{stem}_head_similarity.npy", results["head_similarity"])
        print(f"  head similarity   -> {out_dir / f'{stem}_head_similarity.npy'}")

    print(f"\nDone. Outputs written to: {out_dir.resolve()}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Layer/sublayer/head redundancy analyser")
    p.add_argument("--model", required=True, help="alias (deitsmall, dinobase, vitlarge, ...) or HF id")
    p.add_argument("--dataset", required=True, help="dataset key (dermamnist, mnist, cifar100, imagenet-1k, ...)")
    p.add_argument("--num-samples", type=int, default=5000, help="images to run (default: 5000)")
    p.add_argument("--max-span", type=int, default=8, help="longest contiguous block span to report (default: 8)")
    p.add_argument("--token-cap", type=int, default=10000, help="tokens used for CKA (default: 10000)")
    p.add_argument("--output-dir", default="./outputs", help="output directory (default: ./outputs)")
    p.add_argument("--force", action="store_true", help="recompute even if score files already exist")
    args = p.parse_args()

    run(
        model_arg=args.model,
        dataset_name=args.dataset,
        num_samples=args.num_samples,
        max_span=args.max_span,
        token_cap=args.token_cap,
        output_dir=args.output_dir,
        force=args.force,
    )
