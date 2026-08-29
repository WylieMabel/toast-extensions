"""
Output-level skip-safety predictors: KL divergence + top-1 flip rate (Exp 1) and
KNN@5 overlap + cosine similarity + MSE + CKA on final embeddings (Exp 2).

Evaluates single-layer skips (23 rows for vit-large: one per layer 0..22) on vit-large/imagenet-1k
using the original classification head (no retraining) and deterministic 500-image calibration set.

All embeddings are the final pooled [CLS] token after the last transformer block.

Usage:
    python metric_experiments/output_level_eval.py --model vitlarge --dataset imagenet-1k \
        --num-samples 500 --output metric_experiments/output_level_eval_vitlarge_imagenet1k.csv
"""

import argparse
import csv
import gc
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from transformers import AutoConfig, AutoModelForImageClassification

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from layers.layer_priority import (
    _load_dataset,
    _resolve_hf_id,
    _resolve_layers,
    _sublayer_modules,
    _linear_cka,
)
from toast.utils.dictionaries import MODEL2CONFIGS

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _collect_hidden_states(model, loader, num_layers, layers, max_samples):
    """
    Forward pass through unmodified model, collect hidden states at all layer boundaries.
    Returns dict: {layer_idx: [n_samples, n_tokens, d]} stored as float16 to save memory.
    Layer 0 = embedding output, layers 1..num_layers = block outputs.
    """
    collected = {i: [] for i in range(num_layers + 1)}
    seen = 0

    hooks = []
    layer_outputs = {}

    def _hook_output(store, idx):
        def hook(_module, _inp, out):
            store[idx] = out[0] if isinstance(out, tuple) else out
        return hook

    # Register hooks on MLP outputs (the residual delta added to the stream)
    for idx, layer in enumerate(layers):
        a_mod, m_mod = _sublayer_modules(layer)
        if m_mod is not None:
            hooks.append(m_mod.register_forward_hook(_hook_output(layer_outputs, idx)))

    try:
        with torch.no_grad():
            for images, _ in tqdm.tqdm(loader, desc="  Collecting hidden states"):
                images = images.to(device)
                layer_outputs.clear()
                outputs = model(images, output_hidden_states=True, return_dict=True)

                # Extract all hidden states (embedding + post-block outputs)
                hs = outputs.hidden_states
                for i in range(num_layers + 1):
                    collected[i].append(hs[i].detach().float().cpu().half())

                seen += images.shape[0]
                if seen >= max_samples:
                    break
    finally:
        for h in hooks:
            h.remove()

    # Concatenate and cap to max_samples
    result = {}
    for i in range(num_layers + 1):
        if collected[i]:
            result[i] = torch.cat(collected[i])[:max_samples]
    return result


def _fit_translator(h_in, h_out):
    """
    Fit linear translator: h_out_reconstructed = h_in @ W.
    h_in, h_out: [n_samples, n_tokens, d]
    Returns W: [d, d]
    """
    h_in_flat = h_in.reshape(-1, h_in.shape[-1]).to(device).float()
    h_out_flat = h_out.reshape(-1, h_out.shape[-1]).to(device).float()
    W = torch.linalg.lstsq(h_in_flat, h_out_flat).solution
    return W


def _kl_and_flip(logits_orig, logits_approx):
    """KL divergence and top-1 flip rate between two logit batches [n, num_classes]."""
    log_prob_approx = F.log_softmax(logits_approx, dim=-1)
    prob_orig = F.softmax(logits_orig, dim=-1)
    kl = F.kl_div(log_prob_approx, prob_orig, reduction="batchmean").item()

    top1_orig = logits_orig.argmax(dim=-1)
    top1_approx = logits_approx.argmax(dim=-1)
    flip_rate = (top1_orig != top1_approx).float().mean().item()

    return kl, flip_rate


def _embedding_metrics(E_orig, E_approx):
    """
    Compute cosine similarity, MSE, CKA, and KNN@5 overlap between final embeddings.
    E_orig, E_approx: [n_samples, embedding_dim], already on CPU
    """
    # Cosine similarity: mean over batch
    cosine = F.cosine_similarity(E_orig, E_approx, dim=-1).mean().item()

    # MSE: mean over batch
    mse = (E_orig - E_approx).pow(2).mean().item()

    # CKA: reuse layer_priority's implementation
    cka = _linear_cka(E_orig.to(device), E_approx.to(device))

    # KNN@5 overlap: pairwise cosine similarity, top-5 neighbors per image
    E_orig_norm = F.normalize(E_orig.to(device), p=2, dim=-1)
    E_approx_norm = F.normalize(E_approx.to(device), p=2, dim=-1)

    sim_orig = E_orig_norm @ E_orig_norm.T  # [n, n]
    sim_approx = E_approx_norm @ E_approx_norm.T  # [n, n]

    # Top-5 neighbors per image (excluding self at rank 0)
    knn_orig = torch.topk(sim_orig, k=6, dim=-1).indices[:, 1:]  # [n, 5]
    knn_approx = torch.topk(sim_approx, k=6, dim=-1).indices[:, 1:]  # [n, 5]

    # Compute overlap
    overlap = 0.0
    for i in range(knn_orig.shape[0]):
        set_orig = set(knn_orig[i].cpu().tolist())
        set_approx = set(knn_approx[i].cpu().tolist())
        overlap += len(set_orig & set_approx) / 5.0
    knn_overlap = overlap / knn_orig.shape[0]

    return cosine, mse, cka, knn_overlap


def run(model_arg, dataset_name, num_samples, output_csv):
    hf_id = _resolve_hf_id(model_arg)
    ds_key = dataset_name.lower().strip()

    print(f"\n{'=' * 60}")
    print(f"  Model   : {model_arg}  ({hf_id})")
    print(f"  Dataset : {ds_key}")
    print(f"  Samples : {num_samples}")
    print(f"  Device  : {device}")
    print(f"{'=' * 60}\n")

    # Load model
    print("[1/5] Loading model...")
    enc_config = AutoConfig.from_pretrained(
        hf_id, output_hidden_states=True, return_dict=True
    )
    model = AutoModelForImageClassification.from_pretrained(hf_id, config=enc_config)
    model.eval().to(device)

    # Extract encoder
    if hasattr(model, "vit"):
        encoder = model.vit
    elif hasattr(model, "deit"):
        encoder = model.deit
    else:
        raise ValueError(f"Could not locate encoder submodule in {hf_id}")

    num_layers = encoder.config.num_hidden_layers
    layers = _resolve_layers(encoder, hf_id)
    print(f"      {num_layers} layers")

    # Load dataset
    print("[2/5] Loading dataset...")
    loader, actual_n = _load_dataset(ds_key, num_samples, hf_id)
    print(f"      Using {actual_n} samples (deterministic order, shuffle=False)")

    # Collect hidden states from original model
    print("[3/5] Collecting hidden states from original model...")
    hidden_states = _collect_hidden_states(model, loader, num_layers, layers, actual_n)
    print(f"      Collected layers 0..{num_layers}")

    # Forward original model once to get logits and final embeddings
    print("[4/5] Computing original model outputs...")
    logits_orig_list = []
    embeddings_orig_list = []

    loader, _ = _load_dataset(ds_key, num_samples, hf_id)
    with torch.no_grad():
        for images, _ in tqdm.tqdm(loader, desc="  Orig forward"):
            images = images.to(device)
            outputs = model(images, output_hidden_states=True, return_dict=True)
            # Logits from classifier head
            logits_orig_list.append(outputs.logits.cpu())
            # Final embedding: [CLS] token from last hidden state (after all transformer blocks)
            final_hidden = outputs.hidden_states[-1]  # [B, seq_len, D]
            pooled = final_hidden[:, 0, :].cpu()  # [CLS] token
            embeddings_orig_list.append(pooled)

    logits_orig = torch.cat(logits_orig_list, dim=0)  # [n, num_classes]
    embeddings_orig = torch.cat(embeddings_orig_list, dim=0)  # [n, embedding_dim]
    print(f"      Logits: {logits_orig.shape}, Embeddings: {embeddings_orig.shape}")

    # Evaluate single-layer skips: (0,1), (1,2), ..., (num_layers-2, num_layers-1)
    print(f"[5/5] Evaluating {num_layers - 1} single-layer skips...")
    results = []

    for skip_idx in range(num_layers - 1):
        skip_from = skip_idx
        skip_to = skip_idx + 1

        print(f"\n  Skip [{skip_from}, {skip_to})...")

        # Fit translator on collected hidden states
        h_in = hidden_states[skip_from].float()  # [n, seq_len, d]
        h_out = hidden_states[skip_to].float()  # [n, seq_len, d]
        W = _fit_translator(h_in, h_out)  # [d, d]

        # Local boundary CKA: how similar is this block's input to its own output, i.e.
        # CKA(hidden_states[skip_from], hidden_states[skip_to]) -- the same per-token,
        # per-boundary metric layer_priority.analyse() saves to *_cka.npy, computed here
        # directly on the same 500-image calibration set as every other metric in this
        # script (rather than reading a possibly stale/differently-sampled .npy file).
        # Distinct from cka_embed below, which compares the FULL model's final output to
        # the SKIP model's final output after propagating the approximation through every
        # remaining block -- local_cka only looks at the one block being removed.
        local_cka = _linear_cka(
            h_in.reshape(-1, h_in.shape[-1]).to(device),
            h_out.reshape(-1, h_out.shape[-1]).to(device),
        )

        # Forward skip model using precomputed hidden states
        # This ensures we use exact same embeddings as original model
        logits_approx_list = []
        embeddings_approx_list = []

        with torch.no_grad():
            for batch_idx in tqdm.tqdm(range(0, actual_n, 8), desc=f"    Skip forward", leave=False):
                batch_end = min(batch_idx + 8, actual_n)
                batch_size = batch_end - batch_idx

                # Get precomputed embeddings for this batch
                h_in_batch = h_in[batch_idx:batch_end].to(device).float()  # [B, seq_len, d]

                # Forward through blocks 0..skip_from using precomputed states
                if skip_from > 0:
                    last_h = hidden_states[skip_from][batch_idx:batch_end].to(device).float()
                else:
                    last_h = hidden_states[0][batch_idx:batch_end].to(device).float()

                # Apply translator to bridge gap
                # last_h is [B, seq_len, d]; W is [d, d]
                # Flatten batch and seq dims, apply translation, reshape back
                B_batch, seq_len_actual, d_dim = last_h.shape
                last_h_flat = last_h.reshape(-1, d_dim)  # [B*seq_len, d]
                last_h_approx_flat = last_h_flat @ W.T  # [B*seq_len, d]
                last_h_approx = last_h_approx_flat.reshape(B_batch, seq_len_actual, d_dim)

                # Forward through blocks skip_to..num_layers-1
                for i in range(skip_to, num_layers):
                    layer_out = encoder.encoder.layer[i](last_h_approx)
                    # Handle both tuple (BaseModelOutput) and direct tensor returns
                    last_h_approx = layer_out[0] if isinstance(layer_out, (tuple, list)) else layer_out

                # Apply layer norm if it exists
                if hasattr(encoder, "layernorm"):
                    last_h_approx = encoder.layernorm(last_h_approx)

                # Pool to get final embedding ([CLS] token)
                pooled_approx = last_h_approx[:, 0, :]

                # Get logits via classifier head
                logits_approx = model.classifier(pooled_approx)
                logits_approx_list.append(logits_approx.cpu())
                embeddings_approx_list.append(pooled_approx.cpu())

        logits_approx = torch.cat(logits_approx_list, dim=0)  # [n, num_classes]
        embeddings_approx = torch.cat(embeddings_approx_list, dim=0)  # [n, embedding_dim]

        # Compute metrics
        kl, flip_rate = _kl_and_flip(logits_orig, logits_approx)
        cosine, mse, cka, knn_overlap = _embedding_metrics(embeddings_orig, embeddings_approx)

        results.append({
            "skip_from": skip_from,
            "skip_to": skip_to,
            "kl_div": kl,
            "top1_flip_pct": flip_rate * 100,
            "knn_overlap@5": knn_overlap,
            "cosine_sim": cosine,
            "mse": mse,
            "cka_embed": cka,
            "local_cka": local_cka,
        })

        print(f"      KL={kl:.6f}, flip%={flip_rate*100:.2f}, KNN={knn_overlap:.4f}, "
              f"cosine={cosine:.4f}, MSE={mse:.6f}, CKA={cka:.4f}, local_CKA={local_cka:.4f}")

        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Write output CSV
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["skip_from", "skip_to", "kl_div", "top1_flip_pct", "knn_overlap@5",
                        "cosine_sim", "mse", "cka_embed", "local_cka"],
        )
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults written to: {output_path}")
    print(f"  {len(results)} rows ({num_layers - 1} single-layer skips for {num_layers}-layer model)")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Output-level skip-safety predictors")
    p.add_argument("--model", required=True, help="model alias (vitlarge) or HF id")
    p.add_argument("--dataset", required=True, help="dataset key (imagenet-1k, ...)")
    p.add_argument("--num-samples", type=int, default=500, help="calibration images (default: 500)")
    p.add_argument("--output", default="metric_experiments/output_level_eval.csv", help="output CSV path")
    args = p.parse_args()

    run(
        model_arg=args.model,
        dataset_name=args.dataset,
        num_samples=args.num_samples,
        output_csv=args.output,
    )
