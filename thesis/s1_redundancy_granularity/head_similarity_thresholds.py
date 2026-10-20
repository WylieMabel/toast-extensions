"""
Attention-head redundancy by linear CKA and angular distance, swept over thresholds.

Compares heads with the two representation metrics used for blocks in layer_priority.py
(this folder), in place of the earlier attention-map JSD (archive/s1_head_jsd/):

  - Linear CKA (Kornblith et al. 2019), feature-space form, as layer_priority._linear_cka.
  - Angular distance (Gromov et al. 2024): mean over tokens of arccos(per-token cosine) / pi,
    the same aggregation order as layer_priority.analyse.

What is compared is each head's write into the residual stream: its slice of the attention
context times its slice of attention.output.dense (times LayerScale lambda1 for DINOv2), so
all heads of a layer live in the same D-dim space, as the hidden states do for blocks.

CKA is exact over every token of every sample: the centered covariance of the context is
streamed per layer, and HSIC(a, b) = ||W_a C_ab W_b^T||_F^2 is formed from it at the end, so
no token cap (layer_priority's --token-cap) is needed.

Heads to keep per threshold use a greedy rule (_compute_keep_dicts): a later head is dropped if
its similarity to an earlier kept head is > threshold. Similarity is CKA for "cka" and
1 - angular distance for "angdist". Layers that keep every head are left out of head_dict,
and {} means nothing is pruned.

Outputs (in --output-dir):
  {model}_{dataset}_head_cka.npy, {model}_{dataset}_head_angdist.npy   [L, H, H]
  head_keep_{dataset}.csv     dataset,encoder,model,metric,threshold,total_heads_kept,head_dict
And, in --config-dir, one experiments CSV per model with the distinct head_dicts as rows
(baseline first), ready for run_pipeline_row_by_row.sh.

Usage:
    python thesis/s1_redundancy_granularity/head_similarity_thresholds.py \
        --models vitsmall deitsmall dinobase vitlarge --dataset imagenet-1k --num-samples 500
"""

import argparse
import ast
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from transformers import AutoModel

from layer_priority import _load_dataset, _resolve_hf_id, _resolve_layers

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SECTION_DIR = Path(__file__).resolve().parent
METRICS = ["cka", "angdist"]
DEFAULT_THRESHOLDS = [round(t / 100, 2) for t in range(5, 100, 5)]  # 0.05 .. 0.95
CONFIG_COLUMNS = ["dataset", "encoder", "skip", "mlp_skip", "attn_skip", "head_dict",
                  "skip_translator", "mlp_mode", "attn_mode", "fit_dataset"]


# ----------------------------------------------------------------------------- #
# Per-head residual write
# ----------------------------------------------------------------------------- #
def _head_out_weights(layer, num_heads):
    """[H, D, dh] map from each head's context slice to its residual-stream write."""
    W = layer.attention.output.dense.weight.detach().float()           # [D, D]
    D = W.shape[0]
    dh = D // num_heads
    W = W.reshape(D, num_heads, dh)                                    # column h*dh+k -> [:, h, k]
    if hasattr(layer, "layer_scale1"):                                 # DINOv2: delta = lambda1 * attn
        W = W * layer.layer_scale1.lambda1.detach().float()[:, None, None]
    return W.permute(1, 0, 2).contiguous()                             # [H, D, dh]


def _capture_context(store, idx):
    """Pre-hook on attention.output.dense: its input is the head-concatenated context."""
    def hook(_module, args):
        store[idx] = args[0]
    return hook


@torch.no_grad()
def analyse(model, loader, layers, num_heads):
    num_layers = len(layers)
    W_heads = [_head_out_weights(layer, num_heads).to(device) for layer in layers]
    D = W_heads[0].shape[1]

    # CKA: streamed sums of the context, per layer (float64 for the covariance).
    sum_x = torch.zeros((num_layers, D), dtype=torch.float64, device=device)
    sum_xx = torch.zeros((num_layers, D, D), dtype=torch.float64, device=device)
    # Angular distance: summed per-token arccos(cosine) between every head pair of a layer.
    sum_ang = torch.zeros((num_layers, num_heads, num_heads), dtype=torch.float64, device=device)
    n_tokens = 0

    ctx = {}
    hooks = [layer.attention.output.dense.register_forward_pre_hook(_capture_context(ctx, i))
             for i, layer in enumerate(layers)]
    try:
        for images, _ in tqdm.tqdm(loader, desc="  Analysing"):
            ctx.clear()
            model(pixel_values=images.to(device))
            for l in range(num_layers):
                x = ctx[l].reshape(-1, D).float()                       # [T, D]
                xd = x.double()
                sum_x[l] += xd.sum(0)
                sum_xx[l] += xd.t() @ xd

                heads = x.reshape(x.shape[0], num_heads, -1)            # [T, H, dh]
                out = torch.einsum("thk,hdk->htd", heads, W_heads[l])   # [H, T, D]
                U = F.normalize(out, dim=-1)
                cos = torch.einsum("itd,jtd->ijt", U, U).clamp(-1.0, 1.0)
                sum_ang[l] += torch.arccos(cos).sum(-1).double()
            n_tokens += x.shape[0]
    finally:
        for h in hooks:
            h.remove()

    angdist = (sum_ang / n_tokens / torch.pi).cpu().numpy()

    cka = np.zeros((num_layers, num_heads, num_heads))
    dh = D // num_heads
    for l in range(num_layers):
        mu = sum_x[l] / n_tokens
        C = sum_xx[l] - n_tokens * torch.outer(mu, mu)                 # centered X^T X
        W = W_heads[l].double()                                        # [H, D, dh]
        hsic = torch.zeros((num_heads, num_heads), dtype=torch.float64, device=device)
        for a in range(num_heads):
            for b in range(a, num_heads):
                C_ab = C[a * dh:(a + 1) * dh, b * dh:(b + 1) * dh]
                hsic[a, b] = hsic[b, a] = (W[a] @ C_ab @ W[b].t()).pow(2).sum()
        diag = hsic.diagonal().sqrt()
        cka[l] = (hsic / torch.outer(diag, diag).clamp_min(1e-12)).cpu().numpy()

    return {"cka": cka, "angdist": angdist}, n_tokens


# ----------------------------------------------------------------------------- #
# Thresholds -> head_dicts
# ----------------------------------------------------------------------------- #
def _compute_keep_dicts(similarity, num_layers, num_heads, thresholds):
    """
    For each threshold: within each layer, greedily mark the higher-index head
    redundant if its similarity with any already-kept head exceeds the threshold.
    Returns heads to KEEP (complement of removed) per layer.
    similarity shape: [L, H, H]
    """
    results = []
    all_heads = set(range(num_heads))

    for thresh in thresholds:
        head_dict = {}
        for l in range(num_layers):
            to_remove = set()
            for i in range(num_heads):
                # A head already scheduled for deletion cannot prune other heads.
                if i in to_remove:
                    continue
                for j in range(i + 1, num_heads):
                    if similarity[l][i][j] > thresh:
                        to_remove.add(j)
            head_dict[l] = sorted(all_heads - to_remove)
        results.append((thresh, head_dict))

    return results


def _similarity(metric, scores):
    return scores["cka"] if metric == "cka" else 1.0 - scores["angdist"]


def _pruned_only(head_dict, num_heads):
    """Drop layers that keep every head; HeadPrunedEncoder leaves unlisted layers intact."""
    return {l: keep for l, keep in head_dict.items() if len(keep) < num_heads}


def keep_rows(scores, num_layers, num_heads, thresholds):
    rows = []
    for metric in METRICS:
        for thresh, head_dict in _compute_keep_dicts(
            _similarity(metric, scores), num_layers, num_heads, thresholds
        ):
            rows.append({
                "metric": metric,
                "threshold": f"{thresh:.2f}",
                "total_heads_kept": sum(len(v) for v in head_dict.values()),
                "head_dict": str(_pruned_only(head_dict, num_heads)),
            })
    return rows


def _write_config(path, dataset, hf_id, head_dicts):
    with open(path, "w", newline="") as f:
        w = csv.writer(f, quoting=csv.QUOTE_ALL)
        w.writerow(CONFIG_COLUMNS)
        for hd in head_dicts:
            w.writerow([dataset, hf_id, "[]", "[]", "[]", hd, "identity", "identity", "identity", ""])


# ----------------------------------------------------------------------------- #
# Driver
# ----------------------------------------------------------------------------- #
def run_model(model_arg, dataset, num_samples, out_dir, force):
    hf_id = _resolve_hf_id(model_arg)
    stem = f"{model_arg.lower().replace('/', '_')}_{dataset.replace('-', '')}"
    paths = {m: out_dir / f"{stem}_head_{m}.npy" for m in METRICS}

    model = AutoModel.from_pretrained(hf_id).to(device).eval()
    layers = _resolve_layers(model, hf_id)
    num_layers, num_heads = len(layers), model.config.num_attention_heads

    print(f"\n{'=' * 60}\n  Model   : {model_arg} ({hf_id})  {num_layers} x {num_heads} heads"
          f"\n  Dataset : {dataset}  ({num_samples} samples)\n  Device  : {device}\n{'=' * 60}")

    if not force and all(p.exists() for p in paths.values()):
        print(f"  Scores already exist for {stem}; reusing them (pass --force to recompute).")
        scores = {m: np.load(p) for m, p in paths.items()}
    else:
        loader, n = _load_dataset(dataset, num_samples, hf_id)
        scores, n_tokens = analyse(model, loader, layers, num_heads)
        print(f"  {n} images, {n_tokens} tokens")
        for m, p in paths.items():
            np.save(p, scores[m])
            print(f"  {m:8s} -> {p}")

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return hf_id, num_layers, num_heads, scores


def main():
    p = argparse.ArgumentParser(description="Head redundancy by CKA / angular distance")
    p.add_argument("--models", nargs="+", default=["vitsmall", "deitsmall", "dinobase", "vitlarge"],
                   help="aliases from layer_priority.MODEL_ALIASES or HF ids")
    p.add_argument("--dataset", default="imagenet-1k", help="dataset key (default: imagenet-1k)")
    p.add_argument("--num-samples", type=int, default=500)
    p.add_argument("--thresholds", type=float, nargs="+", default=DEFAULT_THRESHOLDS)
    p.add_argument("--output-dir", default=str(SECTION_DIR / "results" / "head_similarity"))
    p.add_argument("--config-dir", default=str(SECTION_DIR / "configs"))
    p.add_argument("--force", action="store_true", help="recompute even if score files exist")
    args = p.parse_args()

    out_dir, config_dir = Path(args.output_dir), Path(args.config_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)
    ds_tag = args.dataset.replace("-", "")
    keep_csv = out_dir / f"head_keep_{ds_tag}.csv"

    all_rows = []
    for model_arg in args.models:
        hf_id, num_layers, num_heads, scores = run_model(
            model_arg, args.dataset, args.num_samples, out_dir, args.force
        )
        rows = [{"dataset": args.dataset, "encoder": hf_id, "model": model_arg, **r}
                for r in keep_rows(scores, num_layers, num_heads, args.thresholds)]
        all_rows.extend(rows)

        # Rewritten after every model so a later failure keeps the finished ones.
        with open(keep_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(all_rows[0]))
            w.writeheader()
            w.writerows(all_rows)
        print(f"  head_dicts -> {keep_csv}")

        # Baseline first, then the distinct head_dicts from least to most pruned.
        n_pruned = lambda hd: sum(num_heads - len(v) for v in ast.literal_eval(hd).values())
        distinct = ["{}"] + sorted({r["head_dict"] for r in rows} - {"{}"}, key=n_pruned)
        config = config_dir / f"experiments_heads_{model_arg.lower().replace('/', '_')}_{ds_tag}.csv"
        _write_config(config, args.dataset, hf_id, distinct)
        print(f"  config ({len(distinct)} rows) -> {config}")


if __name__ == "__main__":
    main()
