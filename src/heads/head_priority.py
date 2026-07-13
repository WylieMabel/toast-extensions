"""
Compute pairwise Jensen-Shannon Divergence (JSD) between all attention heads,
then output a similarity matrix (.npy) and a CSV of heads-to-keep per layer
at each similarity threshold.

Usage:
    python head_priority.py --model deitsmall --dataset cifar100
    python head_priority.py --model vitlarge --dataset imagenet1k --num-samples 2000
"""

import argparse
import csv
import gc
from pathlib import Path

import numpy as np
import torch
import tqdm
from datasets import load_from_disk
from torch.utils.data import DataLoader
from transformers import (
    AutoConfig,
    AutoImageProcessor,
    AutoModel,
)

MODEL_REGISTRY = {
    "deitsmall": {
        "hf_id": "facebook/deit-small-patch16-224",
        "num_layers": 12,
        "num_heads": 6,
    },
    "dinobase": {
        "hf_id": "facebook/dinov2-base",
        "num_layers": 12,
        "num_heads": 12,
    },
    "vitlarge": {
        "hf_id": "google/vit-large-patch16-224",
        "num_layers": 24,
        "num_heads": 16,
    },
}

DATASET_REGISTRY = {"mnist", "cifar100", "imagenet1k"}

DATASET_LOCAL_PATH = {
    "mnist":      "/cluster/customapps/biomed/vogtlab/users/mwylie/toast/mnist_clean",
    "cifar100":   "/cluster/customapps/biomed/vogtlab/users/mwylie/toast/cifar100_clean",
    "imagenet1k": "/cluster/customapps/biomed/vogtlab/users/mwylie/toast/imagenet1k_clean",
}

DATASET_IMAGE_COL = {
    "mnist":      "image",
    "cifar100":   "img",
    "imagenet1k": "image",
}

DATASET_LABEL_COL = {
    "mnist":      "label",
    "cifar100":   "fine_label",
    "imagenet1k": "label",
}


def _collate(samples, processor, image_col, label_col):
    images = [s[image_col].convert("RGB") for s in samples]
    pixel_values = torch.cat(
        [processor(images=img, return_tensors="pt")["pixel_values"] for img in images],
        dim=0,
    )
    labels = torch.tensor([s[label_col] for s in samples])
    return pixel_values, labels


def _load_dataset(dataset_name: str, num_samples: int, model_hf_id: str):
    raw = load_from_disk(DATASET_LOCAL_PATH[dataset_name])
    split = "validation" if "validation" in raw else "test"
    ds = raw[split]

    n = min(num_samples, len(ds))
    ds = ds.select(range(n))

    processor = AutoImageProcessor.from_pretrained(model_hf_id)
    collate_fn = lambda batch: _collate(
        batch, processor, DATASET_IMAGE_COL[dataset_name], DATASET_LABEL_COL[dataset_name]
    )

    return DataLoader(ds, batch_size=8, shuffle=False, num_workers=1, collate_fn=collate_fn), n


def _load_model(model_key: str, device: torch.device):
    cfg = MODEL_REGISTRY[model_key]
    enc_config = AutoConfig.from_pretrained(cfg["hf_id"], output_attentions=True, output_hidden_states=True, return_dict=True)
    model = AutoModel.from_pretrained(cfg["hf_id"], config=enc_config, attn_implementation="eager")
    model.to(device).eval()
    return model, cfg["num_layers"], cfg["num_heads"]


def _accumulate_jsd(model, loader, num_layers, num_heads, device):
    # Only compute within-layer similarities — cross-layer pairs are never used by _compute_keep_dicts.
    # This reduces comparisons from TH^2 to L * H^2 (e.g. 24x fewer for vitlarge).
    total_jsd = torch.zeros((num_layers, num_heads, num_heads), dtype=torch.float64)
    n_images = 0

    with torch.no_grad():
        for images, _ in tqdm.tqdm(loader, desc="  Batches"):
            images = images.to(device)
            bs = images.shape[0]
            outputs = model(images)

            # Stack per-layer attention tensors and keep shape [bs, L, H, S, S].
            # Each row heads[l, h, q, :] is one softmax over key tokens — sums to 1, no normalization needed.
            stacked = torch.stack(outputs.attentions)              # [L, bs, H, S, S]
            attn = stacked.permute(1, 0, 2, 3, 4)                 # [bs, L, H, S, S]
            del outputs, stacked
            torch.cuda.empty_cache()

            for i in range(bs):
                for l in range(num_layers):
                    heads_c = torch.clamp(attn[i, l], min=1e-10)  # [H, S, S]
                    log_h = torch.log2(heads_c)                    # [H, S, S]

                    # All H×H pairs at once: [H, H, S, S] is small enough to fit on GPU
                    # (e.g. vitlarge: [16, 16, 197, 197] ≈ 400 MB).
                    P = heads_c.unsqueeze(1)                           # [H, 1, S, S]
                    Q = heads_c.unsqueeze(0)                           # [1, H, S, S]
                    M = torch.clamp(0.5 * (P + Q), min=1e-10)         # [H, H, S, S] mixture
                    log_M = torch.log2(M)

                    # KL per query token (sum over keys, dim=-1), then average across query tokens.
                    # Result is mean JSD between each head pair for this image, in [0, 1].
                    kl_pm = torch.sum(P * (log_h.unsqueeze(1) - log_M), dim=-1)  # [H, H, S]
                    kl_qm = torch.sum(Q * (log_h.unsqueeze(0) - log_M), dim=-1)  # [H, H, S]
                    total_jsd[l] += (0.5 * kl_pm + 0.5 * kl_qm).mean(dim=-1).cpu().double()  # [H, H]

                n_images += 1

            del attn
            torch.cuda.empty_cache()
            gc.collect()

    # similarity[l, i, j] = 1 - avg JSD(head_i, head_j in layer l) across all images, in [0, 1].
    # 1 = identical attention patterns, 0 = maximally different.
    similarity = (1.0 - total_jsd / n_images).numpy()
    return similarity, n_images


def _compute_keep_dicts(similarity, num_layers, num_heads, thresholds):
    """
    For each threshold: within each layer, greedily mark the higher-index head
    redundant if its similarity with any already-kept head exceeds the threshold.
    Returns heads to KEEP (complement of removed) per layer.
    similarity shape: [L, H, H]
    """
    # might tweak to sort by similarity but for now it's fine
    results = []
    all_heads = set(range(num_heads))

    for thresh in thresholds:
        head_dict = {}
        for l in range(num_layers):
            to_remove = set()
            for i in range(num_heads):
                # If this head is already scheduled for deletion,
                # it cannot serve as a valid baseline to prune other heads.
                if i in to_remove:
                    continue
                for j in range(i + 1, num_heads):
                    if similarity[l][i][j] > thresh:
                        to_remove.add(j)
            head_dict[l] = sorted(all_heads - to_remove)
        results.append((thresh, head_dict))

    return results


def _save_csv(threshold_results, num_layers, out_path: Path):
    layer_cols = [f"layer_{l}_keep" for l in range(num_layers)]
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["threshold"] + layer_cols)
        for thresh, head_dict in threshold_results:
            row = [f"{thresh:.2f}"] + [
                " ".join(map(str, head_dict[l])) for l in range(num_layers)
            ]
            writer.writerow(row)
    print(f"  CSV saved -> {out_path}")


def run(model_name: str, dataset_name: str, num_samples: int, thresholds: list, output_dir: str):
    model_key = model_name.lower().strip()
    ds_key = dataset_name.lower().strip()

    if model_key not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{model_name}'. Choose from: {list(MODEL_REGISTRY)}")
    if ds_key not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset '{dataset_name}'. Choose from: {DATASET_REGISTRY}")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{model_key}_{ds_key}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  Model   : {model_key}  ({MODEL_REGISTRY[model_key]['hf_id']})")
    print(f"  Dataset : {ds_key}")
    print(f"  Samples : {num_samples}")
    print(f"  Device  : {device}")
    print(f"{'='*60}\n")

    print("[1/4] Loading model...")
    model, num_layers, num_heads = _load_model(model_key, device)
    print(f"      {num_layers} layers x {num_heads} heads = {num_layers * num_heads} total heads")

    print("[2/4] Loading dataset...")
    loader, actual_n = _load_dataset(ds_key, num_samples, MODEL_REGISTRY[model_key]["hf_id"])
    print(f"      Using {actual_n} samples")

    print("[3/4] Accumulating pairwise JSD...")
    similarity, n_images = _accumulate_jsd(model, loader, num_layers, num_heads, device)

    print("[4/4] Saving outputs...")
    npy_path = out_dir / f"{stem}_similarity.npy"
    np.save(npy_path, similarity)
    print(f"  Similarity matrix saved -> {npy_path}")

    threshold_results = _compute_keep_dicts(similarity, num_layers, num_heads, thresholds)
    _save_csv(threshold_results, num_layers, out_dir / f"{stem}.csv")

    print(f"\nDone. Outputs written to: {out_dir.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Attention head JSD similarity analysis")
    parser.add_argument(
        "--model", required=True,
        choices=list(MODEL_REGISTRY),
        help="Model key: deitsmall | dinobase | vitlarge",
    )
    parser.add_argument(
        "--dataset", required=True,
        choices=list(DATASET_REGISTRY),
        help="Dataset key: mnist | cifar100 | imagenet1k",
    )
    parser.add_argument(
        "--num-samples", type=int, default=5000,
        help="Number of images to evaluate (default: 5000)",
    )
    parser.add_argument(
        "--thresholds", type=float, nargs="+",
        default=[round(t / 100, 2) for t in range(5, 100, 5)],
        help="Similarity thresholds (default: 0.05 to 0.95 in steps of 0.05)",
    )
    parser.add_argument(
        "--output-dir", default="./outputs",
        help="Directory for output files (default: ./outputs)",
    )
    args = parser.parse_args()

    run(
        model_name=args.model,
        dataset_name=args.dataset,
        num_samples=args.num_samples,
        thresholds=args.thresholds,
        output_dir=args.output_dir,
    )
