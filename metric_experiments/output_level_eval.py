"""
Output-level skip-safety predictors, in four tiers:
  - local (at the skip boundary): local_cka, angular_dist -- both computed on ALL tokens
    (pooled across seq_len), plus toast_linear_error -- TOAST's (Cannistraci et al.,
    "TOAST: Transformer Optimization using Adaptive and Simple Transformations") OWN
    block-selection criterion, a normalized [CLS]-token-only linear reconstruction residual
    (see the paper's "Patterns of Similarity between Transformer Blocks" section). This is a
    different basis (CLS-only) and a different formula (normalized lstsq residual, not
    cosine-arccos or CKA) from local_cka/angular_dist, and uses its own separate lstsq fit,
    not the all-token translator W used for the actual skip-forward bridge below.
  - global/headless (final embeddings, no head needed): cka_embed, cosine_sim, mse,
    knn_overlap@5, knn_overlap@10, participation_ratio_change
  - task/prediction (frozen original classification head): kl_div, top1_flip_pct, entropy_delta
  - task/probe (LogME/LEEP-style cheap linear classifier fit on TRAIN-split images, never
    test/validation): probe_frozen_accuracy, probe_refit_accuracy, probe_baseline_accuracy

Defaults to single-layer adjacent skips (num_layers-1 rows). Pass --spans-csv to instead
evaluate every contiguous single/multi-block span found for this (model, dataset) in a
skipping_heads/calculate_accuracies.py-style summary CSV -- guarantees the candidate list here
always matches whatever configs the accuracy ground truth was actually computed for. Disjoint
(non-contiguous) spans in that CSV are skipped: this script's forward pass only handles one
contiguous span per candidate.

Uses the original classification head (no retraining) and a deterministic 500-image
calibration set. All embeddings are the final pooled [CLS] token after the last transformer
block (or, for a skip, after whatever blocks remain post-skip).

Layer indexing convention -- this trips people up, worked example for a 4-block model:
    hidden_states is boundary-indexed: H0=embedding output, H1=out(block 0), H2=out(block 1),
    H3=out(block 2), H4=out(block 3). A span (skip_from, skip_to), as read from any
    approx_layer/skip/spans-csv column in this repo, follows the SkipModel convention (see
    src/configs/make_skip_grid.py's docstring and src/toast/modules/module.py's
    SkipModel.filter_layers): it KEEPS block skip_from and DROPS blocks skip_from+1 ..
    skip_to inclusive. So for the label (0, 1): block 0 is kept, block 1 is dropped, and
    the ground-truth accuracy row for that label was produced by bridging out(block 0) = H1
    to out(block 1) = H2 with a translator, then resuming at block 2. Every h_in/h_out and
    hidden_states lookup in this file therefore uses hidden_states[skip_from + 1] and
    hidden_states[skip_to + 1] -- NOT hidden_states[skip_from]/[skip_to] directly, which
    would silently evaluate block skip_from's removal instead of the blocks the CSV label
    actually describes.

Usage:
    python metric_experiments/output_level_eval.py --model vitlarge --dataset imagenet-1k \
        --num-samples 500 --output metric_experiments/output_level_eval_vitlarge_imagenet1k.csv

    # multi-span candidates, matching an accuracy sweep's own config list:
    python metric_experiments/output_level_eval.py --model vitlarge --dataset imagenet-1k \
        --spans-csv results/accuracies_results_vitlarge_deitsmall_contiguous_multiskip.csv \
        --output metric_experiments/output_level_eval_vitlarge_multiskip.csv
"""

import argparse
import ast
import csv
import gc
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from transformers import AutoConfig, AutoModel, AutoModelForImageClassification

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

# Datasets where the checkpoint's bundled classifier head is meaningful (matches
# train_skipped_full.py's ORIG_HEAD_DATASETS convention). Every model here ships an
# ImageNet-1k head; on any other dataset there is no correct head to compare against, so
# kl_div/top1_flip_pct/entropy_delta are skipped entirely rather than computed against the
# wrong task's logits.
ORIG_HEAD_DATASETS = {"imagenet-1k"}

# Backbones that ship NO task classifier (self-supervised / feature encoders loaded as bare
# AutoModel). Even on an ORIG_HEAD_DATASETS dataset there is no head to read logits from --
# AutoModelForImageClassification would bolt on a random-init classifier and the encoder
# lookup below (model.vit / model.deit) would fail -- so force the headless path.
HEADLESS_MODELS = {
    "facebook/dinov2-small",
    "facebook/dinov2-base",
    "microsoft/rad-dino",
    "microsoft/rad-dino-maira-2",
    "owkin/phikon-v2",
}


def _load_spans_from_csv(spans_csv, hf_id, ds_key):
    """
    Read a skipping_heads/calculate_accuracies.py-style summary CSV (columns: model, dataset,
    approx_layer, ...) and return the sorted, deduped list of (skip_from, skip_to) contiguous
    spans it contains for this (model, dataset) -- baseline ("[]") and multi-span/disjoint
    rows (more than one tuple in approx_layer, e.g. "[(1, 2), (4, 5)]") are skipped, since this
    script's forward pass only handles ONE contiguous span per candidate.
    """
    import pandas as pd

    df = pd.read_csv(spans_csv)
    sub = df[
        (df["model"].str.lower().str.replace("/", "_") == hf_id.lower().replace("/", "_"))
        & (df["dataset"].str.lower() == ds_key.lower())
    ]

    spans = set()
    skipped_disjoint = 0
    for raw in sub["approx_layer"]:
        parsed = ast.literal_eval(raw)
        if not parsed:
            continue  # baseline row
        if len(parsed) > 1:
            skipped_disjoint += 1
            continue
        start, end = parsed[0]
        spans.add((start, end))

    if skipped_disjoint:
        print(f"      Skipped {skipped_disjoint} disjoint/multi-span row(s) "
              f"(not supported by this script's single-span forward pass).")

    return sorted(spans)


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


def _collect_hidden_states_with_labels(model, loader, num_layers, layers, max_samples):
    """
    Same as _collect_hidden_states, but also returns the per-image labels alongside the
    hidden states -- needed for the linear-probe calibration set (labels are discarded
    everywhere else in this script since the similarity metrics never touch them).
    """
    collected = {i: [] for i in range(num_layers + 1)}
    labels_list = []
    seen = 0

    hooks = []
    layer_outputs = {}

    def _hook_output(store, idx):
        def hook(_module, _inp, out):
            store[idx] = out[0] if isinstance(out, tuple) else out
        return hook

    for idx, layer in enumerate(layers):
        a_mod, m_mod = _sublayer_modules(layer)
        if m_mod is not None:
            hooks.append(m_mod.register_forward_hook(_hook_output(layer_outputs, idx)))

    try:
        with torch.no_grad():
            for images, labels in tqdm.tqdm(loader, desc="  Collecting probe hidden states"):
                images = images.to(device)
                layer_outputs.clear()
                outputs = model(images, output_hidden_states=True, return_dict=True)

                hs = outputs.hidden_states
                for i in range(num_layers + 1):
                    collected[i].append(hs[i].detach().float().cpu().half())
                labels_list.append(labels)

                seen += images.shape[0]
                if seen >= max_samples:
                    break
    finally:
        for h in hooks:
            h.remove()

    result = {}
    for i in range(num_layers + 1):
        if collected[i]:
            result[i] = torch.cat(collected[i])[:max_samples]
    labels = torch.cat(labels_list)[:max_samples]
    return result, labels


def _split_probe_indices(labels, test_size=0.2, random_state=0):
    """
    One stratified train_test_split of the probe's own indices into a fit half and a score
    half, reused for every span so all probe numbers for a (model, dataset) run are
    comparable. This is a split WITHIN the train-split calibration pool (fit-half vs
    score-half) -- it never touches the real test/validation split.

    Falls back to a non-stratified split (with a printed warning) when stratification isn't
    possible, e.g. a class with fewer than 2 members.
    """
    n = labels.shape[0]
    indices = np.arange(n)
    labels_np = labels.numpy()
    try:
        fit_idx, score_idx = train_test_split(
            indices, test_size=test_size, stratify=labels_np, random_state=random_state
        )
    except ValueError as e:
        print(f"      WARNING: stratified probe split failed ({e}); falling back to "
              f"non-stratified split.")
        fit_idx, score_idx = train_test_split(
            indices, test_size=test_size, random_state=random_state
        )
    return fit_idx, score_idx


def _fit_probe(embeddings, labels, fit_idx, score_idx):
    """
    Fit one linear probe (StandardScaler + multinomial LogisticRegression) on
    embeddings[fit_idx]/labels[fit_idx], score it on embeddings[score_idx]/labels[score_idx].

    This is a closed-form-cheap adaptation of the LogME/LEEP family of transferability
    estimators (Nguyen et al. 2020; You et al. 2021) to skip-span selection: a fitted linear
    head's held-out accuracy stands in for "how good would the retrained (expensive) head
    be", without running the real fine-tuning loop. Not a novel technique -- just the
    standard cheap-linear-probe proxy, applied here per candidate span.

    embeddings: [n, d] float tensor/array. labels: [n] int tensor/array. Returns
    (fitted_probe, accuracy) where accuracy is a float in [0, 1].
    """
    embeddings = np.asarray(embeddings)
    labels = np.asarray(labels)
    probe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000)),
    ])
    probe.fit(embeddings[fit_idx], labels[fit_idx])
    accuracy = probe.score(embeddings[score_idx], labels[score_idx])
    return probe, accuracy


def _apply_skip_forward(hidden_states, actual_n, skip_from, skip_to, W, encoder, num_layers,
                         model=None, has_head=False, batch_size=8):
    """
    Bridge hidden_states[skip_from + 1] (output of the last KEPT block, skip_from) ->
    translator W -> remaining blocks skip_to+1..num_layers-1 -> final embedding (+ logits if
    has_head). (skip_from, skip_to)
    is the repo-wide SkipModel convention: KEEP block skip_from, DROP blocks skip_from+1 ..
    skip_to inclusive (see src/toast/modules/module.py's SkipModel.filter_layers and
    src/configs/make_skip_grid.py's docstring) -- so the blocks that actually resume
    afterwards start at index skip_to + 1, not skip_to. Shared by the existing
    test/validation calibration path and the linear-probe (train-split) calibration path so
    both apply the SAME per-span translator W to their own precomputed hidden states.
    """
    logits_approx_list = []
    embeddings_approx_list = []

    with torch.no_grad():
        for batch_idx in range(0, actual_n, batch_size):
            batch_end = min(batch_idx + batch_size, actual_n)

            last_h = hidden_states[skip_from + 1][batch_idx:batch_end].to(device).float()

            B_batch, seq_len_actual, d_dim = last_h.shape
            last_h_flat = last_h.reshape(-1, d_dim)
            last_h_approx_flat = last_h_flat @ W.T
            last_h_approx = last_h_approx_flat.reshape(B_batch, seq_len_actual, d_dim)

            for i in range(skip_to + 1, num_layers):
                layer_out = encoder.encoder.layer[i](last_h_approx)
                last_h_approx = layer_out[0] if isinstance(layer_out, (tuple, list)) else layer_out

            if hasattr(encoder, "layernorm"):
                last_h_approx = encoder.layernorm(last_h_approx)

            pooled_approx = last_h_approx[:, 0, :]

            if has_head:
                logits_approx = model.classifier(pooled_approx)
                logits_approx_list.append(logits_approx.cpu())
            embeddings_approx_list.append(pooled_approx.cpu())

    logits_approx = torch.cat(logits_approx_list, dim=0) if has_head else None
    embeddings_approx = torch.cat(embeddings_approx_list, dim=0)
    return logits_approx, embeddings_approx


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


def _angular_distance(h_in, h_out):
    """
    Local boundary angular distance (Gromov et al. 2024): (1/pi) * arccos(mean token cosine).
    h_in, h_out: [n_samples, seq_len, d]. Matches layer_priority.analyse()'s angdist formula
    exactly (arccos applied to the mean cosine, not mean of per-token arccos values) so the
    two stay comparable if ever cross-checked against a *_angdist.npy file.
    """
    a = F.normalize(h_in.reshape(-1, h_in.shape[-1]), dim=-1)
    b = F.normalize(h_out.reshape(-1, h_out.shape[-1]), dim=-1)
    mean_cos = (a * b).sum(dim=-1).mean().clamp(-1.0, 1.0)
    return (torch.arccos(mean_cos) / torch.pi).item()


def _toast_linear_approx_error(h_in, h_out):
    """
    TOAST's (Cannistraci et al.) own block-selection metric -- normalized residual of a
    CLS-only linear reconstruction:  eps(s, e) = ||H_e - H_s @ W*||_F / ||H_e||_F, where
    W* = argmin_W ||H_e - H_s @ W||_F^2 (least squares). The paper computes block similarity
    using ONLY the [CLS] token ("providing a consistent and semantically aligned basis for
    comparing blocks"), unlike local_cka/angular_dist above which pool all tokens. This fits
    its OWN lstsq map on the CLS token only -- separate from the all-token translator W used
    for the actual skip-forward bridge elsewhere in this file, which intentionally matches
    TOAST's deployed method, not its selection metric.

    Solved on CPU with the rank-revealing 'gelsd' driver rather than on `device`: for
    skip_from=0 (embedding output, before any transformer block), the [CLS] token is just
    the model's learned CLS embedding parameter, essentially IDENTICAL across every sample in
    the batch (no image content has reached it yet) -- so h_in_cls there is rank-deficient
    (rank ~1). torch.linalg.lstsq on CUDA only supports the 'gels' driver, which assumes full
    rank and silently returns NaN on rank-deficient input; 'gelsd' handles rank-deficient A
    correctly (minimum-norm SVD solution) and is CPU-only. h_in_cls/h_out_cls are tiny
    ([n_samples, d]), so this costs nothing measurable even on the full 500-image set.
    h_in, h_out: [n_samples, seq_len, d]
    """
    h_in_cls = h_in[:, 0, :].cpu().float()  # [n, d]
    h_out_cls = h_out[:, 0, :].cpu().float()  # [n, d]
    W_cls = torch.linalg.lstsq(h_in_cls, h_out_cls, driver="gelsd").solution
    pred = h_in_cls @ W_cls
    eps = torch.linalg.norm(h_out_cls - pred) / torch.linalg.norm(h_out_cls).clamp_min(1e-12)
    return eps.item()


def _kl_and_flip_and_entropy(logits_orig, logits_approx):
    """KL divergence, top-1 flip rate, and entropy delta between two logit batches [n, num_classes]."""
    log_prob_approx = F.log_softmax(logits_approx, dim=-1)
    prob_orig = F.softmax(logits_orig, dim=-1)
    prob_approx = F.softmax(logits_approx, dim=-1)
    kl = F.kl_div(log_prob_approx, prob_orig, reduction="batchmean").item()

    top1_orig = logits_orig.argmax(dim=-1)
    top1_approx = logits_approx.argmax(dim=-1)
    flip_rate = (top1_orig != top1_approx).float().mean().item()

    # Entropy of the predicted distribution, before vs after. Reported as mean ABSOLUTE
    # difference per image rather than signed: a skip can damage a prediction by making it
    # falsely overconfident (entropy drops) or newly uncertain (entropy rises) -- both are
    # "the confidence profile changed", so unsigned magnitude is the safer default rather
    # than assuming damage always pushes entropy in one direction.
    entropy_orig = -(prob_orig * torch.log(prob_orig.clamp_min(1e-12))).sum(dim=-1)
    entropy_approx = -(prob_approx * torch.log(prob_approx.clamp_min(1e-12))).sum(dim=-1)
    entropy_delta = (entropy_approx - entropy_orig).abs().mean().item()

    return kl, flip_rate, entropy_delta


def _participation_ratio(E):
    """
    Participation ratio (effective dimensionality) of an embedding set: PR = (sum(eig))^2 /
    sum(eig^2), from the eigenvalues of the (centered) feature covariance. Ranges [1, d];
    PR=1 means all variance is on one axis, PR=d means variance is spread evenly across all
    axes. E: [n_samples, d] on device.
    """
    Ec = E - E.mean(dim=0, keepdim=True)
    cov = (Ec.T @ Ec) / max(E.shape[0] - 1, 1)
    eigvals = torch.linalg.eigvalsh(cov).clamp_min(0)
    pr = eigvals.sum().pow(2) / eigvals.pow(2).sum().clamp_min(1e-12)
    return pr.item()


def _knn_overlap(E_orig_norm, E_approx_norm, k):
    """Fraction of each image's top-k nearest neighbors preserved between two embedding sets.
    E_*_norm: [n, d] L2-normalized, on device."""
    sim_orig = E_orig_norm @ E_orig_norm.T  # [n, n]
    sim_approx = E_approx_norm @ E_approx_norm.T  # [n, n]

    # Top-k neighbors per image (excluding self at rank 0)
    knn_orig = torch.topk(sim_orig, k=k + 1, dim=-1).indices[:, 1:]  # [n, k]
    knn_approx = torch.topk(sim_approx, k=k + 1, dim=-1).indices[:, 1:]  # [n, k]

    overlap = 0.0
    for i in range(knn_orig.shape[0]):
        set_orig = set(knn_orig[i].cpu().tolist())
        set_approx = set(knn_approx[i].cpu().tolist())
        overlap += len(set_orig & set_approx) / k
    return overlap / knn_orig.shape[0]


def _embedding_metrics(E_orig, E_approx):
    """
    Compute cosine similarity, MSE, CKA, KNN@5/@10 overlap, and participation-ratio change
    between final embeddings. E_orig, E_approx: [n_samples, embedding_dim], already on CPU.
    """
    # Cosine similarity: mean over batch
    cosine = F.cosine_similarity(E_orig, E_approx, dim=-1).mean().item()

    # MSE: mean over batch
    mse = (E_orig - E_approx).pow(2).mean().item()

    # CKA: reuse layer_priority's implementation
    E_orig_dev = E_orig.to(device)
    E_approx_dev = E_approx.to(device)
    cka = _linear_cka(E_orig_dev, E_approx_dev)

    # Participation ratio change: unsigned magnitude, like MSE/entropy_delta above -- a
    # skip could collapse effective dimensionality (PR drops) or spuriously inflate it
    # (PR rises, e.g. if the approximation injects noise along new directions); both are
    # "the representation's effective dimensionality changed", so report |delta|.
    pr_orig = _participation_ratio(E_orig_dev)
    pr_approx = _participation_ratio(E_approx_dev)
    pr_change = abs(pr_approx - pr_orig)

    # KNN overlap @5 and @10: pairwise cosine similarity, top-k neighbors per image
    E_orig_norm = F.normalize(E_orig_dev, p=2, dim=-1)
    E_approx_norm = F.normalize(E_approx_dev, p=2, dim=-1)
    knn_overlap_5 = _knn_overlap(E_orig_norm, E_approx_norm, k=5)
    knn_overlap_10 = _knn_overlap(E_orig_norm, E_approx_norm, k=10)

    return cosine, mse, cka, knn_overlap_5, knn_overlap_10, pr_change


def run(model_arg, dataset_name, num_samples, output_csv, spans_csv=None, probe_num_samples=500):
    hf_id = _resolve_hf_id(model_arg)
    ds_key = dataset_name.lower().strip()

    print(f"\n{'=' * 60}")
    print(f"  Model   : {model_arg}  ({hf_id})")
    print(f"  Dataset : {ds_key}")
    print(f"  Samples : {num_samples} (test/validation calibration), "
          f"{probe_num_samples} (train-split probe calibration)")
    print(f"  Device  : {device}")
    print(f"{'=' * 60}\n")

    has_head = ds_key in ORIG_HEAD_DATASETS and hf_id not in HEADLESS_MODELS
    if not has_head:
        print(f"      Note: no original head for dataset '{ds_key}' -- skipping "
              f"kl_div/top1_flip_pct/entropy_delta (embedding-level metrics only).")

    # Load model
    print("[1/5] Loading model...")
    enc_config = AutoConfig.from_pretrained(
        hf_id, output_hidden_states=True, return_dict=True
    )
    if has_head:
        model = AutoModelForImageClassification.from_pretrained(hf_id, config=enc_config)
        model.eval().to(device)
        # Extract encoder
        if hasattr(model, "vit"):
            encoder = model.vit
        elif hasattr(model, "deit"):
            encoder = model.deit
        else:
            raise ValueError(f"Could not locate encoder submodule in {hf_id}")
    else:
        # Headless: load the bare encoder directly, no classifier to mismatch against.
        model = AutoModel.from_pretrained(hf_id, config=enc_config)
        model.eval().to(device)
        encoder = model

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
            # Logits from classifier head (only meaningful when the checkpoint's head
            # matches this dataset)
            if has_head:
                logits_orig_list.append(outputs.logits.cpu())
            # Final embedding: [CLS] token from last hidden state (after all transformer blocks).
            # outputs.hidden_states[-1] is the RAW last-block output -- HF's ViTModel/DeiTModel
            # (etc.) apply their own final `self.layernorm` to sequence_output AFTER self.encoder
            # returns, so it's never reflected in the hidden_states tuple. _apply_skip_forward's
            # embeddings_approx DOES apply that layernorm (see its own `encoder.layernorm` call
            # below), so it has to be applied here too or embeddings_orig/embeddings_approx would
            # be compared in different normalization states -- biasing cosine_sim/mse/cka_embed/
            # knn_overlap/participation_ratio_change even for a perfectly lossless skip.
            final_hidden = outputs.hidden_states[-1]  # [B, seq_len, D]
            pooled = final_hidden[:, 0, :]  # [CLS] token
            if hasattr(encoder, "layernorm"):
                pooled = encoder.layernorm(pooled)
            pooled = pooled.cpu()
            embeddings_orig_list.append(pooled)

    logits_orig = torch.cat(logits_orig_list, dim=0) if has_head else None
    embeddings_orig = torch.cat(embeddings_orig_list, dim=0)  # [n, embedding_dim]
    print(f"      Logits: {logits_orig.shape if has_head else 'skipped (no head)'}, "
          f"Embeddings: {embeddings_orig.shape}")

    # Linear-probe calibration set: drawn from the TRAIN split (never test/validation, which
    # is where the real ground-truth accuracy comes from -- see module docstring / PR
    # description for why mixing the two would leak labels between the probe's score and the
    # true accuracy it's being compared against). A separate hidden-state collection pass is
    # needed here since these are different images than the test/validation calibration set
    # collected above -- this roughly doubles this script's forward-pass cost.
    print("[probe] Loading train-split probe calibration set...")
    # shuffle=True: unlike every other calibration set in this script, the probe reads
    # labels and needs class diversity -- some datasets' train splits are stored grouped by
    # class (e.g. chestxray_pneumonia is normal-then-pneumonia), so a contiguous slice can
    # land entirely inside one class. See _load_dataset's docstring.
    probe_loader, probe_n = _load_dataset(
        ds_key, probe_num_samples, hf_id, split="train", shuffle=True
    )
    print(f"      Using {probe_n} probe images (train split)")
    probe_hidden_states, probe_labels = _collect_hidden_states_with_labels(
        model, probe_loader, num_layers, layers, probe_n
    )
    probe_labels_np = probe_labels.numpy()
    # Same final-layernorm fix as embeddings_orig above: probe_hidden_states[num_layers] is
    # the RAW last-block output (pre-final-layernorm), but probe_embeddings_approx (below,
    # via _apply_skip_forward) IS post-layernorm -- frozen_probe would otherwise be fit on
    # one normalization state and scored on another.
    probe_cls = probe_hidden_states[num_layers][:, 0, :].float()
    if hasattr(encoder, "layernorm"):
        with torch.no_grad():
            probe_cls = encoder.layernorm(probe_cls.to(device)).cpu()
    probe_embeddings_orig = probe_cls.numpy()

    probe_fit_idx, probe_score_idx = _split_probe_indices(probe_labels)
    frozen_probe, probe_baseline_accuracy = _fit_probe(
        probe_embeddings_orig, probe_labels_np, probe_fit_idx, probe_score_idx
    )
    print(f"      probe_baseline_accuracy (frozen probe on unmodified embeddings) = "
          f"{probe_baseline_accuracy:.4f}")

    # Spans to evaluate: either every contiguous single/multi-span candidate found for this
    # (model, dataset) in a calculate_accuracies.py-style summary CSV, or -- if no spans_csv
    # is given -- the original default of single-layer adjacent skips (0,1), (1,2), ...
    if spans_csv:
        print(f"[5/5] Loading spans from {spans_csv}...")
        spans = _load_spans_from_csv(spans_csv, hf_id, ds_key)
        if not spans:
            raise ValueError(
                f"No spans found in {spans_csv} for model={hf_id}, dataset={ds_key}"
            )
        print(f"      {len(spans)} spans (lengths {min(e-s for s,e in spans)}-"
              f"{max(e-s for s,e in spans)})")
    else:
        spans = [(i, i + 1) for i in range(num_layers - 1)]
        print(f"[5/5] Evaluating {len(spans)} single-layer skips...")

    results = []

    for skip_from, skip_to in spans:
        print(f"\n  Skip [{skip_from}, {skip_to})...")

        # Span (skip_from, skip_to) follows the repo-wide SkipModel convention (see
        # src/configs/make_skip_grid.py's docstring and src/toast/modules/module.py's
        # SkipModel.filter_layers): it KEEPS block skip_from and DROPS blocks
        # skip_from+1 .. skip_to inclusive. hidden_states is boundary-indexed (index 0 =
        # embedding output, index i = output after i blocks -- see docstring at the top of
        # this function), so "output of block skip_from" is hidden_states[skip_from + 1]
        # and "output of block skip_to" (what the dropped span must reproduce) is
        # hidden_states[skip_to + 1]. Using hidden_states[skip_from]/[skip_to] directly
        # (off by one) would silently score the wrong block against this span's ground-truth
        # accuracy row -- see the "Layer Indexing Convention" section in this module's
        # docstring for the worked example.
        h_in = hidden_states[skip_from + 1].float()  # [n, seq_len, d]
        h_out = hidden_states[skip_to + 1].float()  # [n, seq_len, d]
        W = _fit_translator(h_in, h_out)  # [d, d]

        # Local boundary CKA: how similar is this block's input to its own output, i.e.
        # CKA(hidden_states[skip_from + 1], hidden_states[skip_to + 1]) -- the same
        # per-token, per-boundary metric layer_priority.analyse() saves to *_cka.npy,
        # computed here directly on the same 500-image calibration set as every other
        # metric in this script (rather than reading a possibly stale/differently-sampled
        # .npy file).
        # Distinct from cka_embed below, which compares the FULL model's final output to
        # the SKIP model's final output after propagating the approximation through every
        # remaining block -- local_cka only looks at the one block being removed.
        local_cka = _linear_cka(
            h_in.reshape(-1, h_in.shape[-1]).to(device),
            h_out.reshape(-1, h_out.shape[-1]).to(device),
        )
        angular_dist = _angular_distance(h_in, h_out)

        # TOAST's own block-selection criterion (Cannistraci et al.) -- CLS-token-only
        # normalized lstsq residual, distinct from local_cka/angular_dist above (all-token)
        # and fit with its own separate lstsq solve, not the translator W above.
        toast_linear_error = _toast_linear_approx_error(h_in, h_out)

        # Forward skip model using precomputed hidden states (test/validation calibration
        # set -- this ensures we use exact same embeddings as original model)
        logits_approx, embeddings_approx = _apply_skip_forward(
            hidden_states, actual_n, skip_from, skip_to, W, encoder, num_layers,
            model=model, has_head=has_head,
        )

        # Compute metrics
        if has_head:
            kl, flip_rate, entropy_delta = _kl_and_flip_and_entropy(logits_orig, logits_approx)
        else:
            kl, flip_rate, entropy_delta = None, None, None
        cosine, mse, cka, knn_overlap_5, knn_overlap_10, pr_change = _embedding_metrics(
            embeddings_orig, embeddings_approx
        )

        # Linear-probe metrics: apply the SAME translator W to the train-split probe's
        # precomputed hidden states (never the test/validation calibration set), so the
        # probe's labeled fit/score data stays independent of the real accuracy ground
        # truth. See _fit_probe's docstring for the LogME/LEEP framing.
        _, probe_embeddings_approx = _apply_skip_forward(
            probe_hidden_states, probe_n, skip_from, skip_to, W, encoder, num_layers,
        )
        probe_embeddings_approx = probe_embeddings_approx.numpy()
        probe_frozen_accuracy = frozen_probe.score(
            probe_embeddings_approx[probe_score_idx], probe_labels_np[probe_score_idx]
        )
        _, probe_refit_accuracy = _fit_probe(
            probe_embeddings_approx, probe_labels_np, probe_fit_idx, probe_score_idx
        )

        results.append({
            "skip_from": skip_from,
            "skip_to": skip_to,
            "kl_div": kl,
            "top1_flip_pct": flip_rate * 100 if has_head else None,
            "entropy_delta": entropy_delta,
            "knn_overlap@5": knn_overlap_5,
            "knn_overlap@10": knn_overlap_10,
            "cosine_sim": cosine,
            "mse": mse,
            "cka_embed": cka,
            "participation_ratio_change": pr_change,
            "local_cka": local_cka,
            "angular_dist": angular_dist,
            "toast_linear_error": toast_linear_error,
            "probe_frozen_accuracy": probe_frozen_accuracy,
            "probe_refit_accuracy": probe_refit_accuracy,
            "probe_baseline_accuracy": probe_baseline_accuracy,
        })

        head_metrics_str = (
            f"KL={kl:.6f}, flip%={flip_rate*100:.2f}, entropy_d={entropy_delta:.4f}, "
            if has_head else "KL/flip/entropy=skipped (no head), "
        )
        print(f"      {head_metrics_str}"
              f"KNN5={knn_overlap_5:.4f}, KNN10={knn_overlap_10:.4f}, cosine={cosine:.4f}, "
              f"MSE={mse:.6f}, CKA={cka:.4f}, PR_change={pr_change:.4f}, "
              f"local_CKA={local_cka:.4f}, angular_dist={angular_dist:.4f}, "
              f"toast_linear_error={toast_linear_error:.4f}, "
              f"probe_frozen={probe_frozen_accuracy:.4f}, probe_refit={probe_refit_accuracy:.4f} "
              f"(baseline={probe_baseline_accuracy:.4f})")

        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Write output CSV
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["skip_from", "skip_to", "kl_div", "top1_flip_pct", "entropy_delta",
                        "knn_overlap@5", "knn_overlap@10", "cosine_sim", "mse", "cka_embed",
                        "participation_ratio_change", "local_cka", "angular_dist",
                        "toast_linear_error",
                        "probe_frozen_accuracy", "probe_refit_accuracy", "probe_baseline_accuracy"],
        )
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults written to: {output_path}")
    print(f"  {len(results)} rows evaluated ({num_layers}-layer model)")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Output-level skip-safety predictors")
    p.add_argument("--model", required=True, help="model alias (vitlarge) or HF id")
    p.add_argument("--dataset", required=True, help="dataset key (imagenet-1k, ...)")
    p.add_argument("--num-samples", type=int, default=500, help="calibration images (default: 500)")
    p.add_argument("--probe-num-samples", type=int, default=500,
                   help="TRAIN-split images for fitting/scoring the linear-probe metrics "
                        "(default: 500). Independent of --num-samples, which stays "
                        "test/validation-only and feeds every other metric.")
    p.add_argument("--output", default="metric_experiments/output_level_eval.csv", help="output CSV path")
    p.add_argument("--spans-csv", default=None,
                   help="calculate_accuracies.py-style summary CSV to pull the (skip_from, "
                        "skip_to) candidate list from, instead of the default single-layer "
                        "adjacent skips. Disjoint/multi-span rows (more than one tuple in "
                        "approx_layer) are skipped -- this script's forward pass only "
                        "handles one contiguous span per candidate.")
    args = p.parse_args()

    run(
        model_arg=args.model,
        dataset_name=args.dataset,
        num_samples=args.num_samples,
        output_csv=args.output,
        spans_csv=args.spans_csv,
        probe_num_samples=args.probe_num_samples,
    )
