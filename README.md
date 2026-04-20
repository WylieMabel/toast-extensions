<div align="center">

# 🍞 TOAST: Transformer Optimization using Adaptive and Simple Transformations

**Irene Cannistraci**\*, Simone Antonelli, Emanuele Palumbo, Thomas M. Sutter, Emanuele Rodola, Bastian Rieck, Julia E. Vogt

\* *ETH Zurich, Department of Computer Science*

[![OpenReview](https://img.shields.io/badge/OpenReview-TMLR%202026-blue.svg)](https://openreview.net/forum?id=fSwMCsBtTG)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

</div>

---

> Foundation models achieve state-of-the-art performance across different tasks, but their size and computational demands raise concerns about accessibility and sustainability. Existing efficiency methods often require additional retraining or fine-tuning, limiting their practicality. We introduce **TOAST**, a framework that exploits intra-network redundancies to approximate entire transformer blocks with lightweight closed-form mappings, such as linear transformation or even the identity, **without any additional training**. Across SOTA pretrained vision models (ViT, DINOv2, DeiT) and datasets ranging from MNIST to ImageNet-1k, TOAST reduces parameters and computation while preserving, and in some cases improving, downstream performance.

## Installation

```bash
pip install -e .

# (optional) for wandb logging
pip install -e ".[wandb]"
```

## Usage

TOAST operates in two steps: **(1)** extract per-layer representations and encode them through each skip configuration, then **(2)** evaluate with a classifier or fine-tune end-to-end.

### Step 1 &mdash; Encode vision representations

```bash
bash src/toast/scripts/encode_vision.sh
```

Runs [`src/toast/utils/encode_vision.py`](src/toast/utils/encode_vision.py). Key parameters:

| Parameter | Description | Example values |
|-----------|-------------|----------------|
| `dataset_name` | Target dataset | `cifar10`, `cifar100-fine`, `imagenet-1k`, `mnist` |
| `encoder_name` | Pretrained model (HF name) | `google/vit-base-patch16-224`, `facebook/dinov2-small` |
| `translator_name` | Mapping used to approximate skipped blocks | `linear`, `identity`, `mlp`, `deep_mlp` |
| `skips` | List of skip configs to encode | `[[], [(0, 1)], [(2, 5)]]` |
| `samples_to_extract` | Number of samples to fit the translator | `500`, `1000` |
| `seed` | Random seed | `0` |

### Step 2a &mdash; Train a classifier on skipped embeddings

```bash
bash src/toast/scripts/train_skipped.sh
```

Runs [`src/toast/utils/train_skipped.py`](src/toast/utils/train_skipped.py). Key parameters:

| Parameter | Description | Example values |
|-----------|-------------|----------------|
| `dataset_name` | Must match Step 1 | `cifar100-coarse`, `imagenet-1k` |
| `model_name` | Must match `encoder_name` from Step 1 | `google/vit-base-patch16-224` |
| `layers_to_approximate` | Which blocks to skip (list of `(start, end)` ranges) | `[]` (baseline), `[(0, 1)]`, `[(2, 5)]` |
| `classifier_type` | Classification head type | `linear`, `MLP` |
| `translator_name` | Must match Step 1 | `linear` |
| `samples_to_extract` | Must match Step 1 | `500` |
| `seed` | Random seed | `0`, `1`, `2` |

### Step 2b &mdash; End-to-end fine-tuning

```bash
bash src/toast/scripts/finetune_e2e.sh
```

Runs [`src/toast/utils/finetune_e2e.py`](src/toast/utils/finetune_e2e.py). Key parameters:

| Parameter | Description | Example values |
|-----------|-------------|----------------|
| `dataset_name` | Must match Step 1 | `cifar100-fine`, `imagenet-1k` |
| `model_name` | Must match Step 1 | `facebook/dinov2-base` |
| `layers_to_approximate` | Which blocks to skip | `[]`, `[(0, 1)]`, `[(4, 8)]` |
| `translator_name` | Must match Step 1 | `linear` |
| `samples_to_extract` | Must match Step 1 | `500` |
| `lr` | Classifier learning rate | `2e-4` |
| `encoder_lr` | Encoder learning rate (optional, for unfreezing) | `1e-6` |
| `num_epochs` | Training epochs | `20` |
| `batch_size` | Batch size | `128` |
| `seed` | Random seed | `0` |

### Inference with pretrained heads (no training)

For models that already have a pretrained classification head matching the target dataset (e.g., ViT/DeiT trained on ImageNet-1k), you can skip fine-tuning entirely and evaluate TOAST in a purely training-free setting:

```bash
bash src/toast/scripts/imagenet_inference.sh
```

This also runs [`src/toast/utils/finetune_e2e.py`](src/toast/utils/finetune_e2e.py) with `--use_pretrained_head=True`. Only requires Step 1 embeddings to be computed first.

## Supported models

| Family | Short name | HuggingFace / timm identifier |
|--------|-----------|-------------------------------|
| ViT | `vit-tiny-patch16-224` | `WinKawaks/vit-tiny-patch16-224` |
| ViT | `vit-small-patch16-224` | `WinKawaks/vit-small-patch16-224` |
| ViT | `vit-base-patch16-224` | `google/vit-base-patch16-224` |
| ViT | `vit-large-patch16-224` | `google/vit-large-patch16-224` |
| DeiT | `deit-small-patch16-224` | `facebook/deit-small-patch16-224` |
| DeiT | `deit-base-patch16-224` | `facebook/deit-base-patch16-224` |
| DINOv2 | `dinov2-small` | `facebook/dinov2-small` |
| DINOv2 | `dinov2-base` | `facebook/dinov2-base` |
| CLIP | `clip-base` | `openai/clip-vit-base-patch32` |
| timm | `vit-tiny-patch16-224-augreg` | `timm/vit_tiny_patch16_224.augreg_in21k_ft_in1k` |
| timm | `vit-small-patch16-224-augreg` | `timm/vit_small_patch16_224.augreg_in21k_ft_in1k` |
| timm | `vit-large-patch16-224-augreg` | `timm/vit_large_patch16_224.augreg_in21k_ft_in1k` |
| timm | `deit-base-patch16-224-fb` | `timm/deit_base_patch16_224.fb_in1k` |

## Supported datasets

| Name | HuggingFace identifier | Classes |
|------|----------------------|---------|
| `mnist` | `mnist` | 10 |
| `fashion-mnist` | `zalando-datasets/fashion_mnist` | 10 |
| `cifar10` | `cifar10` | 10 |
| `cifar100-fine` | `cifar100` | 100 |
| `cifar100-coarse` | `cifar100` | 20 |
| `imagenet-1k` | `ILSVRC/imagenet-1k` | 1000 |

See [`src/toast/utils/dictionaries.py`](src/toast/utils/dictionaries.py) for the full configuration.

## Citation (WIP)

```bibtex
@article{
cannistraci2026toast,
title={{TOAST}: Transformer Optimization using Adaptive and Simple Transformations},
author={Irene Cannistraci and Simone Antonelli and Emanuele Palumbo and Thomas M. Sutter and Emanuele Rodol{\`a} and Bastian Rieck and Julia E. Vogt},
journal={Transactions on Machine Learning Research},
issn={},
year={2026},
url={https://openreview.net/forum?id=fSwMCsBtTG},
note={}
}
```

## License

MIT &mdash; see [LICENSE](LICENSE) for details.
