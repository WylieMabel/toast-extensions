import hashlib
import json
import os
import random
from functools import partial, reduce
from pathlib import Path
import numpy as np
import torch
from torch import nn
from typing import Optional, Sequence, List, Dict
from collections import defaultdict
from tqdm import tqdm


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed: Optional[int] = None, workers: bool = False) -> int:
    """Seed python, numpy and torch. Drop-in replacement for pytorch_lightning's version.

    pytorch_lightning was a dependency of this project for this one function -- no
    LightningModule, no Trainer, just seeding. That import segfaults against torch 2.11 +
    numpy 2.x (a compiled extension loading with a mismatched ABI, which crashes in dlopen
    rather than raising), taking down every entry point in the pipeline with it. Since the
    only thing needed was five lines of seeding, the dependency is gone.

    Behaviour matches pytorch_lightning.seed_everything(seed, workers=False): the same three
    RNGs, seeded in the same way, so runs stay reproducible against earlier results. The
    ``workers`` argument is accepted for signature compatibility and ignored, as it was
    already never passed anywhere in this codebase.
    """
    seed = 0 if seed is None else int(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    # numpy only accepts seeds in [0, 2**32); mask rather than raise on a large seed.
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    return seed


def cfg_embedding_dir(cfg: dict, samples: int, base) -> Path:
    """Return a unique, deterministic directory for a single experiment config's embeddings.

    NOTE: this function is duplicated verbatim in src/toast/scripts/get_embed_dir.py (which
    inlines it to avoid importing torch). Any change here must be mirrored there, or the
    row-by-row runner will delete the wrong embedding directory.
    """
    head_dict = cfg.get("head_dict") or {}
    payload = {
        "dataset":        cfg.get("dataset"),
        "encoder":        cfg.get("encoder"),
        "skip":           str(cfg.get("skip") or []),
        "mlp_skip":       str(cfg.get("mlp_skip") or []),
        "attn_skip":      str(cfg.get("attn_skip") or []),
        "head_dict":      json.dumps(
            {str(k): sorted(v) for k, v in head_dict.items()}, sort_keys=True
        ),
        "skip_translator": cfg.get("skip_translator"),
        "mlp_mode":       cfg.get("mlp_mode", "identity"),
        "attn_mode":      cfg.get("attn_mode", "identity"),
        "samples":        samples,
    }

    # fit_dataset is only added to the hash when it actually differs from dataset. Adding it
    # unconditionally would change every existing hash and invalidate every embedding already
    # on disk; this way old caches keep resolving while transfer runs still get their own
    # directory instead of colliding with the same-dataset run.
    fit_dataset = cfg.get("fit_dataset")
    if fit_dataset and fit_dataset != cfg.get("dataset"):
        payload["fit_dataset"] = fit_dataset

    h = hashlib.md5(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]
    enc_slug = (cfg.get("encoder") or "unknown").split("/")[-1]
    return Path(base) / (cfg.get("dataset") or "unknown") / enc_slug / h


def resolve_path(obj, path: Optional[str]):
    """Gets an attribute/module using a dot-separated path string."""
    if not path:
        return obj
    try:
        return reduce(getattr, path.split("."), obj)
    except AttributeError:
        return None


@torch.no_grad()
def image_encode(
    samples: Sequence[Dict],
    processor,
    image_name,
    label_name,
):

    images: List[torch.Tensor] = [sample[image_name].convert("RGB") for sample in samples]
    images: List[torch.Tensor] = [processor(images=image, return_tensors="pt")["pixel_values"] for image in images]

    images = torch.cat(images, dim=0)

    return {"images": images, "labels": torch.tensor([sample[label_name] for sample in samples])}


@torch.no_grad()
def open_clip_image_encode(batch, processor, image_name="image", label_name="label"):
    images = [item[image_name] for item in batch]

    pixel_values = torch.stack([processor(img) for img in images])
    labels = torch.tensor([item[label_name] for item in batch])

    return {"pixel_values": pixel_values, "labels": labels, "attention_mask": None}


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def convert_parameters(num_parameters):
    if num_parameters >= 1_000_000_000:
        return f"{num_parameters / 1_000_000_000:.2f}B"
    elif num_parameters >= 1_000_000:
        return f"{num_parameters / 1_000_000:.2f}M"
    elif num_parameters >= 1_000:
        return f"{num_parameters / 1_000:.2f}K"
    else:
        return str(num_parameters)


@torch.no_grad()
def extract_representations(
    encoder: nn.Module,
    max_samples: int,
    loader: torch.utils.data.DataLoader,
    model_config: dict,
    model_is_open_clip: bool = False,
    use_hooks: bool = False,
    seed: int = 0,
):
    seed_everything(seed)
    encoder.eval().to(device)

    stored_samples = 0
    layer_outputs_batches = defaultdict(list)
    hooks = []

    try:
        if model_is_open_clip or use_hooks:
            layers_parent_module = resolve_path(encoder, model_config["layers_parent_path"])
            if layers_parent_module is None:
                raise ValueError(f"Could not resolve layers_parent_path: {model_config['layers_parent_path']}")

            layers_list = getattr(layers_parent_module, model_config["layers_attribute_name"])
            if not isinstance(layers_list, (nn.ModuleList, nn.Sequential, list)):
                raise TypeError(
                    f"Expected ModuleList, Sequential or list at {model_config['layers_parent_path']}.{model_config['layers_attribute_name']}, found {type(layers_list)}"
                )

            num_layers = len(layers_list)

            def get_output_hook(module, input, output, layer_idx):
                hidden_state = output[0] if isinstance(output, tuple) else output
                current_batch_size = hidden_state.shape[0]
                samples_needed_from_batch = max(0, min(current_batch_size, max_samples - stored_samples))

                if samples_needed_from_batch > 0:
                    layer_outputs_batches[layer_idx].append(
                        hidden_state[:samples_needed_from_batch].cpu().detach().clone()
                    )

            for i, layer_module in enumerate(layers_list):
                hook = layer_module.register_forward_hook(partial(get_output_hook, layer_idx=i))
                hooks.append(hook)

            for batch in tqdm(loader, desc="Extracting via Hooks"):
                if stored_samples >= max_samples:
                    break

                image_input = batch.get("pixel_values", batch.get("images"))
                if image_input is None:
                    continue
                image_input = image_input.to(device)
                batch_size = image_input.shape[0]

                if hasattr(encoder, "encode_image"):
                    _ = encoder.encode_image(image_input)
                elif hasattr(encoder, "visual"):
                    _ = encoder.visual(image_input)
                else:
                    _ = encoder(image_input)

                stored_samples += min(batch_size, max_samples - stored_samples)

        else:
            if not hasattr(encoder, "config") or not hasattr(encoder.config, "num_hidden_layers"):
                raise ValueError("Cannot determine number of layers. Model lacks standard config.")

            num_layers = encoder.config.num_hidden_layers

            for batch in tqdm(loader, desc="Extracting via HiddenStates"):
                if stored_samples >= max_samples:
                    break

                image_input = batch.get("pixel_values", batch.get("images"))
                if image_input is None:
                    continue
                image_input = image_input.to(device)
                attn_mask = batch.get("attention_mask")
                if attn_mask is not None:
                    attn_mask = attn_mask.to(device)

                outputs = encoder(
                    pixel_values=image_input,
                    output_hidden_states=True,
                    return_dict=True,
                )

                if not hasattr(outputs, "hidden_states") or outputs.hidden_states is None:
                    raise ValueError("Model did not return 'hidden_states'. Ensure 'output_hidden_states=True'.")

                actual_hidden_states = outputs.hidden_states[1:]

                num_to_add = min(image_input.size(0), max_samples - stored_samples)

                if num_to_add > 0:
                    for layer_idx, layer_output in enumerate(actual_hidden_states):
                        layer_outputs_batches[layer_idx].append(layer_output[:num_to_add].cpu().detach().clone())
                    stored_samples += num_to_add

    finally:
        if hooks:
            for hook in hooks:
                hook.remove()

    final_layer_embeddings = {}
    captured_layers = sorted(layer_outputs_batches.keys())
    for layer_idx in captured_layers:
        if layer_outputs_batches[layer_idx]:
            concatenated = torch.cat(layer_outputs_batches[layer_idx], dim=0)
            final_layer_embeddings[layer_idx] = concatenated[:max_samples]


    return final_layer_embeddings
