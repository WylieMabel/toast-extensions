from pathlib import Path
import pickle
from typing import Callable, Dict, Sequence, Optional, Any
from torch import nn
import torch
import torch.nn.functional as F
from functools import partial
from toast.utils.dictionaries import NAME2TRANSLATORS
from toast.utils.translator_keys import span_translator_key
from toast.utils.utils import resolve_path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#: Translators that cannot round-trip through save_translator/load_translator.
#:
#: This is a limitation of the *persistence path only* -- these translators work fine in the
#: normal fit-and-use-in-one-process flow that every existing config uses. The problem is that
#: save_translator writes translator.aligner.state_dict() and nothing else, while these
#: translators keep fitted state elsewhere: "sgd_mlp_aligner" wraps StandardScaling() as its
#: x_transform/y_transform, and those learn a mean and std during fit(). Reloading would
#: restore the aligner but leave the scalers unfitted, feeding raw data to an aligner that was
#: fit on standardised data -- wrong numbers with no error. Refuse instead of half-restoring.
#:
#: The translators used for transfer experiments (linear, lowrank_*, rrr_*, lora_*) have no
#: input/output transforms, so they are unaffected.
_TRANSLATORS_WITH_UNSAVED_STATE = {"sgd_mlp_aligner"}


class HFwrapper(nn.Module):
    def __init__(self, encoder, classifier):
        super().__init__()
        self.encoder = encoder
        self.classifier = classifier

    def freeze_encoder(self):
        self.encoder.eval()
        for param in self.encoder.parameters():
            param.requires_grad = False

    def encode(self, embedding_tensor: torch.Tensor) -> torch.Tensor:
        x = self.encoder(embedding_tensor)

        if hasattr(x, 'last_hidden_state'):
            x = x.last_hidden_state[:, 0]
        elif hasattr(x, 'pooler_output') and x.pooler_output is not None:
            x = x.pooler_output
        elif not isinstance(x, torch.Tensor):
            if hasattr(x, 'last_hidden_state'):
                x = x.last_hidden_state.mean(dim=1)
            else:
                raise ValueError(f"Unexpected encoder output type: {type(x)}")

        return x

    def decode(self, encoded_embeddings: torch.Tensor) -> torch.Tensor:
        logits = self.classifier(encoded_embeddings)
        return logits

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        embedding_tensor = batch["images"]

        encoded_x = self.encode(embedding_tensor)
        logits = self.decode(encoded_x)

        return logits


class NoEncoder(nn.Module):

    def __init__(self, embeddings=None):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x

class MLPLinearisedEncoder(nn.Module):
    """
    Replaces target MLP layers with a linear approximation in one of two modes.

    mode="collapse" (default): purely algebraic. Composes fc1 and fc2 weight
        matrices into a single linear map at init time — no data needed, no
        non-linearity. output = F.linear(x, W2@W1, W2@b1+b2).

    mode="fitted": data-driven. Runs a one-time calibration pass to collect
        (input, output) pairs at each target MLP boundary, then fits a
        least-squares linear map (same translator as SkipModel "linear").
        Call .fit(loader) after construction; every forward pass after that
        is just a single matrix multiply.
    """

    def __init__(self, encoder, mlp_layers_to_linearize=None, mode: str = "collapse"):
        super().__init__()

        if mode not in ("collapse", "linear", "identity"):
            raise ValueError(f"mode must be 'collapse', 'linear', or 'identity', got '{mode}'")

        self.encoder = encoder
        self.mlp_layers_to_linearize = set(mlp_layers_to_linearize or [])
        self.mode = mode

        if mode == "collapse":
            self._patch_mlp_collapse()
        elif mode == "identity":
            self._patch_mlp_identity()

    def _get_layers(self):
        if hasattr(self.encoder, "encoder") and hasattr(self.encoder.encoder, "layer"):
            return self.encoder.encoder.layer
        if hasattr(self.encoder, "vision_model"):
            vm = self.encoder.vision_model
            if hasattr(vm.encoder, "layers"):
                return vm.encoder.layers
        raise ValueError("Could not find transformer layers in the encoder")

    def _patch_mlp_collapse(self):
        layers = self._get_layers()

        for idx, layer in enumerate(layers):
            if idx not in self.mlp_layers_to_linearize:
                continue

            # CLIP-style: layer.mlp.fc1 / layer.mlp.fc2
            if hasattr(layer, "mlp"):
                mlp = layer.mlp
                if not (hasattr(mlp, "fc1") and hasattr(mlp, "fc2")):
                    continue
                W1 = mlp.fc1.weight.detach()
                b1 = mlp.fc1.bias.detach()
                W2 = mlp.fc2.weight.detach()
                b2 = mlp.fc2.bias.detach()
                W_combined = (W2 @ W1).clone()
                b_combined = (W2 @ b1 + b2).clone()
                mlp.register_buffer("_W_collapsed", W_combined)
                mlp.register_buffer("_b_collapsed", b_combined)
                def _make_clip_forward(mod):
                    def collapsed_forward(x):
                        s = x.shape
                        return F.linear(x.reshape(-1, s[-1]), mod._W_collapsed, mod._b_collapsed).reshape(s)
                    return collapsed_forward
                mlp.forward = _make_clip_forward(mlp)

            # HF ViT/DeiT-style: layer.intermediate (dense + act) + layer.output (dense + dropout + residual)
            elif hasattr(layer, "intermediate") and hasattr(layer, "output"):
                inter = layer.intermediate
                out = layer.output
                if not (hasattr(inter, "dense") and hasattr(out, "dense")):
                    continue
                W1 = inter.dense.weight.detach()
                b1 = inter.dense.bias.detach()
                W2 = out.dense.weight.detach()
                b2 = out.dense.bias.detach()
                W_combined = (W2 @ W1).clone()
                b_combined = (W2 @ b1 + b2).clone()
                # Register as buffers so .to(device) moves them with the model
                inter.register_buffer("_W_collapsed", W_combined)
                inter.register_buffer("_b_collapsed", b_combined)
                def _make_vit_intermediate_forward(mod):
                    def collapsed_intermediate(x):
                        s = x.shape
                        return F.linear(x.reshape(-1, s[-1]), mod._W_collapsed, mod._b_collapsed).reshape(s)
                    return collapsed_intermediate
                def _make_vit_output_forward(mod):
                    def passthrough_output(hidden_states, input_tensor):
                        # dense is replaced by collapsed_intermediate above; keep dropout + residual
                        hidden_states = mod.dropout(hidden_states)
                        return hidden_states + input_tensor
                    return passthrough_output
                inter.forward = _make_vit_intermediate_forward(inter)
                out.forward = _make_vit_output_forward(out)

    def _patch_mlp_identity(self):
        layers = self._get_layers()
        for idx, layer in enumerate(layers):
            if idx not in self.mlp_layers_to_linearize:
                continue
            if hasattr(layer, "mlp"):
                def identity_mlp(x, *_):
                    return torch.zeros_like(x)
                layer.mlp.forward = identity_mlp
            elif hasattr(layer, "intermediate") and hasattr(layer, "output"):
                def _make_identity_output():
                    def identity_output(_, input_tensor):
                        return input_tensor
                    return identity_output
                layer.output.forward = _make_identity_output()

    @torch.no_grad()
    def fit(self, loader, max_samples: int = 500):
        """
        Calibration pass for mode="fitted".

        Registers a forward hook on each target MLP to collect (input, output)
        token-level pairs across batches. After enough samples are seen, removes
        the hooks, fits one least-squares linear translator per MLP, and patches
        each mlp.forward with the resulting map. Called once; not called again
        during inference.
        """
        if self.mode != "linear":
            raise RuntimeError("fit() is only valid for mode='linear'")

        layers = self._get_layers()

        mlp_inputs: dict[int, list] = {i: [] for i in self.mlp_layers_to_linearize}
        mlp_outputs: dict[int, list] = {i: [] for i in self.mlp_layers_to_linearize}
        hooks = []

        for idx, layer in enumerate(layers):
            if idx not in self.mlp_layers_to_linearize:
                continue

            if hasattr(layer, "mlp"):
                # CLIP-style: single mlp module; hook captures input and output together
                def make_clip_hook(i):
                    def hook(_, inp, out):
                        x = inp[0].detach().cpu()
                        y = out.detach().cpu()
                        mlp_inputs[i].append(x.reshape(-1, x.shape[-1]))
                        mlp_outputs[i].append(y.reshape(-1, y.shape[-1]))
                    return hook
                hooks.append(layer.mlp.register_forward_hook(make_clip_hook(idx)))

            elif hasattr(layer, "intermediate") and hasattr(layer, "output"):
                # HF ViT/DeiT-style: hook intermediate for inputs, output.dense for targets
                # (output.dense gives the MLP result before dropout and residual)
                def make_vit_input_hook(i):
                    def hook(_, inp, __):
                        x = inp[0].detach().cpu()
                        mlp_inputs[i].append(x.reshape(-1, x.shape[-1]))
                    return hook

                def make_vit_output_hook(i):
                    def hook(_, __, out):
                        y = out.detach().cpu()
                        mlp_outputs[i].append(y.reshape(-1, y.shape[-1]))
                    return hook

                hooks.append(layer.intermediate.register_forward_hook(make_vit_input_hook(idx)))
                hooks.append(layer.output.dense.register_forward_hook(make_vit_output_hook(idx)))

        self.encoder.eval()
        n_seen = 0
        for batch in loader:
            images = batch.get("pixel_values", batch.get("images"))
            if images is None:
                raise KeyError("Batch missing 'pixel_values' or 'images'")
            self.encoder(images.to(next(self.encoder.parameters()).device))
            n_seen += images.shape[0]
            if n_seen >= max_samples:
                break

        for h in hooks:
            h.remove()

        translator_factory = NAME2TRANSLATORS["linear"]
        for idx, layer in enumerate(layers):
            if idx not in self.mlp_layers_to_linearize:
                continue
            if not mlp_inputs[idx]:
                continue

            X = torch.cat(mlp_inputs[idx]).float()  # [N*seq_len, d_in]
            Y = torch.cat(mlp_outputs[idx]).float()  # [N*seq_len, d_out]

            translator = translator_factory()
            translator.fit(x=X, y=Y)

            def make_forward(t):
                def fitted_forward(x):
                    s = x.shape
                    dev = x.device
                    out, _ = t.transform(x.reshape(-1, s[-1]).float().cpu())
                    return out.reshape(s).to(dev).to(x.dtype)
                return fitted_forward

            if hasattr(layer, "mlp"):
                layer.mlp.forward = make_forward(translator)
            elif hasattr(layer, "intermediate") and hasattr(layer, "output"):
                layer.intermediate.forward = make_forward(translator)
                def _make_vit_output_forward(mod):
                    def passthrough_output(hidden_states, input_tensor):
                        hidden_states = mod.dropout(hidden_states)
                        return hidden_states + input_tensor
                    return passthrough_output
                layer.output.forward = _make_vit_output_forward(layer.output)

        return self

    def forward(self, *args, **kwargs):
        return self.encoder(*args, **kwargs)


class AttentionLinearisedEncoder(nn.Module):
    """
    Replaces target self-attention layers with a linear approximation.

    mode="identity": bypasses attention entirely — the patched forward returns zeros,
        so the residual connection in ViTLayer leaves hidden_states unchanged.
        The MLP sublayer of each targeted layer still runs normally.

    mode="linear": data-driven. Runs a calibration pass to collect (input, output)
        pairs at each target attention layer, fits a least-squares linear map, and
        patches each layer.attention.forward. Call .fit(loader) after construction.
    """

    def __init__(self, encoder, attention_layers_to_linearize=None, mode: str = "linear"):
        super().__init__()
        if mode not in ("identity", "linear"):
            raise ValueError(f"mode must be 'identity' or 'linear', got '{mode}'")
        self.encoder = encoder
        flat = []
        for item in (attention_layers_to_linearize or []):
            if isinstance(item, (list, tuple)):
                flat.extend(item)
            else:
                flat.append(item)
        self.attention_layers_to_linearize = set(flat)
        self.mode = mode

        if mode == "identity":
            self._patch_attention_identity()

    def _get_layers(self):
        if hasattr(self.encoder, "encoder") and hasattr(self.encoder.encoder, "layer"):
            return self.encoder.encoder.layer
        if hasattr(self.encoder, "vision_model"):
            vm = self.encoder.vision_model
            if hasattr(vm.encoder, "layers"):
                return vm.encoder.layers
        raise ValueError("Could not find transformer layers in the encoder")

    def _patch_attention_identity(self):
        layers = self._get_layers()
        for idx, layer in enumerate(layers):
            if idx not in self.attention_layers_to_linearize:
                continue
            if not hasattr(layer, "attention"):
                continue
            def identity_attn(hidden_states, *_):
                return torch.zeros_like(hidden_states)
            layer.attention.forward = identity_attn

    @torch.no_grad()
    def fit(self, loader, max_samples: int = 500):
        if self.mode != "linear":
            raise RuntimeError("fit() is only valid for mode='linear'")

        layers = self._get_layers()
        attn_inputs: dict[int, list] = {i: [] for i in self.attention_layers_to_linearize}
        attn_outputs: dict[int, list] = {i: [] for i in self.attention_layers_to_linearize}
        hooks = []

        for idx, layer in enumerate(layers):
            if idx not in self.attention_layers_to_linearize or not hasattr(layer, "attention"):
                continue

            def make_input_hook(i):
                def hook(_, inp, __):
                    x = inp[0].detach().cpu()
                    attn_inputs[i].append(x.reshape(-1, x.shape[-1]))
                return hook

            def make_output_hook(i):
                def hook(_, __, out):
                    y = out.detach().cpu() if isinstance(out, torch.Tensor) else out[0].detach().cpu()
                    attn_outputs[i].append(y.reshape(-1, y.shape[-1]))
                return hook

            hooks.append(layer.attention.register_forward_pre_hook(make_input_hook(idx)))
            hooks.append(layer.attention.register_forward_hook(make_output_hook(idx)))

        self.encoder.eval()
        n_seen = 0
        for batch in loader:
            images = batch.get("pixel_values", batch.get("images"))
            if images is None:
                raise KeyError("Batch missing 'pixel_values' or 'images'")
            self.encoder(images.to(next(self.encoder.parameters()).device))
            n_seen += images.shape[0]
            if n_seen >= max_samples:
                break

        for h in hooks:
            h.remove()

        translator_factory = NAME2TRANSLATORS["linear"]
        for idx, layer in enumerate(layers):
            if idx not in self.attention_layers_to_linearize or not hasattr(layer, "attention"):
                continue
            if not attn_inputs[idx]:
                continue

            X = torch.cat(attn_inputs[idx]).float()
            Y = torch.cat(attn_outputs[idx]).float()

            translator = translator_factory()
            translator.fit(x=X, y=Y)

            def make_forward(t):
                def fitted_attn(hidden_states, *_):
                    s = hidden_states.shape
                    dev = hidden_states.device
                    out, _ = t.transform(hidden_states.reshape(-1, s[-1]).float().cpu())
                    return out.reshape(s).to(dev).to(hidden_states.dtype)
                return fitted_attn

            layer.attention.forward = make_forward(translator)

        return self

    def forward(self, *args, **kwargs):
        return self.encoder(*args, **kwargs)


class HeadPrunedEncoder(nn.Module):
    """
    Keeps only the specified attention heads in target layers; all others are
    permanently removed by resizing Q/K/V and the output projection in-place.

    heads_to_keep: {layer_idx: [head_indices_to_keep]}
    Layers not in the dict are left unchanged.
    Only supports HF ViT/DeiT-style attention (layer.attention.attention).
    """

    def __init__(self, encoder, heads_to_keep: dict):
        super().__init__()
        self.encoder = encoder
        self.heads_to_keep = {int(k): list(v) for k, v in heads_to_keep.items()}
        self._patch_heads()

    def _get_layers(self):
        if hasattr(self.encoder, "encoder") and hasattr(self.encoder.encoder, "layer"):
            return self.encoder.encoder.layer
        if hasattr(self.encoder, "vision_model"):
            vm = self.encoder.vision_model
            if hasattr(vm.encoder, "layers"):
                return vm.encoder.layers
        raise ValueError("Could not find transformer layers in the encoder")

    def _patch_heads(self):
        layers = self._get_layers()

        for layer_idx, keep in self.heads_to_keep.items():
            layer = layers[layer_idx]
            if not hasattr(layer, "attention") or not hasattr(layer.attention, "attention"):
                raise ValueError(f"Layer {layer_idx} does not have HF-style attention.attention block")

            attn_block = layer.attention.attention
            out_dense = layer.attention.output.dense

            num_heads = attn_block.num_attention_heads
            hidden_size = attn_block.query.weight.shape[1]
            head_size = hidden_size // num_heads

            remove = sorted(set(range(num_heads)) - set(keep), reverse=True)
            if not remove:
                continue

            with torch.no_grad():
                q_w = attn_block.query.weight.data.clone()
                q_b = attn_block.query.bias.data.clone()
                k_w = attn_block.key.weight.data.clone()
                k_b = attn_block.key.bias.data.clone()
                v_w = attn_block.value.weight.data.clone()
                v_b = attn_block.value.bias.data.clone()
                d_w = out_dense.weight.data.clone()
                d_b = out_dense.bias.data.clone()

                for h in remove:
                    s, e = h * head_size, (h + 1) * head_size
                    q_w = torch.cat([q_w[:s], q_w[e:]], dim=0)
                    q_b = torch.cat([q_b[:s], q_b[e:]], dim=0)
                    k_w = torch.cat([k_w[:s], k_w[e:]], dim=0)
                    k_b = torch.cat([k_b[:s], k_b[e:]], dim=0)
                    v_w = torch.cat([v_w[:s], v_w[e:]], dim=0)
                    v_b = torch.cat([v_b[:s], v_b[e:]], dim=0)
                    d_w = torch.cat([d_w[:, :s], d_w[:, e:]], dim=1)

                rem_heads = len(keep)
                rem_feat = rem_heads * head_size

                attn_block.query = nn.Linear(hidden_size, rem_feat, bias=True)
                attn_block.query.weight = nn.Parameter(q_w)
                attn_block.query.bias = nn.Parameter(q_b)
                attn_block.key = nn.Linear(hidden_size, rem_feat, bias=True)
                attn_block.key.weight = nn.Parameter(k_w)
                attn_block.key.bias = nn.Parameter(k_b)
                attn_block.value = nn.Linear(hidden_size, rem_feat, bias=True)
                attn_block.value.weight = nn.Parameter(v_w)
                attn_block.value.bias = nn.Parameter(v_b)

                new_dense = nn.Linear(rem_feat, hidden_size, bias=True)
                new_dense.weight = nn.Parameter(d_w)
                new_dense.bias = nn.Parameter(d_b)
                layer.attention.output.dense = new_dense

                attn_block.num_attention_heads = rem_heads
                attn_block.all_head_size = rem_feat
                if hasattr(attn_block, "attention_head_size"):
                    attn_block.attention_head_size = head_size

    def forward(self, *args, **kwargs):
        return self.encoder(*args, **kwargs)


class SkipModel(nn.Module):

    def __init__(
        self,
        encoder: nn.Module,
        skips: Sequence[tuple[int, int]],
        mode: int,
        precomputed_embeddings: dict[int, torch.Tensor],
        translator_factory_name: str,
        embeddings_path: str,
        layers_parent_path: str,
        layers_attribute_name: str,
        layers_accept_masks: bool,
        pre_norm_path: Optional[str] = None,
        post_norm_path: Optional[str] = None,
        pooler_path: Optional[str] = None,
        needs_conv1_processing: bool = False,
        class_embedding_path: Optional[str] = None,
        positional_embedding_path: Optional[str] = None,
        embedding_dropout_path: Optional[str] = None,
        precomputed_translator_path: Optional[Path] = None,
        to_save_translator_path: Optional[Path] = None,
        translator_key: Optional[str] = None,
        needs_position_ids: bool = False,
    ):
        super().__init__()

        self.encoder = encoder
        self.skips = skips
        self.mode = mode
        self.precomputed_embeddings = precomputed_embeddings
        self.translator_factory_name = translator_factory_name
        self.precomputed_translator_path = precomputed_translator_path
        self.to_save_translator_path = to_save_translator_path
        self.translator_key = translator_key

        self.check_skip_consistency()
        self.check_translator_consistency()

        self.needs_conv1_processing = needs_conv1_processing
        self.layers_accept_masks = layers_accept_masks
        self.needs_position_ids = needs_position_ids

        self.embeddings_module = resolve_path(self.encoder, embeddings_path)
        layers_parent_module = resolve_path(self.encoder, layers_parent_path)
        self.encoder_layers_list = getattr(layers_parent_module, layers_attribute_name)
        self.pre_norm_module = resolve_path(self.encoder, pre_norm_path) if pre_norm_path else None
        self.post_norm_module = resolve_path(self.encoder, post_norm_path) if post_norm_path else nn.Identity()
        self.pooler_module = resolve_path(self.encoder, pooler_path) if pooler_path else None
        self.class_embedding = resolve_path(self.encoder, class_embedding_path) if class_embedding_path else None
        self.positional_embedding = (
            resolve_path(self.encoder, positional_embedding_path) if positional_embedding_path else None
        )
        self.embedding_dropout = (
            resolve_path(self.encoder, embedding_dropout_path) if embedding_dropout_path else nn.Identity()
        )

        self.filtered_layers_list: Sequence[IndexedLayer] = self.filter_layers(
            self.encoder_layers_list, self.skips, self.layers_accept_masks, self.needs_position_ids
        )

        if self.precomputed_translator_path and self.mode != 1:
            raise ValueError(
                "precomputed_translator_path is only supported for mode=1. save_translator "
                "serialises a single aligner's state_dict, so the per-token translators of "
                f"mode={self.mode} have no on-disk representation."
            )

        # Translators are loaded per span inside compute_skipping, not once here -- one
        # config can carry several spans and each needs its own map.
        self.computed_skips: Sequence[IndexedLayer] = self.compute_skipping(
            self.precomputed_embeddings,
            self.skips,
            self.mode,
            self.to_save_translator_path,
            self.translator_key,
        )

        self.final_layers_list = sorted(
            (self.filtered_layers_list + self.computed_skips), key=lambda layer: layer.index
        )

    def encode(self, x: Any, attention_mask: Optional[torch.Tensor] = None):
        hidden_states = None

        if attention_mask is not None:
            if attention_mask.dtype in [torch.int64, torch.long]:
                attention_mask = attention_mask.float()
            if attention_mask.ndim == 2:
                attention_mask = (1.0 - attention_mask[:, None, None, :]) * torch.finfo(attention_mask.dtype).min

        if self.needs_conv1_processing:
            if self.embeddings_module is None or self.class_embedding is None or self.positional_embedding is None:
                raise ValueError(
                    "Missing required components (embeddings_module, class_embedding, positional_embedding) for needs_conv1_processing=True"
                )

            hidden_states = self.embeddings_module(x)
            hidden_states = hidden_states.reshape(hidden_states.shape[0], hidden_states.shape[1], -1)
            hidden_states = hidden_states.permute(0, 2, 1)

            class_embedding_expanded = (
                self.class_embedding.unsqueeze(0)
                .expand(hidden_states.shape[0], -1, -1)
                .to(hidden_states.device, dtype=hidden_states.dtype)
            )
            hidden_states = torch.cat([class_embedding_expanded, hidden_states], dim=1)

            pos_embedding_ready = self.positional_embedding.to(hidden_states.device, dtype=hidden_states.dtype)
            if pos_embedding_ready.shape[0] == hidden_states.shape[1]:
                hidden_states = hidden_states + pos_embedding_ready.unsqueeze(0)
            elif (
                pos_embedding_ready.shape[0] == 1 and pos_embedding_ready.shape[1] == hidden_states.shape[1]
            ):
                hidden_states = hidden_states + pos_embedding_ready
            else:
                raise ValueError(
                    f"Positional embedding shape {pos_embedding_ready.shape} incompatible with hidden_states shape {hidden_states.shape}"
                )

            hidden_states = self.embedding_dropout(hidden_states)

            if self.pre_norm_module:
                hidden_states = self.pre_norm_module(hidden_states)

        else:
            if self.embeddings_module is None:
                raise ValueError("embeddings_module is required for standard processing")

            hidden_states = self.embeddings_module(x)

            if self.pre_norm_module:
                hidden_states = self.pre_norm_module(hidden_states)

        current_attention_mask = attention_mask
        current_causal_attention_mask = None

        current_position_ids = None
        if self.needs_position_ids and hidden_states is not None:
            seq_length = hidden_states.shape[1]
            current_position_ids = torch.arange(seq_length, device=hidden_states.device).unsqueeze(0).expand(hidden_states.shape[0], -1)

        for indexed_layer in self.final_layers_list:
            layer_callable = indexed_layer.layer

            is_skip_transform = (
                isinstance(layer_callable, partial) and layer_callable.func == self.transform_similar_spaces
            )

            if is_skip_transform:
                hidden_states = indexed_layer(hidden_states)
            else:
                hidden_states = indexed_layer(
                    hidden_states,
                    attention_mask=current_attention_mask,
                    causal_attention_mask=current_causal_attention_mask,
                    position_ids=current_position_ids,
                )

                if isinstance(hidden_states, tuple):
                    hidden_states = hidden_states[0]

        return hidden_states

    def forward(self, x: Any, attention_mask: Optional[torch.Tensor] = None, return_sequence: bool = False):
        hidden_states = self.encode(x, attention_mask=attention_mask)
        sequence_output = self.post_norm_module(hidden_states)

        if return_sequence:
            return sequence_output

        pooled_output = None
        if self.pooler_module:
            pooled_output = self.pooler_module(sequence_output)
        else:
            if sequence_output is not None and sequence_output.ndim >= 3 and sequence_output.shape[1] > 0:
                pooled_output = sequence_output[:, 0, :]
            else:
                pooled_output = None

        if pooled_output is None:
            return sequence_output

        return pooled_output

    def _prepare_translators_for_inference(self, translators, dtype: torch.dtype = torch.float32):
        def _move_one(t):
            if hasattr(t, "aligner") and isinstance(getattr(t, "aligner"), nn.Module):
                t.aligner.to(device=device, dtype=dtype)
                for _, p in t.aligner.named_parameters(recurse=True):
                    p.data = p.data.to(device=device, dtype=dtype)
                for _, b in t.aligner.named_buffers(recurse=True):
                    b.data = b.data.to(device=device, dtype=dtype)

        if isinstance(translators, (list, tuple)):
            for t in translators:
                _move_one(t)
        else:
            _move_one(translators)

    def compute_skipping(
        self,
        precomputed_embeddings: Dict[int, torch.Tensor],
        skips: Sequence[tuple[int, int]],
        mode: int,
        to_save_translator_path=None,
        translator_key=None,
    ):
        computed_skips: Sequence[IndexedLayer] = []

        for skip_from, skip_to in skips:
            if skip_from not in precomputed_embeddings or skip_to not in precomputed_embeddings:
                raise ValueError(
                    f"Precomputed embeddings missing for skip ({skip_from}, {skip_to}). Available keys: {list(precomputed_embeddings.keys())}"
                )

            # Every span needs its own translator, so the on-disk key has to name the span.
            # Without the suffix a multi-span config saved each span's translator over the
            # last one and then loaded that single survivor back for *all* spans -- silently
            # bridging (2,4) with the map fitted for (9,10).
            span_key = span_translator_key(translator_key, skip_from, skip_to)

            if self.precomputed_translator_path:
                translators = [
                    load_translator(
                        translator_key=span_key,
                        translator_factory_name=self.translator_factory_name,
                        dir_to_load=self.precomputed_translator_path,
                    )
                ]
            else:
                translators = self.fit_translators(
                    spaces_to_fit=precomputed_embeddings,
                    skip_from=skip_from,
                    skip_to=skip_to,
                    mode=mode,
                )
                if to_save_translator_path:
                    save_translator(
                        translator=translators[0] if mode == 1 else translators,
                        translator_name=span_key,
                        dir_to_save=to_save_translator_path,
                    )

            self._prepare_translators_for_inference(translators, dtype=torch.float32)

            computed_skips.append(
                IndexedLayer(
                    index=skip_from + 1,
                    layer=partial(
                        self.transform_similar_spaces,
                        translators=translators,
                        mode=mode,
                    ),
                    layer_name=f"skip_{skip_from}_{skip_to}",
                )
            )

        return computed_skips

    def fit_translators(self, spaces_to_fit: Dict[int, torch.Tensor], skip_from: int, skip_to: int, mode: int):
        dtype = torch.float32

        x = spaces_to_fit[skip_from].to(dtype).to(device)
        y = spaces_to_fit[skip_to].to(dtype).to(device)
        sequence_length = x.shape[1]

        translators = []
        translator_factory = NAME2TRANSLATORS[self.translator_factory_name]

        if mode == 1:
            translator = translator_factory()
            x_flat = x.reshape(-1, x.shape[-1])
            y_flat = y.reshape(-1, y.shape[-1])

            translator.fit(x=x_flat, y=y_flat)
            translators.append(translator)
        elif mode == 2:
            for i in range(sequence_length):
                translator = translator_factory()
                x_i = x[:, i, :]
                y_i = y[:, i, :]
                translators.append(translator.fit(x=x_i, y=y_i))
        else:
            raise ValueError(f"Invalid mode: {mode}. Choose 1 or 2.")

        return translators

    def transform_similar_spaces(self, current_space: torch.Tensor, translators: list, mode: int):
        dtype = current_space.dtype
        x = current_space
        original_shape = x.shape
        transformed_space = None

        if mode == 1:
            # Loaded translators are wrapped in a single-element list by compute_skipping, so
            # fitted and loaded paths index identically here.
            transformed_space = translators[0].transform(x.to(dtype))[0]
            transformed_space = transformed_space.reshape(original_shape)
        elif mode == 2:
            transformed_spaces = []
            for i in range(original_shape[1]):
                x_i = x[:, i, :]
                translator = translators[i]
                transformed_spaces.append(translator.transform(x_i.to(dtype))[0])
            transformed_space = torch.stack(transformed_spaces, dim=1)
        else:
            raise ValueError(f"Invalid mode: {mode}. Choose 1 or 2.")

        return transformed_space.to(dtype)

    def filter_layers(self, layers: nn.ModuleList, skips: Sequence[tuple[int, int]], layers_accept_masks: bool, needs_position_ids: bool = False):
        filtered_layers: Sequence[IndexedLayer] = []
        skip_indices = set()
        max_layer_index = len(layers) - 1

        for start, end in skips:
            if start >= end:
                continue
            actual_start = max(0, start + 1)
            actual_end = min(max_layer_index, end)
            skip_indices.update(range(actual_start, actual_end + 1))

        def create_layer_wrapper(layer_module: nn.Module, accepts_masks: bool, needs_pos_ids: bool):
            def wrapper(
                hidden_states: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                causal_attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.Tensor] = None,
                *args,
                **kwargs,
            ) -> torch.Tensor:

                output = None
                if accepts_masks:
                    call_kwargs = dict(kwargs)
                    if needs_pos_ids and position_ids is not None:
                        call_kwargs['position_ids'] = position_ids

                    try:
                        output = layer_module(
                            hidden_states,
                            attn_mask=attention_mask,
                            **call_kwargs,
                        )
                    except (TypeError, RuntimeError) as e:
                        try:
                            output = layer_module(
                                hidden_states,
                                attention_mask=attention_mask,
                                **call_kwargs,
                            )
                        except (TypeError, RuntimeError):
                            output = layer_module(hidden_states, *args, **kwargs)
                else:
                    output = layer_module(hidden_states, *args, **kwargs)

                if isinstance(output, tuple):
                    return output[0]
                elif isinstance(output, torch.Tensor):
                    return output
                else:
                    if hasattr(output, "last_hidden_state"):
                        return output.last_hidden_state
                    else:
                        raise TypeError(
                            f"Unexpected output type {type(output)} from layer {layer_module.__class__.__name__}"
                        )

            return wrapper

        for i, layer_module in enumerate(layers):
            if i not in skip_indices:
                wrapped_layer = create_layer_wrapper(layer_module, layers_accept_masks, needs_position_ids)
                filtered_layers.append(IndexedLayer(index=i, layer=wrapped_layer, layer_name=f"original_layer_{i}"))

        print(f"Filtered layers (kept {len(filtered_layers)} out of {len(layers)})")
        return filtered_layers

    def check_skip_consistency(self):
        max_val = -1

        for a, b in sorted(self.skips):

            if a == b:
                raise ValueError(f"Skipping from {a} to {b} is invalid")

            if (a < max_val) or (b <= max_val):
                raise ValueError(f"Skips {sorted(self.skips)} overlaps")

            max_val = b

    def check_translator_consistency(self):
        if self.precomputed_translator_path:
            if not self.translator_key:
                raise ValueError("You should provide a translator_key when loading from precomputed_translator_path")

        if self.to_save_translator_path:
            if not self.translator_key:
                raise ValueError("You should provide a translator_key when using to_save_translator_path")

        if self.translator_key and not self.precomputed_translator_path and not self.to_save_translator_path:
            raise ValueError(
                "You provided a translator_key but neither precomputed_translator_path nor to_save_translator_path"
            )


class IndexedLayer:
    def __init__(self, index: int, layer: Callable, layer_name: Optional[str] = None):
        self.index = index
        self.layer = layer
        self.layer_name = layer_name or f"layer_{index}"

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.layer(*args, **kwargs)

    def __repr__(self) -> str:
        layer_repr = getattr(self.layer, "__name__", repr(self.layer))
        if isinstance(self.layer, partial):
            layer_repr = f"partial({getattr(self.layer.func, '__name__', repr(self.layer.func))})"

        if "lambda" in layer_repr:
            layer_repr = f"<lambda_wrapper_for_{self.layer_name}>"

        if hasattr(self.layer, "keywords") and "layer" in self.layer.keywords:
            layer_repr = f"{self.layer.keywords['layer'].__class__.__name__}"

        return f"IndexedLayer(index={self.index}, name={self.layer_name}, layer={layer_repr})"


def save_translator(translator, translator_name, dir_to_save: Path):

    state_dict = {k: v for k, v in translator.aligner.state_dict().items()}

    for k, v in state_dict.items():
        state_to_save = dir_to_save / translator_name / "aligner" / k
        state_to_save.parent.mkdir(exist_ok=True, parents=True)

        # clone() is load-bearing: pickling a tensor writes its whole underlying storage,
        # not just the view onto it. "linear" gets its matrix from torch.linalg.lstsq, whose
        # solution is the LAPACK workspace [max(n_rows, D), D] narrowed to [D, D] -- so the
        # uncloned tensor wrote every scratch row too. For rad-dino (D=768, 250 images x 1370
        # tokens) that was 1.05GB on disk for a 2.36MB matrix, and it filled the home quota.
        # .contiguous() alone does not help: the narrow is on dim 0, so the view is already
        # contiguous and contiguous() returns it unchanged, still sharing the big storage.
        if isinstance(v, torch.Tensor):
            v = v.detach().cpu().contiguous().clone()

        with open(state_to_save, "wb") as f:
            pickle.dump(v, f)


def load_translator(translator_key, translator_factory_name, dir_to_load: Path):
    """Rebuild a translator saved by save_translator.

    See _TRANSLATORS_WITH_UNSAVED_STATE for which translators cannot round-trip and why.
    """
    if translator_factory_name in _TRANSLATORS_WITH_UNSAVED_STATE:
        raise ValueError(
            f"translator '{translator_factory_name}' keeps fitted state outside its aligner "
            f"(x_transform/y_transform), which save_translator does not persist -- reloading "
            f"it would silently drop that state. This affects saving/loading only; the "
            f"translator still works normally when fit and used in one run. For transfer "
            f"experiments use linear, lowrank_*, rrr_* or lora_*, which have no transforms."
        )

    translator_dir = dir_to_load / translator_key
    if not translator_dir.is_dir():
        raise FileNotFoundError(
            f"No saved translator at {translator_dir}. Run the fitting config (the row whose "
            f"fit_dataset equals its dataset) before any row that transfers from it."
        )

    translator_factory = NAME2TRANSLATORS[translator_factory_name]
    translator = translator_factory()

    n_restored = 0
    for subdir in sorted(translator_dir.iterdir()):
        # Skip stray files (.DS_Store and friends) that would otherwise be treated as a
        # component name and blow up in getattr.
        if not subdir.is_dir() or subdir.name.startswith("."):
            continue
        translator_attribute = getattr(translator, subdir.name)
        for attr in sorted(subdir.iterdir()):
            if not attr.is_file() or attr.name.startswith("."):
                continue

            with open(attr, "rb") as f:
                state_value = pickle.load(f)

            translator_attribute.register_buffer(attr.name, state_value)
            n_restored += 1

    if n_restored == 0:
        raise ValueError(f"Found {translator_dir} but restored no state from it.")

    translator._fitted = True

    return translator
