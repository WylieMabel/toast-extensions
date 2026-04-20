from pathlib import Path
import pickle
from typing import Callable, Dict, Sequence, Optional, Any
from torch import nn
import torch
from functools import partial
from toast.utils.dictionaries import NAME2TRANSLATORS
from toast.utils.utils import resolve_path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
        self.precomputed_translator = None

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

        if self.precomputed_translator_path:
            self.precomputed_translator = load_translator(
                translator_key=self.translator_key,
                translator_factory_name=self.translator_factory_name,
                dir_to_load=self.precomputed_translator_path,
            )

        self.computed_skips: Sequence[IndexedLayer] = self.compute_skipping(
            self.precomputed_embeddings,
            self.skips,
            self.mode,
            self.precomputed_translator,
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
        precomputed_translator=None,
        to_save_translator_path=None,
        translator_key=None,
    ):
        computed_skips: Sequence[IndexedLayer] = []

        for skip_from, skip_to in skips:
            if skip_from not in precomputed_embeddings or skip_to not in precomputed_embeddings:
                raise ValueError(
                    f"Precomputed embeddings missing for skip ({skip_from}, {skip_to}). Available keys: {list(precomputed_embeddings.keys())}"
                )

            if precomputed_translator:
                translators = precomputed_translator
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
                        translator_name=translator_key,
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
            translator = translators if self.precomputed_translator_path else translators[0]
            transformed_space = translator.transform(x.to(dtype))[0]
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

        with open(state_to_save, "wb") as f:
            pickle.dump(v, f)


def load_translator(translator_key, translator_factory_name, dir_to_load: Path):

    translator_factory = NAME2TRANSLATORS[translator_factory_name]
    translator = translator_factory()

    translator_dir = dir_to_load / translator_key
    for subdir in translator_dir.iterdir():
        translator_attribute = getattr(translator, subdir.name)
        for attr in subdir.iterdir():
            state_key = attr.name

            with open(attr, "rb") as f:
                state_value = pickle.load(f)

            translator_attribute.register_buffer(state_key, state_value)

    translator._fitted = True

    return translator
