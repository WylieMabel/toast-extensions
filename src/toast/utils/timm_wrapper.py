from types import SimpleNamespace

import timm
import torch
import torch.nn as nn
from timm.data import create_transform, resolve_data_config


TIMM_PREFIX = "timm/"


def is_timm_model(name: str) -> bool:
    return name.startswith(TIMM_PREFIX)


def _strip_prefix(name: str) -> str:
    return name[len(TIMM_PREFIX):]


class TimmViTEmbeddings(nn.Module):

    def __init__(self, timm_model):
        super().__init__()
        self.patch_embed = timm_model.patch_embed
        self.cls_token = timm_model.cls_token
        self.pos_embed = timm_model.pos_embed
        self.pos_drop = timm_model.pos_drop

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(pixel_values)
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        return x


class TimmViTWrapper(nn.Module):

    def __init__(self, timm_model):
        super().__init__()
        self.embeddings = TimmViTEmbeddings(timm_model)
        self.blocks = timm_model.blocks
        self.norm = timm_model.norm

        self.config = SimpleNamespace(
            hidden_size=timm_model.embed_dim,
            num_hidden_layers=len(timm_model.blocks),
            patch_size=timm_model.patch_embed.patch_size[0],
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        x = self.embeddings(pixel_values)
        x = self.blocks(x)
        x = self.norm(x)
        return x


class TimmImageProcessor:

    def __init__(self, timm_model_name: str):
        data_cfg = resolve_data_config(model=timm.create_model(timm_model_name, pretrained=False))
        self._transform = create_transform(**data_cfg, is_training=False)

    def __call__(self, images, return_tensors="pt", **kwargs):
        if not isinstance(images, (list, tuple)):
            images = [images]
        tensors = [self._transform(img.convert("RGB")) for img in images]
        pixel_values = torch.stack(tensors)
        return {"pixel_values": pixel_values}


def load_timm_encoder(name: str, pretrained: bool = True) -> TimmViTWrapper:
    timm_name = _strip_prefix(name)
    raw = timm.create_model(timm_name, pretrained=pretrained)
    raw.eval()
    return TimmViTWrapper(raw)


def load_timm_processor(name: str) -> TimmImageProcessor:
    timm_name = _strip_prefix(name)
    return TimmImageProcessor(timm_name)


def get_timm_classifier_head(name: str) -> nn.Module:
    timm_name = _strip_prefix(name)
    raw = timm.create_model(timm_name, pretrained=True)
    return raw.head
