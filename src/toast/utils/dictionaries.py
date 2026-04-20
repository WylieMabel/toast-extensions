from typing import Any, Mapping
import torch
import torch.nn as nn
from latentis.transform.translate.aligner import Translator, MatrixAligner
from toast.modules.mlp_translator import SGDMLPAligner
from toast.modules.deepmlp_translator import SGDDeepMLPAligner
from latentis.transform.translate.functional import lstsq_align_state
from latentis.transform.base import StandardScaling
from latentis.transform import Estimator


DATASET2INPUT_COLUMN = {
    "mnist": "image",
    "fashion-mnist": "image",
    "imagenet-1k": "image",
    "cifar10": "img",
    "cifar100": "img",
    "cifar100-fine": "img",
    "cifar100-coarse": "img",
    "sceneparse150": "image",
    # text data
    "sst2": "sentence",
    "ag_news": "text",
}

DATASET2LABEL_COLUMN = {
    "mnist": "label",
    "fashion-mnist": "label",
    "imagenet-1k": "label",
    "cifar10": "label",
    "cifar100": "fine_label",
    "cifar100-fine": "fine_label",
    "cifar100-coarse": "coarse_label",
    "sceneparse150": "annotation",
    # text data
    "sst2": "label",
    "ag_news": "label",
}

DATASET2NUM_CLASSES = {
    "mnist": 10,
    "fashion-mnist": 10,
    "imagenet-1k": 1000,
    "cifar10": 10,
    "cifar100": 100,
    "cifar100-fine": 100,
    "cifar100-coarse": 20,
    "sceneparse150": 150,
    # text data
    "sst2": 2,
    "ag_news": 4,
}

MODEL_NAME2HF_NAME = {
    # VIT
    "vit-small-patch16-224": "WinKawaks/vit-small-patch16-224",
    "vit-tiny-patch16-224": "WinKawaks/vit-tiny-patch16-224",
    "vit-base-patch16-224": "google/vit-base-patch16-224",
    "vit-large-patch16-224": "google/vit-large-patch16-224",
    # DEIT
    "deit-small-patch16-224": "facebook/deit-small-patch16-224",
    "deit-base-patch16-224": "facebook/deit-base-patch16-224",
    # DINO
    "dinov2-small": "facebook/dinov2-small",
    "dinov2-base": "facebook/dinov2-base",
    # CLIP
    "clip-base": "openai/clip-vit-base-patch32",
    # TIMM (DC-ViT Table 5 checkpoints)
    "vit-tiny-patch16-224-augreg": "timm/vit_tiny_patch16_224.augreg_in21k_ft_in1k",
    "vit-small-patch16-224-augreg": "timm/vit_small_patch16_224.augreg_in21k_ft_in1k",
    "vit-large-patch16-224-augreg": "timm/vit_large_patch16_224.augreg_in21k_ft_in1k",
    "deit-base-patch16-224-fb": "timm/deit_base_patch16_224.fb_in1k",
    # TEXT
    "modern-bert-base": "answerdotai/ModernBERT-base",
    "bert-base": "google-bert/bert-base-uncased",
    "roberta-base": "FacebookAI/xlm-roberta-base",
}

DATASET_NAME2HF_NAME = {
    "mnist": "mnist",
    "fashion-mnist": "zalando-datasets/fashion_mnist",
    "imagenet-1k": "ILSVRC/imagenet-1k",
    "cifar10": "cifar10",
    "cifar100": "cifar100",
    "cifar100-fine": "cifar100",
    "cifar100-coarse": "cifar100",
    "sceneparse150": "zhoubolei/scene_parse_150",
    # text data
    "sst2": "glue",
    "imdb": "imdb",
    "ag_news": "ag_news",
    "yelp_polarity": "yelp_polarity",
}

# Text dataset configurations with full HuggingFace loading details
TEXT_DATASET_CONFIGS = {
    "ag_news": {
        "hf_name": "ag_news",
        "hf_subset": None,
        "text_column": "text",
        "label_column": "label",
        "num_classes": 4,
        "train_split": "train",
        "test_split": "test",
    },
    "sst2": {
        "hf_name": "nyu-mll/glue",
        "hf_subset": "sst2",
        "text_column": "sentence",
        "label_column": "label",
        "num_classes": 2,
        "train_split": "train",
        "test_split": "validation",
    },
    "mnli": {
        "hf_name": "nyu-mll/glue",
        "hf_subset": "mnli",
        "text_column": ["premise", "hypothesis"],
        "label_column": "label",
        "num_classes": 3,
        "train_split": "train",
        "test_split": "validation_matched",
    },
}

MODEL2NUM_LAYERS = {
    "WinKawaks/vit-small-patch16-224": 12,
    "WinKawaks/vit-tiny-patch16-224": 12,
    "facebook/deit-small-patch16-224": 12,
    "facebook/deit-base-patch16-224": 12,
    "google/vit-base-patch16-224": 12,
    "google/vit-large-patch16-224": 24,
    "facebook/dinov2-small": 12,
    "facebook/dinov2-base": 12,
    "openai/clip-vit-base-patch32": 12,
    # TIMM (DC-ViT Table 5 checkpoints)
    "timm/vit_tiny_patch16_224.augreg_in21k_ft_in1k": 12,
    "timm/vit_small_patch16_224.augreg_in21k_ft_in1k": 12,
    "timm/vit_large_patch16_224.augreg_in21k_ft_in1k": 24,
    "timm/deit_base_patch16_224.fb_in1k": 12,
    # Text models
    "answerdotai/ModernBERT-base": 22,
}

MODEL2CONFIGS = {
    "facebook/dinov2-small": {
        "embeddings_path": "embeddings",
        "layers_parent_path": "encoder",
        "layers_attribute_name": "layer",
        "pre_norm_path": None,
        "post_norm_path": "layernorm",
        "pooler_path": None,
        "layers_accept_masks": False,
    },
    "facebook/dinov2-base": {
        "embeddings_path": "embeddings",
        "layers_parent_path": "encoder",
        "layers_attribute_name": "layer",
        "pre_norm_path": None,
        "post_norm_path": "layernorm",
        "pooler_path": None,
        "layers_accept_masks": False,
    },
    "WinKawaks/vit-tiny-patch16-224": {
        "embeddings_path": "embeddings",
        "layers_parent_path": "encoder",
        "layers_attribute_name": "layer",
        "pre_norm_path": None,
        "post_norm_path": "layernorm",
        "pooler_path": "pooler",
        "layers_accept_masks": False,
    },
    "WinKawaks/vit-small-patch16-224": {
        "embeddings_path": "embeddings",
        "layers_parent_path": "encoder",
        "layers_attribute_name": "layer",
        "pre_norm_path": None,
        "post_norm_path": "layernorm",
        "pooler_path": "pooler",
        "layers_accept_masks": False,
    },
    "google/vit-base-patch16-224": {
        "embeddings_path": "embeddings",
        "layers_parent_path": "encoder",
        "layers_attribute_name": "layer",
        "pre_norm_path": None,
        "post_norm_path": "layernorm",
        "pooler_path": "pooler",
        "layers_accept_masks": False,
    },
    "google/vit-large-patch16-224": {
        "embeddings_path": "embeddings",
        "layers_parent_path": "encoder",
        "layers_attribute_name": "layer",
        "pre_norm_path": None,
        "post_norm_path": "layernorm",
        "pooler_path": "pooler",
        "layers_accept_masks": False,
    },
    "facebook/deit-small-patch16-224": {
        "embeddings_path": "embeddings",
        "layers_parent_path": "encoder",
        "layers_attribute_name": "layer",
        "pre_norm_path": None,
        "post_norm_path": "layernorm",
        "pooler_path": "pooler",
        "layers_accept_masks": False,
    },
    "facebook/deit-base-patch16-224": {
        "embeddings_path": "embeddings",
        "layers_parent_path": "encoder",
        "layers_attribute_name": "layer",
        "pre_norm_path": None,
        "post_norm_path": "layernorm",
        "pooler_path": "pooler",
        "layers_accept_masks": False,
    },
    "openai/clip-vit-base-patch32": {
        "embeddings_path": "vision_model.embeddings",
        "layers_parent_path": "vision_model.encoder",
        "layers_attribute_name": "layers",
        "pre_norm_path": "vision_model.pre_layrnorm",
        "post_norm_path": "vision_model.post_layernorm",
        "pooler_path": None,
        "layers_accept_masks": True,
    },
    "timm/vit_tiny_patch16_224.augreg_in21k_ft_in1k": {
        "embeddings_path": "embeddings",
        "layers_parent_path": "",
        "layers_attribute_name": "blocks",
        "pre_norm_path": None,
        "post_norm_path": "norm",
        "pooler_path": None,
        "layers_accept_masks": False,
    },
    "timm/vit_small_patch16_224.augreg_in21k_ft_in1k": {
        "embeddings_path": "embeddings",
        "layers_parent_path": "",
        "layers_attribute_name": "blocks",
        "pre_norm_path": None,
        "post_norm_path": "norm",
        "pooler_path": None,
        "layers_accept_masks": False,
    },
    "timm/vit_large_patch16_224.augreg_in21k_ft_in1k": {
        "embeddings_path": "embeddings",
        "layers_parent_path": "",
        "layers_attribute_name": "blocks",
        "pre_norm_path": None,
        "post_norm_path": "norm",
        "pooler_path": None,
        "layers_accept_masks": False,
    },
    "timm/deit_base_patch16_224.fb_in1k": {
        "embeddings_path": "embeddings",
        "layers_parent_path": "",
        "layers_attribute_name": "blocks",
        "pre_norm_path": None,
        "post_norm_path": "norm",
        "pooler_path": None,
        "layers_accept_masks": False,
    },
    "open_clip:laion/CLIP-ViT-B-16-laion2B-s34B-b88K": {
        "embeddings_path": "visual.conv1",
        "layers_parent_path": "visual.transformer",
        "layers_attribute_name": "resblocks",
        "pre_norm_path": "visual.ln_pre",
        "post_norm_path": "visual.ln_post",
        "pooler_path": None,
        "layers_accept_masks": True,
        "needs_conv1_processing": True,
        "class_embedding_path": "visual.class_embedding",
        "positional_embedding_path": "visual.positional_embedding",
        "embedding_dropout_path": "visual.patch_dropout",
    },
    # Text models
    "bert-base": {
        "embeddings_path": "embeddings",
        "layers_parent_path": "encoder",
        "layers_attribute_name": "layer",
        "pre_norm_path": None,
        "post_norm_path": None,
        "pooler_path": "pooler",
        "layers_accept_masks": True,
    },
    "roberta-base": {
        "embeddings_path": "embeddings",
        "layers_parent_path": "encoder",
        "layers_attribute_name": "layer",
        "pre_norm_path": None,
        "post_norm_path": None,
        "pooler_path": "pooler",
        "layers_accept_masks": True,
    },
    "modern-bert-base": {
        "embeddings_path": "embeddings",
        "layers_parent_path": "",
        "layers_attribute_name": "layers",
        "pre_norm_path": None,
        "post_norm_path": "final_norm",
        "pooler_path": None,
        "layers_accept_masks": True,
        "needs_position_ids": True,
    },
    "answerdotai/ModernBERT-base": {
        "embeddings_path": "embeddings",
        "layers_parent_path": "",
        "layers_attribute_name": "layers",
        "pre_norm_path": None,
        "post_norm_path": "final_norm",
        "pooler_path": None,
        "layers_accept_masks": True,
        "needs_position_ids": True,
    },
}


class IdentityTranslator(Estimator):
    def __init__(
        self,
    ) -> None:
        super().__init__(name="Identity")

    def fit(self, x: torch.Tensor, y: torch.Tensor) -> Mapping[str, Any]:
        return self

    def transform(self, x: torch.Tensor, y=None) -> torch.Tensor:
        return x, y


NAME2TRANSLATORS = {
    "linear": lambda: Translator(
        aligner=MatrixAligner(name="linear", align_fn_state=lstsq_align_state),
    ),
    "sgd_mlp_aligner": lambda: Translator(
        aligner=SGDMLPAligner(num_steps=50, lr=1e-3, random_seed=0),
        x_transform=StandardScaling(),
        y_transform=StandardScaling(),
    ),
    "identity": lambda: IdentityTranslator(),
    "mlp": lambda: SGDMLPAligner(num_steps=300, lr=1e-3, random_seed=0),
    "deep_mlp": lambda: SGDDeepMLPAligner(num_steps=300, lr=1e-3, random_seed=0),
}
