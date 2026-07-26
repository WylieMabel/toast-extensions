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
    # MedMNIST datasets
    "pathmnist": "image",
    "chestmnist": "image",
    "dermamnist": "image",
    "pneumoniamnist": "image",
    "retinamnist": "image",
    "breastmnist": "image",
    "bloodmnist": "image",
    "tissuemnist": "image",
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
    # MedMNIST datasets
    "pathmnist": "label",
    "chestmnist": "label",
    "dermamnist": "label",
    "pneumoniamnist": "label",
    "retinamnist": "label",
    "breastmnist": "label",
    "bloodmnist": "label",
    "tissuemnist": "label",
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
    # MedMNIST datasets
    "pathmnist": 9,
    "chestmnist": 2,
    "dermamnist": 7,
    "pneumoniamnist": 2,
    "retinamnist": 5,
    "breastmnist": 2,
    "bloodmnist": 8,
    "tissuemnist": 8,
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
    "rad-dino": "microsoft/rad-dino",
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
    # MedMNIST datasets
    "pathmnist": "albertvillanova/medmnist-v2",
    "chestmnist": "albertvillanova/medmnist-v2",
    "dermamnist": "albertvillanova/medmnist-v2",
    "pneumoniamnist": "albertvillanova/medmnist-v2",
    "retinamnist": "albertvillanova/medmnist-v2",
    "breastmnist": "albertvillanova/medmnist-v2",
    "bloodmnist": "albertvillanova/medmnist-v2",
    "tissuemnist": "albertvillanova/medmnist-v2",
    # text data
    "sst2": "glue",
    "imdb": "imdb",
    "ag_news": "ag_news",
    "yelp_polarity": "yelp_polarity",
}

DATASET2LOCAL_PATH = {
    "mnist": "/cluster/customapps/biomed/vogtlab/users/mwylie/toast/mnist_clean",
    "fashion-mnist": "/cluster/customapps/biomed/vogtlab/users/mwylie/toast/fashion_mnist_clean",
    "cifar10": "/cluster/customapps/biomed/vogtlab/users/mwylie/toast/cifar10_clean",
    "cifar100": "/cluster/customapps/biomed/vogtlab/users/mwylie/toast/cifar100_clean",
    "imagenet-1k": "/cluster/customapps/biomed/vogtlab/users/mwylie/toast/imagenet1k_clean",
    # MedMNIST datasets
    "pathmnist": "/cluster/customapps/biomed/vogtlab/users/mwylie/toast/medmnist_pathmnist_clean",
    "chestmnist": "/cluster/customapps/biomed/vogtlab/users/mwylie/toast/medmnist_chestmnist_clean",
    "dermamnist": "/cluster/customapps/biomed/vogtlab/users/mwylie/toast/medmnist_dermamnist_clean",
    "pneumoniamnist": "/cluster/customapps/biomed/vogtlab/users/mwylie/toast/medmnist_pneumoniamnist_clean",
    "retinamnist": "/cluster/customapps/biomed/vogtlab/users/mwylie/toast/medmnist_retinamnist_clean",
    "breastmnist": "/cluster/customapps/biomed/vogtlab/users/mwylie/toast/medmnist_breastmnist_clean",
    "bloodmnist": "/cluster/customapps/biomed/vogtlab/users/mwylie/toast/medmnist_bloodmnist_clean",
    "tissuemnist": "/cluster/customapps/biomed/vogtlab/users/mwylie/toast/medmnist_tissuemnist_clean",
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
    "microsoft/rad-dino": 12,

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
    "microsoft/rad-dino": {
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

# Parameter savings per single edit, keyed by HF encoder name (matching
# MODEL2NUM_LAYERS / MODEL2CONFIGS so the training loop can look an encoder up directly).
# branch = sublayer + its LayerNorm (+ LayerScale for DINOv2).
# drop_*   = identity-mode saving (branch removed outright).
# linear_* = branch replaced by Linear(d, d, bias=True); equals drop_* minus (d*d + d).
# Use params_saved_for_config() below to total these for a full config row.

PARAM_SAVINGS = {
    "WinKawaks/vit-tiny-patch16-224": {
        "d": 192, "heads": 3, "head_dim": 64, "mlp": 768, "layers": 12,
        "params_per_block": 444864, "encoder_body_params": 5338368,
        "drop_head": 49344,
        "drop_mlp": 296256,
        "drop_attn": 148608,
        "drop_block": 444864,
        "linear_mlp": 259200,
        "linear_attn": 111552,
        "linear_block": 407808,
    },
    "WinKawaks/vit-small-patch16-224": {
        "d": 384, "heads": 6, "head_dim": 64, "mlp": 1536, "layers": 12,
        "params_per_block": 1774464, "encoder_body_params": 21293568,
        "drop_head": 98496,
        "drop_mlp": 1182336,
        "drop_attn": 592128,
        "drop_block": 1774464,
        "linear_mlp": 1034496,
        "linear_attn": 444288,
        "linear_block": 1626624,
    },
    "google/vit-base-patch16-224": {
        "d": 768, "heads": 12, "head_dim": 64, "mlp": 3072, "layers": 12,
        "params_per_block": 7087872, "encoder_body_params": 85054464,
        "drop_head": 196800,
        "drop_mlp": 4723968,
        "drop_attn": 2363904,
        "drop_block": 7087872,
        "linear_mlp": 4133376,
        "linear_attn": 1773312,
        "linear_block": 6497280,
    },
    "google/vit-large-patch16-224": {
        "d": 1024, "heads": 16, "head_dim": 64, "mlp": 4096, "layers": 24,
        "params_per_block": 12596224, "encoder_body_params": 302309376,
        "drop_head": 262336,
        "drop_mlp": 8395776,
        "drop_attn": 4200448,
        "drop_block": 12596224,
        "linear_mlp": 7346176,
        "linear_attn": 3150848,
        "linear_block": 11546624,
    },
    "facebook/deit-small-patch16-224": {
        "d": 384, "heads": 6, "head_dim": 64, "mlp": 1536, "layers": 12,
        "params_per_block": 1774464, "encoder_body_params": 21293568,
        "drop_head": 98496,
        "drop_mlp": 1182336,
        "drop_attn": 592128,
        "drop_block": 1774464,
        "linear_mlp": 1034496,
        "linear_attn": 444288,
        "linear_block": 1626624,
    },
    "facebook/deit-base-patch16-224": {
        "d": 768, "heads": 12, "head_dim": 64, "mlp": 3072, "layers": 12,
        "params_per_block": 7087872, "encoder_body_params": 85054464,
        "drop_head": 196800,
        "drop_mlp": 4723968,
        "drop_attn": 2363904,
        "drop_block": 7087872,
        "linear_mlp": 4133376,
        "linear_attn": 1773312,
        "linear_block": 6497280,
    },
    "facebook/dinov2-small": {
        "d": 384, "heads": 6, "head_dim": 64, "mlp": 1536, "layers": 12,
        "params_per_block": 1775232, "encoder_body_params": 21302784,
        "drop_head": 98496,
        "drop_mlp": 1182720,
        "drop_attn": 592512,
        "drop_block": 1775232,
        "linear_mlp": 1034880,
        "linear_attn": 444672,
        "linear_block": 1627392,
    },
    "facebook/dinov2-base": {
        "d": 768, "heads": 12, "head_dim": 64, "mlp": 3072, "layers": 12,
        "params_per_block": 7089408, "encoder_body_params": 85072896,
        "drop_head": 196800,
        "drop_mlp": 4724736,
        "drop_attn": 2364672,
        "drop_block": 7089408,
        "linear_mlp": 4134144,
        "linear_attn": 1774080,
        "linear_block": 6498816,
    },
    "microsoft/rad-dino": {  # DINOv2-base arch, patch14/518px
        "d": 768, "heads": 12, "head_dim": 64, "mlp": 3072, "layers": 12,
        "params_per_block": 7089408, "encoder_body_params": 85072896,
        "drop_head": 196800,
        "drop_mlp": 4724736,
        "drop_attn": 2364672,
        "drop_block": 7089408,
        "linear_mlp": 4134144,
        "linear_attn": 1774080,
        "linear_block": 6498816,
    },
    "openai/clip-vit-base-patch32": {  # vision tower only
        "d": 768, "heads": 12, "head_dim": 64, "mlp": 3072, "layers": 12,
        "params_per_block": 7087872, "encoder_body_params": 85054464,
        "drop_head": 196800,
        "drop_mlp": 4723968,
        "drop_attn": 2363904,
        "drop_block": 7087872,
        "linear_mlp": 4133376,
        "linear_attn": 1773312,
        "linear_block": 6497280,
    },
    "timm/vit_tiny_patch16_224.augreg_in21k_ft_in1k": {
        "d": 192, "heads": 3, "head_dim": 64, "mlp": 768, "layers": 12,
        "params_per_block": 444864, "encoder_body_params": 5338368,
        "drop_head": 49344,
        "drop_mlp": 296256,
        "drop_attn": 148608,
        "drop_block": 444864,
        "linear_mlp": 259200,
        "linear_attn": 111552,
        "linear_block": 407808,
    },
    "timm/vit_small_patch16_224.augreg_in21k_ft_in1k": {
        "d": 384, "heads": 6, "head_dim": 64, "mlp": 1536, "layers": 12,
        "params_per_block": 1774464, "encoder_body_params": 21293568,
        "drop_head": 98496,
        "drop_mlp": 1182336,
        "drop_attn": 592128,
        "drop_block": 1774464,
        "linear_mlp": 1034496,
        "linear_attn": 444288,
        "linear_block": 1626624,
    },
    "timm/vit_large_patch16_224.augreg_in21k_ft_in1k": {
        "d": 1024, "heads": 16, "head_dim": 64, "mlp": 4096, "layers": 24,
        "params_per_block": 12596224, "encoder_body_params": 302309376,
        "drop_head": 262336,
        "drop_mlp": 8395776,
        "drop_attn": 4200448,
        "drop_block": 12596224,
        "linear_mlp": 7346176,
        "linear_attn": 3150848,
        "linear_block": 11546624,
    },
    "timm/deit_base_patch16_224.fb_in1k": {
        "d": 768, "heads": 12, "head_dim": 64, "mlp": 3072, "layers": 12,
        "params_per_block": 7087872, "encoder_body_params": 85054464,
        "drop_head": 196800,
        "drop_mlp": 4723968,
        "drop_attn": 2363904,
        "drop_block": 7087872,
        "linear_mlp": 4133376,
        "linear_attn": 1773312,
        "linear_block": 6497280,
    },
    "answerdotai/ModernBERT-base": {  # layer 0 attn_norm is Identity
        "d": 768, "heads": 12, "head_dim": 64, "mlp": 1152, "layers": 22,
        "params_per_block": 5015040, "encoder_body_params": 110330880,
        "drop_head": 196608,
        "drop_mlp": 2654976,
        "drop_attn": 2360064,
        "drop_block": 5015040,
        "linear_mlp": 2064384,
        "linear_attn": 1769472,
        "linear_block": 4424448,
    },
    "google-bert/bert-base-uncased": {
        "d": 768, "heads": 12, "head_dim": 64, "mlp": 3072, "layers": 12,
        "params_per_block": 7087872, "encoder_body_params": 85054464,
        "drop_head": 196800,
        "drop_mlp": 4723968,
        "drop_attn": 2363904,
        "drop_block": 7087872,
        "linear_mlp": 4133376,
        "linear_attn": 1773312,
        "linear_block": 6497280,
    },
    "FacebookAI/xlm-roberta-base": {
        "d": 768, "heads": 12, "head_dim": 64, "mlp": 3072, "layers": 12,
        "params_per_block": 7087872, "encoder_body_params": 85054464,
        "drop_head": 196800,
        "drop_mlp": 4723968,
        "drop_attn": 2363904,
        "drop_block": 7087872,
        "linear_mlp": 4133376,
        "linear_attn": 1773312,
        "linear_block": 6497280,
    },
    "clip-base-text-tower": {  # not an encoder in MODEL_NAME2HF_NAME; kept under its own key
        "d": 512, "heads": 8, "head_dim": 64, "mlp": 2048, "layers": 12,
        "params_per_block": 3152384, "encoder_body_params": 37828608,
        "drop_head": 131264,
        "drop_mlp": 2100736,
        "drop_attn": 1051648,
        "drop_block": 3152384,
        "linear_mlp": 1838080,
        "linear_attn": 788992,
        "linear_block": 2889728,
    },
}


def _resolve_param_savings(encoder):
    """PARAM_SAVINGS entry for an encoder given by HF name (preferred) or short name."""
    if encoder in PARAM_SAVINGS:
        return PARAM_SAVINGS[encoder]
    hf = MODEL_NAME2HF_NAME.get(encoder)
    if hf is not None and hf in PARAM_SAVINGS:
        return PARAM_SAVINGS[hf]
    return None


def _translator_cost(name, d):
    """Params added by one skip-bridge translator. identity is free; linear and the
    learned aligners are all costed as a single Linear(d, d, bias=True) = d*d + d."""
    return 0 if name == "identity" else d * d + d


def params_saved_for_config(
    encoder,
    skip=None,
    mlp_skip=None,
    attn_skip=None,
    head_dict=None,
    skip_translator="identity",
    mlp_mode="identity",
    attn_mode="identity",
):
    """Total encoder-body params removed by one config row, and that as a percent of the
    intact body. Returns {"params_saved", "encoder_body_params", "params_saved_pct"};
    params_saved is 0 for the baseline (all-empty) row. If the encoder has no
    PARAM_SAVINGS entry, all three values come back as None.

    Savings honor the per-edit mode: sub-block edits use drop_* under identity mode and
    linear_* otherwise; each whole-block skip (skip_from, skip_to) frees
    (skip_to - skip_from) blocks minus one skip_translator bridge; head_dict maps a layer
    to the heads it keeps, so (heads - kept) heads are pruned per listed layer.

    Double counting is suppressed with a fixed precedence -- whole block > mlp/attention >
    heads -- so an edit is never counted for params another edit already removed:
      * a layer inside a dropped block contributes only drop_block (its mlp/attn/head
        edits are ignored, since that layer's sublayers are already gone);
      * a layer whose attention is dropped ignores any head pruning on it;
      * overlapping skip spans count each block once; repeated indices count once.
    """
    entry = _resolve_param_savings(encoder)
    if entry is None:
        return {"params_saved": None, "encoder_body_params": None, "params_saved_pct": None}

    d = entry["d"]
    saved = 0

    # Layers whose whole block is removed (union across spans -> each block counted once).
    dropped_layers = set()
    for skip_from, skip_to in (skip or []):
        dropped_layers.update(range(skip_from + 1, skip_to + 1))
    saved += len(dropped_layers) * entry["drop_block"]
    # one bridge translator per span (identity is free, linear/learned cost d*d + d).
    saved -= len(skip or []) * _translator_cost(skip_translator, d)

    # MLP sublayers not already inside a dropped block.
    for i in set(mlp_skip or []):
        if i not in dropped_layers:
            saved += entry["drop_mlp"] if mlp_mode == "identity" else entry["linear_mlp"]

    # Attention sublayers not already inside a dropped block.
    attn_layers = set(attn_skip or [])
    for i in attn_layers:
        if i not in dropped_layers:
            saved += entry["drop_attn"] if attn_mode == "identity" else entry["linear_attn"]

    # Head pruning, skipped where the whole block OR its attention sublayer is already gone.
    for layer, kept in (head_dict or {}).items():
        if layer in dropped_layers or layer in attn_layers:
            continue
        pruned = entry["heads"] - len(kept)
        if pruned > 0:
            saved += pruned * entry["drop_head"]

    body = entry["encoder_body_params"]
    pct = 100.0 * saved / body if body else 0.0
    return {"params_saved": int(saved), "encoder_body_params": body, "params_saved_pct": pct}