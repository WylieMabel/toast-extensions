import copy
import functools
import os
import shutil

import fire
import torch
from datasets import (
    DatasetDict,
    load_dataset,
    load_from_disk,
)
from pytorch_lightning import seed_everything
from transformers import (
    AutoModel,
    AutoConfig,
    AutoImageProcessor,
    CLIPVisionConfig,
    CLIPImageProcessor,
    CLIPVisionModel,
)
from tqdm import tqdm
from torch.utils.data import DataLoader

from toast import PROJECT_ROOT
from toast.utils.dictionaries import (
    DATASET2INPUT_COLUMN,
    DATASET2LABEL_COLUMN,
    DATASET2LOCAL_PATH,
    DATASET_NAME2HF_NAME,
    MODEL2CONFIGS,
)
from toast.utils.utils import image_encode, extract_representations, open_clip_image_encode
from toast.modules.module import SkipModel, MLPLinearisedEncoder, AttentionLinearisedEncoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

MLP_MODE = "zero"  # "zero" or "fitted"


@torch.no_grad()
def encode_data(loader, skip_encoder):
    embeddings = []
    skip_encoder.eval()
    for batch in tqdm(loader, desc="Encoding Batches"):
        image_input = batch.get("pixel_values", batch.get("images"))
        if image_input is None:
            raise KeyError("Batch missing required key 'pixel_values' or 'images'")
        image_input = image_input.to(device)
        attn_mask = batch.get("attention_mask", None)
        if attn_mask is not None:
            attn_mask = attn_mask.to(device)
        x = skip_encoder(image_input, attention_mask=attn_mask)
        embeddings.extend(x.cpu().tolist())
    return embeddings


def _parse_list_of_lists(val):
    import ast
    if isinstance(val, str):
        val = ast.literal_eval(val)
    # Single list like [1,2] → wrap to [[1,2]]; already [[],[1,2]] → keep
    if not val or not isinstance(val[0], list):
        val = [val]
    return val


@torch.no_grad()
def run_encoding(
    dataset_name: str,
    encoder_name: str,
    translator_name: str,
    seed: int,
    skips: str = "[[], [(0, 1)]]",
    mlp_skips: str = "[[]]",
    attention_skips: str = "[[]]",
    samples_to_extract: int = 500,
    batch_size: int = 32,
    mode: int = 1,
):
    seed_everything(seed)

    skips = _parse_list_of_lists(skips)
    mlp_skips = _parse_list_of_lists(mlp_skips)
    attention_skips = _parse_list_of_lists(attention_skips)

    print(f"Skips: {skips} | MLP skips: {mlp_skips} | Attention skips: {attention_skips}")

    if encoder_name not in MODEL2CONFIGS:
        raise ValueError(f"Model config not found for {encoder_name}.")
    model_config = MODEL2CONFIGS[encoder_name]

    DATASET_DIR = (
        PROJECT_ROOT
        / "data"
        / f"{translator_name}_skipped_embeddings_full"
        / dataset_name
        / encoder_name.split("/")[1]
        / str(samples_to_extract)
    )

    # Always load raw data (with images) for DataLoaders — never stripped
    if dataset_name == "svhn":
        raw_data = DatasetDict(
            train=load_dataset(DATASET_NAME2HF_NAME[dataset_name], "cropped_digits", split="train"),
            test=load_dataset(DATASET_NAME2HF_NAME[dataset_name], "cropped_digits", split="test"),
        )
    else:
        if dataset_name not in DATASET2LOCAL_PATH:
            raise ValueError(f"No local path configured for dataset '{dataset_name}'. Add it to DATASET2LOCAL_PATH in dictionaries.py.")
        dataset_path = DATASET2LOCAL_PATH[dataset_name]
        if not os.path.exists(dataset_path):
            raise ValueError(f"Dataset path does not exist: {dataset_path}")
        raw_data = load_from_disk(dataset_path)

    # Separate embeddings dataset: labels only, grows with each encoded column
    label_col = DATASET2LABEL_COLUMN[dataset_name]
    if (DATASET_DIR / "dataset_dict.json").exists():
        print(f"Loading existing dataset from {DATASET_DIR}")
        data: DatasetDict = load_from_disk(dataset_path=str(DATASET_DIR))
    else:
        print(f"Creating dataset dir: {DATASET_DIR}")
        DATASET_DIR.mkdir(parents=True, exist_ok=True)
        data = DatasetDict({
            split: raw_data[split].select_columns([label_col])
            for split in raw_data.keys()
        })

    if encoder_name.startswith("open_clip:"):
        import open_clip
        open_clip_hub_name = f"hf-hub:{encoder_name.split(':', 1)[1]}"
        model, _, preprocess_val = open_clip.create_model_and_transforms(open_clip_hub_name, device=device)
        encoder = model
        collate_fn = functools.partial(
            open_clip_image_encode,
            processor=preprocess_val,
            image_name=DATASET2INPUT_COLUMN[dataset_name],
            label_name=DATASET2LABEL_COLUMN[dataset_name],
        )
    elif encoder_name == "openai/clip-vit-base-patch32":
        config = CLIPVisionConfig.from_pretrained(encoder_name, output_hidden_states=True, return_dict=True)
        processor = CLIPImageProcessor.from_pretrained(encoder_name)
        encoder = CLIPVisionModel.from_pretrained(encoder_name, config=config)
        collate_fn = functools.partial(
            image_encode,
            processor=processor,
            image_name=DATASET2INPUT_COLUMN[dataset_name],
            label_name=DATASET2LABEL_COLUMN[dataset_name],
        )
    else:
        config = AutoConfig.from_pretrained(encoder_name, output_hidden_states=True, return_dict=True)
        processor = AutoImageProcessor.from_pretrained(encoder_name)
        encoder = AutoModel.from_pretrained(encoder_name, config=config)
        collate_fn = functools.partial(
            image_encode,
            processor=processor,
            image_name=DATASET2INPUT_COLUMN[dataset_name],
            label_name=DATASET2LABEL_COLUMN[dataset_name],
        )

    encoder.eval().to(device)

    train_loader = DataLoader(
        raw_data["train"], batch_size=batch_size, pin_memory=True,
        shuffle=False, num_workers=1, collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        raw_data["test"], batch_size=batch_size, pin_memory=True,
        shuffle=False, num_workers=1, collate_fn=collate_fn,
    )

    all_layer_embeddings = extract_representations(
        encoder=encoder,
        max_samples=samples_to_extract,
        loader=train_loader,
        model_config=model_config,
        model_is_open_clip=encoder_name.startswith("open_clip:"),
        seed=seed,
    )
    print(f"Captured embeddings for layers: {list(all_layer_embeddings.keys())}")

    total = len(skips) * len(mlp_skips) * len(attention_skips)
    combo_idx = 0
    for skip in skips:
        for attn_skip in attention_skips:
            for mlp_skip in mlp_skips:
                combo_idx += 1
                print(f"\n[{combo_idx}/{total}] skip={skip} | attn={attn_skip} | mlp={mlp_skip}")

                column_name = f"skip={skip}_mlp={mlp_skip}_attn={attn_skip}"

                if column_name in data["train"].column_names and column_name in data["test"].column_names:
                    print(f"Column '{column_name}' already exists, skipping.")
                    continue

                # Use a fresh deep copy so patches from previous combinations don't accumulate
                combo_encoder = copy.deepcopy(encoder)

                AttentionLinearisedEncoder(
                    combo_encoder,
                    attention_layers_to_linearize=attn_skip,
                    mode="zero",
                ).to(device).eval()

                mlp_enc = MLPLinearisedEncoder(
                    combo_encoder,
                    mlp_layers_to_linearize=mlp_skip,
                    mode=MLP_MODE,
                ).to(device).eval()
                if MLP_MODE == "fitted":
                    mlp_enc.fit(train_loader, max_samples=samples_to_extract)

                skip_encoder = SkipModel(
                    encoder=combo_encoder,
                    skips=skip,
                    mode=mode,
                    precomputed_embeddings=all_layer_embeddings,
                    translator_factory_name=translator_name,
                    **model_config,
                ).to(device).eval()

                split2encoding = {
                    "train": encode_data(loader=train_loader, skip_encoder=skip_encoder),
                    "test": encode_data(loader=test_loader, skip_encoder=skip_encoder),
                }

                print("Saving results to disk...")
                for split_name, encoding in split2encoding.items():
                    if not encoding:
                        print(f"Warning: no embeddings for split '{split_name}'. Skipping.")
                        continue
                    if len(encoding) != len(raw_data[split_name]):
                        print(f"Error: length mismatch for split '{split_name}'. Skipping.")
                        continue
                    if column_name in data[split_name].column_names:
                        data[split_name] = data[split_name].remove_columns([column_name])
                    data[split_name] = data[split_name].add_column(column_name, encoding)

                del skip_encoder, combo_encoder
                torch.cuda.empty_cache()

                temp_dir = DATASET_DIR.parent / f"{DATASET_DIR.name}_temp"
                try:
                    if temp_dir.exists():
                        shutil.rmtree(temp_dir)
                    data.save_to_disk(str(temp_dir))
                    if DATASET_DIR.exists():
                        shutil.rmtree(DATASET_DIR)
                    shutil.move(str(temp_dir), DATASET_DIR)
                    print(f"Saved to {DATASET_DIR}")
                    data = load_from_disk(str(DATASET_DIR))
                except Exception as e:
                    print(f"Error saving: {e}")
                    if temp_dir.exists():
                        shutil.rmtree(temp_dir)


if __name__ == "__main__":
    fire.Fire(run_encoding)
