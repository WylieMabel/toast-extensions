import functools
import os
import shutil

import fire
import torch
from datasets import (
    DatasetDict,
    DownloadConfig,
    VerificationMode,
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
    DATASET_NAME2HF_NAME,
    MODEL2CONFIGS,
)
from toast.utils.utils import image_encode, extract_representations, open_clip_image_encode
from toast.modules.module import SkipModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def encode_data(loader, skip_encoder):
    embeddings = []
    skip_encoder.eval()

    for batch in tqdm(loader, desc="Encoding Batches with SkipModel"):
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


def _parse_skips(skips):
    import ast
    if isinstance(skips, str):
        skips = ast.literal_eval(skips)
    if not skips or not isinstance(skips[0], list):
        skips = [skips]
    return skips


@torch.no_grad()
def run_encoding(
    dataset_name: str,
    encoder_name: str,
    translator_name: str,
    seed: int,
    skips: str = "[[], [(0, 1)]]",
    samples_to_extract: int = 500,
    batch_size: int = 256,
    mode: int = 1,
):

    seed_everything(seed)
    split2encoding = {}

    skips = _parse_skips(skips)

    if encoder_name not in MODEL2CONFIGS:
        raise ValueError(f"Model configuration not found for {encoder_name}. Please add it to MODEL2CONFIGS.")

    model_config = MODEL2CONFIGS[encoder_name]

    print(f"Dataset: {dataset_name}, Encoder: {encoder_name}, Translator: {translator_name}, Skips: {skips}")

    DATASET_DIR = (
        PROJECT_ROOT
        / "data"
        / f"{translator_name}_skipped_embeddings"
        / dataset_name
        / encoder_name.split("/")[1]
        / str(samples_to_extract)
    )

    if (DATASET_DIR / "dataset_dict.json").exists():
        print(f"Loading existing dataset from {DATASET_DIR}")
        data: DatasetDict = load_from_disk(dataset_path=str(DATASET_DIR))
    else:
        print(f"Dataset directory does not exist. Creating: {DATASET_DIR}")
        DATASET_DIR.mkdir(parents=True, exist_ok=True)
        if dataset_name == "imagenet-1k":
            val_data = load_dataset(
                DATASET_NAME2HF_NAME[dataset_name],
                split="validation",
                data_files={"val": "data/val_images.tar.gz"},
                revision="refs/pr/20",
                trust_remote_code=True,
                download_config=DownloadConfig(resume_download=True),
                verification_mode=VerificationMode.NO_CHECKS,
            )

            train_test_split = val_data.train_test_split(test_size=0.2)

            data: DatasetDict = DatasetDict(
                train=train_test_split["train"],
                test=train_test_split["test"],
            )
        elif dataset_name == "svhn":
            data: DatasetDict = DatasetDict(
                train=load_dataset(DATASET_NAME2HF_NAME[dataset_name], "cropped_digits", split="train"),
                test=load_dataset(DATASET_NAME2HF_NAME[dataset_name], "cropped_digits", split="test"),
            )
        else:
            print(f"Loading dataset: {dataset_name}")
            data: DatasetDict = DatasetDict(
                train=load_dataset(DATASET_NAME2HF_NAME[dataset_name], split="train"),
                test=load_dataset(DATASET_NAME2HF_NAME[dataset_name], split="test"),
            )

    if encoder_name.startswith("open_clip:"):
        import open_clip

        print(f"Loading OpenCLIP model: {encoder_name}")
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
        print(f"Loading HF CLIP model: {encoder_name}")
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
        print(f"Loading HF AutoModel: {encoder_name}")
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
        data["train"],
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        num_workers=8,
        collate_fn=collate_fn,
    )

    test_loader = DataLoader(
        data["test"],
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        num_workers=8,
        collate_fn=collate_fn,
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

    for skip in tqdm(skips, desc="Encoding Different Skips"):
        print(f"\nProcessing skip: {skip}")

        split2encoding = {}

        skip_encoder = SkipModel(
            encoder=encoder,
            skips=skip,
            mode=mode,
            precomputed_embeddings=all_layer_embeddings,
            translator_factory_name=translator_name,
            **model_config,
        )
        skip_encoder = skip_encoder.to(device).eval()

        split2encoding["train"] = encode_data(loader=train_loader, skip_encoder=skip_encoder)
        split2encoding["test"] = encode_data(loader=test_loader, skip_encoder=skip_encoder)

        print("Saving results to disk...")
        for split, encoding in split2encoding.items():
            if not encoding:
                print(f"Warning: No embeddings generated for split '{split}', skip '{skip}'. Skipping saving.")
                continue
            column_name = str(skip)
            if column_name not in data[split].column_names:
                if len(encoding) != len(data[split]):
                    print(
                        f"Error: Encoding length ({len(encoding)}) does not match dataset length ({len(data[split])}) for split '{split}', skip '{skip}'."
                    )
                    continue
                data[split] = data[split].add_column(column_name, encoding)
            else:
                final_column_name = f"{column_name}_new"
                print(f"Column '{column_name}' already exists. Saving them with a new name: {final_column_name}")
                data[split] = data[split].add_column(final_column_name, encoding)

        del skip_encoder
        torch.cuda.empty_cache()

        if DATASET_DIR.exists():
            temp_dir = DATASET_DIR.parent / f"{DATASET_DIR.name}_temp"
            try:
                if temp_dir.exists():
                    shutil.rmtree(temp_dir)
                data.save_to_disk(str(temp_dir))
                shutil.rmtree(DATASET_DIR)
                shutil.move(str(temp_dir), DATASET_DIR)
                print(f"Saved intermediate results for skip {skip} to {DATASET_DIR}")
            except Exception as e:
                print(f"Error saving intermediate results: {e}")
                if temp_dir.exists():
                    shutil.rmtree(temp_dir)
        else:
            DATASET_DIR.mkdir(parents=True, exist_ok=True)
            data.save_to_disk(str(DATASET_DIR))
            print(f"Saved initial results for skip {skip} to {DATASET_DIR}")


if __name__ == "__main__":
    fire.Fire(run_encoding)
