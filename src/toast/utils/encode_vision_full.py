import ast
import copy
import functools
import os
import shutil

import fire
import torch
from datasets import (
    Dataset,
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
from toast.utils.utils import (
    cfg_embedding_dir,
    image_encode,
    extract_representations,
    open_clip_image_encode,
)
from toast.modules.module import SkipModel, MLPLinearisedEncoder, AttentionLinearisedEncoder, HeadPrunedEncoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

heads_to_keep = {
    0: [0, 1, 2],
    5: [0, 2, 4],
    1: [0, 1],
    2: [0, 1],
    3: [0],
    4: [0],
    6: [0],
    7: [0],
    11: [0],
}

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


def _read_config_csv(path: str):
    import csv as _csv
    configs = []
    with open(path) as f:
        for row in _csv.DictReader(f):
            configs.append({
                "dataset":          row.get("dataset") or None,
                "encoder":          row.get("encoder") or None,
                "skip":             ast.literal_eval(row["skip"]),
                "mlp_skip":         ast.literal_eval(row["mlp_skip"]),
                "attn_skip":        ast.literal_eval(row["attn_skip"]),
                "head_dict":        ast.literal_eval(row.get("head_dict", "{}")),
                "skip_translator":  row.get("skip_translator") or None,
                "mlp_mode":         row.get("mlp_mode") or "identity",
                "attn_mode":        row.get("attn_mode") or "identity",
            })
    return configs


def _parse_list_of_lists(val):
    if isinstance(val, str):
        val = ast.literal_eval(val)
    if not val or not isinstance(val[0], list):
        val = [val]
    return val


@torch.no_grad()
def run_encoding(
    translator_name: str,
    seed: int,
    dataset_name: str = None,
    encoder_name: str = None,
    config_csv: str = None,
    skips: str = "[[]]",
    mlp_skips: str = "[[]]",
    attention_skips: str = "[[]]",
    reduced_heads: bool = False,
    samples_to_extract: int = 500,
    batch_size: int = 32,
    mode: int = 1,
):
    seed_everything(seed)

    if config_csv:
        all_configs = _read_config_csv(config_csv)
    else:
        skips = _parse_list_of_lists(skips)
        mlp_skips = _parse_list_of_lists(mlp_skips)
        attention_skips = _parse_list_of_lists(attention_skips)
        all_configs = [
            {"dataset": None, "encoder": None,
             "skip": s, "mlp_skip": m, "attn_skip": a,
             "head_dict": heads_to_keep if reduced_heads else {}}
            for s in skips for a in attention_skips for m in mlp_skips
        ]

    # Resolve per-row translator/dataset/encoder and fill defaults from CLI args
    for cfg in all_configs:
        if not cfg.get("dataset"):
            cfg["dataset"] = dataset_name
        if not cfg.get("encoder"):
            cfg["encoder"] = encoder_name
        if not cfg.get("skip_translator"):
            cfg["skip_translator"] = translator_name

    # Group by (dataset, encoder) to load each model/dataset only once
    groups: dict = {}
    for cfg in all_configs:
        key = (cfg["dataset"], cfg["encoder"])
        groups.setdefault(key, []).append(cfg)

    embeddings_base = PROJECT_ROOT / "data" / "embeddings"

    for (ds_name, enc_name), config_rows in groups.items():
        print(f"\n=== dataset={ds_name} | encoder={enc_name} | {len(config_rows)} configs ===")

        if enc_name not in MODEL2CONFIGS:
            raise ValueError(f"Model config not found for {enc_name}.")
        model_config = MODEL2CONFIGS[enc_name]

        if ds_name == "svhn":
            raw_data = DatasetDict(
                train=load_dataset(DATASET_NAME2HF_NAME[ds_name], "cropped_digits", split="train"),
                test=load_dataset(DATASET_NAME2HF_NAME[ds_name], "cropped_digits", split="test"),
            )
        else:
            if ds_name not in DATASET2LOCAL_PATH:
                raise ValueError(f"No local path configured for '{ds_name}'. Add it to DATASET2LOCAL_PATH.")
            dataset_path = DATASET2LOCAL_PATH[ds_name]
            if not os.path.exists(dataset_path):
                raise ValueError(f"Dataset path does not exist: {dataset_path}")
            raw_data = load_from_disk(dataset_path)

        label_col = DATASET2LABEL_COLUMN[ds_name]

        try:
            if enc_name.startswith("open_clip:"):
                import open_clip
                open_clip_hub_name = f"hf-hub:{enc_name.split(':', 1)[1]}"
                model, _, preprocess_val = open_clip.create_model_and_transforms(open_clip_hub_name, device=device)
                encoder = model
                collate_fn = functools.partial(
                    open_clip_image_encode,
                    processor=preprocess_val,
                    image_name=DATASET2INPUT_COLUMN[ds_name],
                    label_name=DATASET2LABEL_COLUMN[ds_name],
                )
            elif enc_name == "openai/clip-vit-base-patch32":
                enc_config = CLIPVisionConfig.from_pretrained(enc_name, output_hidden_states=True, return_dict=True)
                processor = CLIPImageProcessor.from_pretrained(enc_name)
                encoder = CLIPVisionModel.from_pretrained(enc_name, config=enc_config)
                collate_fn = functools.partial(
                    image_encode,
                    processor=processor,
                    image_name=DATASET2INPUT_COLUMN[ds_name],
                    label_name=DATASET2LABEL_COLUMN[ds_name],
                )
            else:
                enc_config = AutoConfig.from_pretrained(enc_name, output_hidden_states=True, return_dict=True)
                processor = AutoImageProcessor.from_pretrained(enc_name)
                encoder = AutoModel.from_pretrained(enc_name, config=enc_config)
                collate_fn = functools.partial(
                    image_encode,
                    processor=processor,
                    image_name=DATASET2INPUT_COLUMN[ds_name],
                    label_name=DATASET2LABEL_COLUMN[ds_name],
                )

            encoder.eval().to(device)
        except Exception as e:
            print(f"✗ Error loading encoder '{enc_name}': {e}")
            continue

        try:
            train_loader = DataLoader(
                raw_data["train"], batch_size=batch_size, pin_memory=True,
                shuffle=False, num_workers=1, collate_fn=collate_fn,
            )
            test_loader = DataLoader(
                raw_data["test"], batch_size=batch_size, pin_memory=True,
                shuffle=False, num_workers=1, collate_fn=collate_fn,
            )
        except Exception as e:
            print(f"✗ Error creating data loaders: {e}")
            del encoder
            torch.cuda.empty_cache()
            continue

        try:
            all_layer_embeddings = extract_representations(
                encoder=encoder,
                max_samples=samples_to_extract,
                loader=train_loader,
                model_config=model_config,
                model_is_open_clip=enc_name.startswith("open_clip:"),
                seed=seed,
            )
            print(f"Captured embeddings for layers: {list(all_layer_embeddings.keys())}")
        except Exception as e:
            print(f"✗ Error extracting representations: {e}")
            del encoder
            torch.cuda.empty_cache()
            continue

        total = len(config_rows)
        for combo_idx, cfg in enumerate(config_rows, 1):
            try:
                skip           = cfg["skip"]
                mlp_skip       = cfg["mlp_skip"]
                attn_skip      = cfg["attn_skip"]
                head_dict      = cfg["head_dict"]
                mlp_mode       = cfg.get("mlp_mode", "identity")
                attn_mode      = cfg.get("attn_mode", "identity")
                row_translator = cfg["skip_translator"]

                cfg_dir = cfg_embedding_dir(cfg, samples_to_extract, embeddings_base)

                print(f"\n[{combo_idx}/{total}] skip={skip} | attn={attn_skip}({attn_mode}) | mlp={mlp_skip}({mlp_mode}) | heads={head_dict or 'full'} | translator={row_translator}")
                print(f"  -> {cfg_dir}")

                if (cfg_dir / "dataset_dict.json").exists():
                    print("  Already encoded, skipping.")
                    continue

                combo_encoder = copy.deepcopy(encoder)

                AttentionLinearisedEncoder(
                    combo_encoder,
                    attention_layers_to_linearize=attn_skip,
                    mode=attn_mode,
                ).to(device).eval()

                mlp_enc = MLPLinearisedEncoder(
                    combo_encoder,
                    mlp_layers_to_linearize=mlp_skip,
                    mode=mlp_mode,
                ).to(device).eval()
                if mlp_mode == "linear":
                    mlp_enc.fit(train_loader, max_samples=samples_to_extract)

                if head_dict:
                    HeadPrunedEncoder(combo_encoder, head_dict).to(device).eval()

                skip_encoder = SkipModel(
                    encoder=combo_encoder,
                    skips=skip,
                    mode=mode,
                    precomputed_embeddings=all_layer_embeddings,
                    translator_factory_name=row_translator,
                    **model_config,
                ).to(device).eval()

                split2encoding = {
                    "train": encode_data(loader=train_loader, skip_encoder=skip_encoder),
                    "test":  encode_data(loader=test_loader,  skip_encoder=skip_encoder),
                }

                valid = True
                for split, enc in split2encoding.items():
                    if len(enc) != len(raw_data[split]):
                        print(f"  Error: length mismatch for '{split}' ({len(enc)} vs {len(raw_data[split])}). Skipping.")
                        valid = False
                        break
                if not valid:
                    del skip_encoder, combo_encoder
                    torch.cuda.empty_cache()
                    continue

                new_dataset = DatasetDict({
                    split: Dataset.from_dict({
                        "embeddings": enc,
                        label_col:    raw_data[split][label_col],
                    })
                    for split, enc in split2encoding.items()
                })

                temp_dir = cfg_dir.parent / f"{cfg_dir.name}_temp"
                try:
                    if temp_dir.exists():
                        shutil.rmtree(temp_dir)
                    cfg_dir.parent.mkdir(parents=True, exist_ok=True)
                    new_dataset.save_to_disk(str(temp_dir))
                    if cfg_dir.exists():
                        shutil.rmtree(cfg_dir)
                    shutil.move(str(temp_dir), str(cfg_dir))
                    print(f"  Saved to {cfg_dir}")
                except Exception as e:
                    print(f"  Error saving: {e}")
                    if temp_dir.exists():
                        shutil.rmtree(temp_dir)

                del skip_encoder, combo_encoder
                torch.cuda.empty_cache()

            except Exception as e:
                print(f"  ✗ Error processing config {combo_idx}/{total}: {e}")
                import traceback
                traceback.print_exc()
                try:
                    del skip_encoder, combo_encoder
                except:
                    pass
                torch.cuda.empty_cache()
                continue

        del encoder
        torch.cuda.empty_cache()


if __name__ == "__main__":
    try:
        fire.Fire(run_encoding)
    except Exception as e:
        print(f"\n✗ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
