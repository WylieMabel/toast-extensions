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
from toast.utils.utils import seed_everything
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
from toast.utils.translator_keys import (
    read_sweep_rows,
    row_saves_translator,
    transfer_requirements,
    translator_store_key,
)
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
    """Returns (embeddings, cls_embeddings, labels), all in the order the loader yielded them.

    Labels MUST come from the batch itself, not from re-reading the source dataset's label
    column afterward: with shuffle=True the loader's iteration order no longer matches the
    dataset's on-disk order, so re-fetching labels separately silently pairs each embedding
    with the wrong image's label. (This is exactly what happened when shuffle was briefly
    enabled here -- accuracy collapsed to worse than chance because every embedding got a
    randomly mismatched label instead of an error.)

    `embeddings` is SkipModel's normal pooled output (through .pooler when the model config
    has one) -- unchanged from before, so every existing/retrained-classifier sweep stays
    comparable. `cls_embeddings` is the raw CLS token (sequence_output[:, 0, :]), computed from
    the same forward pass rather than a second one. This split exists because HF's own
    ImageClassification heads (ViTForImageClassification etc.) are trained on the raw CLS
    token, NOT on .pooler's output -- feeding a frozen pretrained head SkipModel's pooled
    output collapses accuracy to chance even at zero skips, since .pooler (Linear+Tanh) is a
    representation the head was never trained on. cls_embeddings exists so orig_head_accuracy
    can score the pretrained head in the space it actually expects, without touching what
    every other sweep in this repo has always trained/evaluated on.
    """
    embeddings = []
    cls_embeddings = []
    labels = []
    skip_encoder.eval()
    for batch in tqdm(loader, desc="Encoding Batches"):
        image_input = batch.get("pixel_values", batch.get("images"))
        if image_input is None:
            raise KeyError("Batch missing required key 'pixel_values' or 'images'")
        image_input = image_input.to(device)
        attn_mask = batch.get("attention_mask", None)
        if attn_mask is not None:
            attn_mask = attn_mask.to(device)

        sequence_output = skip_encoder(image_input, attention_mask=attn_mask, return_sequence=True)
        cls = sequence_output[:, 0, :]
        pooled = skip_encoder.pooler_module(sequence_output) if skip_encoder.pooler_module else cls

        embeddings.extend(pooled.cpu().tolist())
        cls_embeddings.extend(cls.cpu().tolist())
        labels.extend(batch["labels"].tolist())
    return embeddings, cls_embeddings, labels


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
                # Optional 10th column. Blank (or equal to dataset) means the ordinary
                # fit-here-evaluate-here run; naming a different dataset makes this row load
                # the translator fitted by that dataset's row instead of fitting its own.
                "fit_dataset":      row.get("fit_dataset") or None,
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
    batch_size: int = 8,
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
        # A row that names no fit_dataset fits its translator on its own data, which is the
        # existing behaviour for every config written before this column existed.
        if not cfg.get("fit_dataset"):
            cfg["fit_dataset"] = cfg["dataset"]

    # Group by (dataset, encoder) to load each model/dataset only once
    groups: dict = {}
    for cfg in all_configs:
        key = (cfg["dataset"], cfg["encoder"])
        groups.setdefault(key, []).append(cfg)

    embeddings_base = PROJECT_ROOT / "data" / "embeddings"
    # Fitted translators persist here so a transfer row in the same sweep can reuse one across
    # datasets. Unlike embeddings (which run_pipeline_row_by_row.sh deletes per row to bound
    # disk use), these outlive the row that produced them -- until the sweep ends and the
    # runner deletes the ones it created.
    translators_base = PROJECT_ROOT / "data" / "translators"

    # Only save translators some transfer row will actually load: one identical to the fitting
    # row in every respect except its dataset. Saving on every non-identity row wrote a map per
    # span for sweeps that never read one back; combined with the storage bug in
    # save_translator that came to 15GB and exhausted the home quota mid-sweep.
    #
    # The sweep CSV, not config_csv: run_pipeline_row_by_row.sh invokes this once per row with
    # a single-row temp CSV, so the process encoding a fitting row cannot otherwise see that a
    # later row transfers from it. SWEEP_CSV names the original. Falling back to config_csv
    # keeps the whole-CSV invocation (encode_vision_full.sh on its own) working unchanged.
    sweep_csv = os.environ.get("SWEEP_CSV") or config_csv
    transfer_reqs = set()
    if sweep_csv:
        try:
            transfer_reqs = transfer_requirements(read_sweep_rows(sweep_csv))
        except Exception as e:
            # Never fatal: a bad SWEEP_CSV should not sink an encoding run. Saving nothing is
            # the safe direction -- an absent translator fails loudly in load_translator,
            # whereas a stale one would silently bridge the wrong span.
            print(f"  Warning: could not read sweep CSV '{sweep_csv}' ({e}); saving no translators.")
    if transfer_reqs:
        print(f"Saving translators for {len(transfer_reqs)} config(s) needed by transfer rows.")

    # Errors below are caught per group/config so one bad row does not kill a long sweep, but
    # they are counted and re-raised at the end. Without this the process exits 0 having
    # written nothing, run_pipeline_row_by_row.sh treats phase 1 as successful and deletes the
    # log holding the actual error, and the failure only surfaces later as "embeddings not
    # found" during training -- with the cause already thrown away.
    n_failed = 0

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
            import traceback
            traceback.print_exc()
            n_failed += len(config_rows)
            continue

        try:
            # shuffle=True so the train split's iteration order (and hence which images a
            # given translator/classifier sees first) varies with `seed` (RandomSampler
            # draws from the global RNG seed_everything(seed) already set above). This
            # previously caused a real bug -- embeddings came out in shuffled order while
            # labels were re-read separately in the dataset's original order, silently
            # mispairing every embedding with the wrong label (vit-large/imagenet-1k baseline
            # collapsed from 0.7864 to 0.0010 accuracy). Fixed: encode_data() now returns
            # labels pulled from the same batches as the embeddings, so both are always in
            # the loader's actual iteration order regardless of shuffle. Test stays
            # shuffle=False -- a fixed eval set across seeds keeps seed-to-seed accuracy
            # deltas attributable to training noise, not to different seeds getting an
            # easier or harder sample of test images.
            train_loader = DataLoader(
                raw_data["train"], batch_size=batch_size, pin_memory=True,
                shuffle=True, num_workers=1, collate_fn=collate_fn,
            )
            test_loader = DataLoader(
                raw_data["test"], batch_size=batch_size, pin_memory=True,
                shuffle=False, num_workers=1, collate_fn=collate_fn,
            )
        except Exception as e:
            print(f"✗ Error creating data loaders: {e}")
            import traceback
            traceback.print_exc()
            n_failed += len(config_rows)
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
            import traceback
            traceback.print_exc()
            n_failed += len(config_rows)
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
                row_fit_dataset = cfg.get("fit_dataset") or ds_name

                # Transfer rows load the map fitted by their fit_dataset's row; a fitting row
                # saves only when this sweep holds a transfer row with an identical config.
                # identity has no state worth persisting, so it opts out entirely.
                is_transfer = row_fit_dataset != ds_name
                translator_key = translator_store_key(
                    row_fit_dataset, enc_name, row_translator, samples_to_extract
                )
                save_path = load_path = None
                if row_translator != "identity":
                    if is_transfer:
                        load_path = translators_base
                    elif row_saves_translator(cfg, transfer_reqs):
                        save_path = translators_base

                cfg_dir = cfg_embedding_dir(cfg, samples_to_extract, embeddings_base)

                print(f"\n[{combo_idx}/{total}] skip={skip} | attn={attn_skip}({attn_mode}) | mlp={mlp_skip}({mlp_mode}) | heads={head_dict or 'full'} | translator={row_translator}")
                if is_transfer:
                    print(f"  translator fitted on '{row_fit_dataset}', evaluating on '{ds_name}' (key: {translator_key})")
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
                    precomputed_translator_path=load_path,
                    to_save_translator_path=save_path,
                    translator_key=translator_key if (load_path or save_path) else None,
                    **model_config,
                ).to(device).eval()

                split2encoding = {
                    "train": encode_data(loader=train_loader, skip_encoder=skip_encoder),
                    "test":  encode_data(loader=test_loader,  skip_encoder=skip_encoder),
                }

                valid = True
                for split, (enc, cls_enc, lbls) in split2encoding.items():
                    if len(enc) != len(raw_data[split]) or len(cls_enc) != len(raw_data[split]) or len(lbls) != len(raw_data[split]):
                        print(f"  Error: length mismatch for '{split}' ({len(enc)} embeddings, "
                              f"{len(cls_enc)} cls_embeddings, {len(lbls)} labels vs "
                              f"{len(raw_data[split])} source rows). Skipping.")
                        valid = False
                        break
                if not valid:
                    del skip_encoder, combo_encoder
                    torch.cuda.empty_cache()
                    continue

                # Labels come from encode_data's own per-batch labels, NOT raw_data[split][label_col]
                # -- the latter is in the dataset's original order, which only matches the loader's
                # iteration order when shuffle=False. See encode_data's docstring.
                new_dataset = DatasetDict({
                    split: Dataset.from_dict({
                        "embeddings":     enc,
                        "cls_embeddings": cls_enc,
                        label_col:        lbls,
                    })
                    for split, (enc, cls_enc, lbls) in split2encoding.items()
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
                n_failed += 1
                try:
                    del skip_encoder, combo_encoder
                except:
                    pass
                torch.cuda.empty_cache()
                continue

        del encoder
        torch.cuda.empty_cache()

    if n_failed:
        raise RuntimeError(
            f"{n_failed}/{len(all_configs)} configs failed to encode -- see the tracebacks "
            f"above. Nothing was written for them, so training would report their embeddings "
            f"as missing rather than showing this cause."
        )


if __name__ == "__main__":
    try:
        fire.Fire(run_encoding)
    except Exception as e:
        print(f"\n✗ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
