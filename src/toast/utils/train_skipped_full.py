import ast
import csv as _csv
import os
from collections import Counter

import fire
import pandas as pd
import torch
from datasets import DatasetDict
from torch import nn, optim
from torch.utils.data import DataLoader
from transformers import AutoModelForImageClassification

from toast import PROJECT_ROOT
from toast.modules.module import HFwrapper, NoEncoder
from toast.modules.lowrank_translator import parse_translator_name
from toast.pl_modules.train_NN import train_classifier
from toast.utils.dictionaries import (
    DATASET2LABEL_COLUMN,
    DATASET2NUM_CLASSES,
    MULTILABEL_DATASETS,
    params_saved_for_config,
)
from toast.utils.utils import cfg_embedding_dir, seed_everything

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# The original pretrained head is only meaningful when (a) the dataset is what it was trained
# for -- these HF checkpoints ship a 1000-class ImageNet-1k head, so scoring it against e.g.
# pathmnist labels would just be noise -- and (b) the checkpoint actually has a classification
# head at all: dinov2/rad-dino/clip are backbone-only or non-ImageNet and have none. Rather than
# hardcode which encoders qualify (easy to get wrong), each encoder is probed once and cached;
# a failed load or an output width that doesn't match the dataset's class count both resolve to
# "no usable head" (cached as None) so later rows for the same encoder don't retry.
ORIG_HEAD_DATASETS = {"imagenet-1k"}
_orig_head_cache: dict = {}


def _get_orig_head(encoder_hf_id: str, num_classes: int):
    if encoder_hf_id not in _orig_head_cache:
        try:
            head = AutoModelForImageClassification.from_pretrained(encoder_hf_id).classifier
            out_features = head.out_features if isinstance(head, nn.Linear) else None
            if out_features != num_classes:
                print(f"  WARNING: '{encoder_hf_id}' classifier outputs "
                      f"{out_features} classes, expected {num_classes}; treating as no "
                      f"usable original head.")
                head = None
        except Exception as e:
            print(f"  WARNING: could not load original head for '{encoder_hf_id}': {e}; "
                  f"treating as no usable original head.")
            head = None
        if head is not None:
            head.eval().to(device)
            for p in head.parameters():
                p.requires_grad_(False)
        _orig_head_cache[encoder_hf_id] = head
    return _orig_head_cache[encoder_hf_id]


@torch.no_grad()
def _orig_head_accuracy(encoder_hf_id: str, num_classes: int, test_loader) -> float:
    head = _get_orig_head(encoder_hf_id, num_classes)
    if head is None:
        return float("nan")
    correct, total = 0, 0
    for batch in test_loader:
        embeddings = batch["images"].to(device)
        labels = batch["labels"].to(device)
        preds = head(embeddings).argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.shape[0]
    return correct / total if total else float("nan")


def _read_config_csv(path: str):
    configs = []
    with open(path) as f:
        for row in _csv.DictReader(f):
            configs.append({
                "dataset":         row.get("dataset") or None,
                "encoder":         row.get("encoder") or None,
                "skip":            ast.literal_eval(row["skip"]),
                "mlp_skip":        ast.literal_eval(row["mlp_skip"]),
                "attn_skip":       ast.literal_eval(row["attn_skip"]),
                "head_dict":       ast.literal_eval(row.get("head_dict", "{}")),
                "skip_translator": row.get("skip_translator") or None,
                "mlp_mode":        row.get("mlp_mode") or "identity",
                "attn_mode":       row.get("attn_mode") or "identity",
                # Must be parsed here too, not just in encode_vision_full: it feeds
                # cfg_embedding_dir, so omitting it would look for the embeddings of the
                # same-dataset run instead of the transfer run's own directory.
                "fit_dataset":     row.get("fit_dataset") or None,
            })
    return configs


def skip_and_train_full_run(
    seed: int,
    classifier_type: str,
    translator_name: str,
    samples_to_extract: int,
    dataset_name: str = None,
    model_name: str = None,
    config_csv: str = None,
    layers_to_approximate=None,
    mlp_layers_to_linearize=None,
    attention_layers_to_linearize=None,
    save_checkpoint: bool = False,
):
    # Embeddings were generated with seed=0, so compute paths with seed=0
    # Training seed is set later after loading embeddings
    seed_everything(0)

    if config_csv:
        configs = _read_config_csv(config_csv)
    else:
        if isinstance(layers_to_approximate, str):
            layers_to_approximate = ast.literal_eval(layers_to_approximate)
        if isinstance(mlp_layers_to_linearize, str):
            mlp_layers_to_linearize = ast.literal_eval(mlp_layers_to_linearize)
        if isinstance(attention_layers_to_linearize, str):
            attention_layers_to_linearize = ast.literal_eval(attention_layers_to_linearize)
        configs = [{
            "skip":      layers_to_approximate,
            "mlp_skip":  mlp_layers_to_linearize,
            "attn_skip": attention_layers_to_linearize,
            "head_dict": {},
        }]

    # Resolve per-row fields against CLI defaults
    for cfg in configs:
        if not cfg.get("dataset"):
            cfg["dataset"] = dataset_name
        if not cfg.get("encoder"):
            cfg["encoder"] = model_name
        if not cfg.get("skip_translator"):
            cfg["skip_translator"] = translator_name

    results_columns = [
        # fit_dataset equals dataset for an ordinary run; it differs only for a transfer row,
        # where the translator was fitted on one dataset and evaluated on another.
        "seed", "dataset", "fit_dataset", "model", "optimizer", "lr", "classifier",
        # translator is the raw name ("rrr_32"); translator_method/translator_rank break it
        # out so a rank sweep can be grouped without string-parsing every row. Both are None
        # for identity/linear/mlp, which have no rank.
        "translator", "translator_method", "translator_rank", "batch_size", "num_epochs",
        "approx_layer", "mlp_linearize", "attn_linearize", "head_dict",
        "mlp_mode", "attn_mode",
        "num_layers", "original_accuracy", "accuracy", "delta_acc",
        # orig_head_* mirror original_accuracy/accuracy/delta_acc but score the *frozen,
        # pretrained* classifier head instead of a newly trained one -- NaN outside
        # ORIG_HEAD_DATASETS, where the pretrained head has no meaningful class mapping.
        "orig_head_accuracy", "orig_head_delta_acc", "num_samples",
        # accuracy <= majority_class_rate means the probe collapsed and the row is a null
        # result. Stored per row so it can be checked across a whole sweep after the fact.
        "majority_class_rate",
        "params_saved", "params_saved_pct",
    ]
    results_path = PROJECT_ROOT / "results" / os.environ.get("RESULTS_CSV_NAME", "results_new.csv")
    if os.path.exists(results_path):
        try:
            results_df = pd.read_csv(results_path)
            for col in results_columns:
                if col not in results_df.columns:
                    results_df[col] = None
        except Exception:
            results_df = pd.DataFrame(columns=results_columns)
    else:
        results_path.parent.mkdir(parents=True, exist_ok=True)
        results_df = pd.DataFrame(columns=results_columns)

    embeddings_base = PROJECT_ROOT / "data" / "embeddings"
    hidden_size = None

    # Rows that fail are skipped so one bad config does not kill a long sweep, but they
    # are counted here and re-raised at the end. Without this the process exits 0 even
    # when every row failed, so run_pipeline_row_by_row.sh marches on and the results CSV
    # just quietly has holes in it.
    n_failed = 0
    n_missing_embeddings = 0

    for cfg in configs:
        try:
            skip           = cfg["skip"]
            mlp_skip       = cfg["mlp_skip"]
            attn_skip      = cfg["attn_skip"]
            head_dict      = cfg["head_dict"]
            mlp_mode       = cfg.get("mlp_mode", "identity")
            attn_mode      = cfg.get("attn_mode", "identity")
            row_dataset    = cfg["dataset"]
            row_encoder    = cfg["encoder"]
            row_translator = cfg["skip_translator"]

            cfg_dir = cfg_embedding_dir(cfg, samples_to_extract, embeddings_base)

            savings = params_saved_for_config(
                row_encoder,
                skip=skip,
                mlp_skip=mlp_skip,
                attn_skip=attn_skip,
                head_dict=head_dict,
                skip_translator=row_translator,
                mlp_mode=mlp_mode,
                attn_mode=attn_mode,
            )

            print(f"\nDataset: {row_dataset} | Model: {row_encoder} | translator: {row_translator}")
            print(f"  skip={skip} | mlp={mlp_skip}({mlp_mode}) | attn={attn_skip}({attn_mode}) | heads={head_dict or 'full'}")
            if savings["params_saved"] is None:
                print(f"  Params saved: n/a (no PARAM_SAVINGS entry for '{row_encoder}')")
            else:
                print(f"  Params saved: {savings['params_saved']:,} "
                      f"({savings['params_saved_pct']:.2f}% of encoder body)")
            print(f"  -> {cfg_dir}")

            if not cfg_dir.exists():
                print(f"  WARNING: embeddings not found at '{cfg_dir}'. Run encode first. Skipping.")
                n_missing_embeddings += 1
                continue

            embeddings = DatasetDict.load_from_disk(str(cfg_dir))
            embeddings.set_format("torch")

            # Set training seed after loading embeddings (which were generated with seed=0)
            seed_everything(seed)
        except Exception as e:
            print(f"  ✗ Error loading embeddings: {e}")
            import traceback
            traceback.print_exc()
            n_failed += 1
            continue

        try:
            row_label_col   = DATASET2LABEL_COLUMN[row_dataset]
            row_num_classes = DATASET2NUM_CLASSES[row_dataset]
            is_multilabel = row_dataset in MULTILABEL_DATASETS

            hf_train = (
                embeddings["train"]
                .select_columns(["embeddings", row_label_col])
                .rename_column("embeddings", "images")
                .rename_column(row_label_col, "labels")
            )
            hf_test = (
                embeddings["test"]
                .select_columns(["embeddings", row_label_col])
                .rename_column("embeddings", "images")
                .rename_column(row_label_col, "labels")
            )

            # cls_embeddings (raw CLS token) is what the pretrained head was actually trained
            # on -- "embeddings" is SkipModel's pooled output, a different space (see
            # encode_data's docstring). Directories saved before this column existed won't
            # have it; orig_head_accuracy just goes to NaN for those rather than KeyError'ing.
            has_cls_embeddings = "cls_embeddings" in embeddings["test"].column_names
            if has_cls_embeddings:
                hf_test_cls = (
                    embeddings["test"]
                    .select_columns(["cls_embeddings", row_label_col])
                    .rename_column("cls_embeddings", "images")
                    .rename_column(row_label_col, "labels")
                )

            if hidden_size is None:
                hidden_size = embeddings["train"][0]["embeddings"].shape[-1]

            test_labels = hf_test["labels"]
            if torch.is_tensor(test_labels):
                test_labels = test_labels.tolist()
            if test_labels and not isinstance(test_labels[0], (list, tuple)):
                counts = Counter(int(v) for v in test_labels)
                majority_rate = max(counts.values()) / len(test_labels)
            else:
                majority_rate = None  # multi-label: no single majority class

            batch_size = 256
            train_loader = DataLoader(hf_train, shuffle=True,  batch_size=batch_size, num_workers=2, pin_memory=True)
            test_loader  = DataLoader(hf_test,  shuffle=False, batch_size=batch_size, num_workers=2, pin_memory=True)
            test_loader_cls = (
                DataLoader(hf_test_cls, shuffle=False, batch_size=batch_size, num_workers=2, pin_memory=True)
                if has_cls_embeddings else None
            )

            if classifier_type == "MLP":
                classifier = nn.Sequential(
                    nn.Linear(hidden_size, hidden_size),
                    nn.Dropout(0.5),
                    nn.LayerNorm(hidden_size),
                    nn.SiLU(),
                    nn.Linear(hidden_size, row_num_classes),
                )
                lr, num_epochs = 0.001, 50
                optimizer = optim.Adam(classifier.parameters(), lr=lr, weight_decay=1e-5)
            elif classifier_type == "linear":
                classifier = nn.Linear(hidden_size, row_num_classes)
                lr, num_epochs = 0.01, 5
                optimizer = optim.Adam(classifier.parameters(), lr=lr)
            else:
                raise ValueError(f"Unsupported classifier_type: {classifier_type}")

            model = HFwrapper(encoder=NoEncoder(), classifier=classifier)
            model.to(device)
            model.freeze_encoder()

            criterion = nn.BCEWithLogitsLoss() if is_multilabel else nn.CrossEntropyLoss()

            print("  Starting classifier training...")
            _, _, _, eval_accuracies, _ = train_classifier(
                model=model,
                train_data_loader=train_loader,
                test_data_loader=test_loader,
                optimizer=optimizer,
                criterion=criterion,
                label_column_name="labels",
                num_epochs=num_epochs,
                is_multilabel=is_multilabel,
            )
            accuracy = eval_accuracies[-1]
            metric_name = "macro-AUC" if is_multilabel else "Accuracy"
            print(f"  Done. {metric_name}: {accuracy:.4f}")

            # Frozen-head accuracy: no training, just the pretrained classifier scored
            # directly on this row's (possibly skip-approximated) test embeddings. Must use
            # the raw CLS token (test_loader_cls), not the pooled "embeddings" test_loader
            # above -- the pretrained head was trained on the CLS token, not on SkipModel's
            # .pooler output. See encode_data's docstring for why these differ.
            if row_dataset in ORIG_HEAD_DATASETS and test_loader_cls is not None:
                orig_head_accuracy = _orig_head_accuracy(row_encoder, row_num_classes, test_loader_cls)
                print(f"  Orig head accuracy: {orig_head_accuracy:.4f}")
            elif row_dataset in ORIG_HEAD_DATASETS:
                print("  WARNING: no cls_embeddings in this row's embedding dir (encoded "
                      "before that column existed); orig_head_accuracy will be NaN. "
                      "Re-run phase 1 for this config to get it.")
                orig_head_accuracy = float("nan")
            else:
                orig_head_accuracy = float("nan")
        except Exception as e:
            print(f"  ✗ Error during training: {e}")
            import traceback
            traceback.print_exc()
            n_failed += 1
            continue

        try:
            is_baseline = not skip and not mlp_skip and not attn_skip and not head_dict
            if is_baseline:
                original_accuracy = accuracy
                orig_head_original_accuracy = orig_head_accuracy
            else:
                # The baseline is the unmodified encoder: no skips, no sublayer edits, no head
                # pruning. With nothing to bridge, the translator is never invoked, so the
                # baseline accuracy cannot depend on translator / mlp_mode / attn_mode /
                # fit_dataset and must NOT be matched on them.
                #
                # This used to match on translator, which meant a "linear" row only accepted a
                # "linear" baseline. Since baseline rows are conventionally written with
                # translator=identity, every non-identity row failed the lookup and silently
                # reported delta_acc = 0.0 -- all 125 linear rows in results_pipeline.csv are
                # affected. A rank sweep, where each row carries a different translator name,
                # would have lost its baseline on every single row.
                filtered = results_df[
                    (results_df["approx_layer"]  == str([]))
                    & (results_df["mlp_linearize"] == str([]))
                    & (results_df["attn_linearize"] == str([]))
                    & (results_df["head_dict"]     == str({}))
                    & (results_df["dataset"]       == row_dataset)
                    & (results_df["model"]         == row_encoder)
                    & (results_df["classifier"]    == classifier.__class__.__name__)
                    & (results_df["seed"]          == seed)
                    & (results_df["num_samples"]   == samples_to_extract)
                ]
                if filtered.empty:
                    # Distinguish "no baseline to compare against" from "identical to
                    # baseline". Both used to come out as delta_acc = 0.0.
                    print(f"  WARNING: no baseline row for dataset={row_dataset} "
                          f"model={row_encoder} seed={seed}; delta_acc will be NaN. "
                          f"Put the all-empty baseline row first in the config CSV.")
                    original_accuracy = float("nan")
                    orig_head_original_accuracy = float("nan")
                else:
                    original_accuracy = filtered["accuracy"].iloc[0]
                    # A results CSV written before this column existed backfills it with
                    # None (not NaN, see the load block above), which breaks the subtraction
                    # below with a TypeError rather than propagating as an unknown value.
                    raw = filtered["orig_head_accuracy"].iloc[0]
                    orig_head_original_accuracy = float("nan") if raw is None else raw

            # NaN propagates when there was no baseline, which is what we want: the drop is
            # genuinely unknown. The old guard collapsed a missing baseline to 0.0, making it
            # indistinguishable from a config that cost no accuracy at all.
            delta_acc = original_accuracy - accuracy
            # NaN whenever either side is NaN -- no baseline, or this encoder/dataset has no
            # usable original head -- which is what we want rather than a fabricated 0.0.
            orig_head_delta_acc = orig_head_original_accuracy - orig_head_accuracy
            num_layers_skipped = sum(end - start for start, end in skip) if skip else 0

            translator_method, translator_rank = parse_translator_name(row_translator)

            row = {
                "seed":              seed,
                "dataset":           row_dataset,
                "fit_dataset":       cfg.get("fit_dataset") or row_dataset,
                "model":             row_encoder,
                "optimizer":         optimizer.__class__.__name__,
                "lr":                lr,
                "classifier":        classifier.__class__.__name__,
                "translator":        row_translator,
                "translator_method": translator_method,
                "translator_rank":   translator_rank,
                "batch_size":        batch_size,
                "num_epochs":        num_epochs,
                "approx_layer":      str(skip),
                "mlp_linearize":     str(mlp_skip),
                "attn_linearize":    str(attn_skip),
                "head_dict":         str(head_dict),
                "mlp_mode":          mlp_mode,
                "attn_mode":         attn_mode,
                "num_layers":        num_layers_skipped,
                "original_accuracy": original_accuracy,
                "accuracy":          accuracy,
                "delta_acc":         delta_acc,
                "orig_head_accuracy": orig_head_accuracy,
                "orig_head_delta_acc": orig_head_delta_acc,
                "num_samples":       samples_to_extract,
                "majority_class_rate": majority_rate,
                "params_saved":      savings["params_saved"],
                "params_saved_pct":  savings["params_saved_pct"],
            }

            try:
                results_df = pd.concat([results_df, pd.DataFrame([row])], ignore_index=True)
                results_df.to_csv(results_path, index=False)
                print(f"  Results saved to: {results_path}")
            except Exception as e:
                print(f"  ✗ Error saving results: {e}")

            if save_checkpoint:
                try:
                    model_dir = PROJECT_ROOT / "models" / row_encoder.split("/")[-1]
                    model_dir.mkdir(parents=True, exist_ok=True)
                    torch.save(model.classifier, model_dir / f"{row_dataset}_full_classifier.ckpt")
                    print(f"  Checkpoint saved to: {model_dir / f'{row_dataset}_full_classifier.ckpt'}")
                except Exception as e:
                    print(f"  ✗ Error saving checkpoint: {e}")

        except Exception as e:
            print(f"  ✗ Error processing results: {e}")
            import traceback
            traceback.print_exc()
            n_failed += 1
            continue

    if n_failed or n_missing_embeddings:
        raise RuntimeError(
            f"{n_failed}/{len(configs)} config rows failed and "
            f"{n_missing_embeddings}/{len(configs)} had no embeddings on disk. "
            f"Results in {results_path} are incomplete -- see the tracebacks above."
        )


if __name__ == "__main__":
    try:
        fire.Fire(skip_and_train_full_run)
    except Exception as e:
        print(f"\n✗ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
