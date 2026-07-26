import ast
import csv as _csv
import os

import fire
import pandas as pd
import torch
from datasets import DatasetDict
from pytorch_lightning import seed_everything
from torch import nn, optim
from torch.utils.data import DataLoader

from toast import PROJECT_ROOT
from toast.modules.module import HFwrapper, NoEncoder
from toast.pl_modules.train_NN import train_classifier
from toast.utils.dictionaries import (
    DATASET2LABEL_COLUMN,
    DATASET2NUM_CLASSES,
    params_saved_for_config,
)
from toast.utils.utils import cfg_embedding_dir

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
        "seed", "dataset", "model", "optimizer", "lr", "classifier",
        "translator", "batch_size", "num_epochs",
        "approx_layer", "mlp_linearize", "attn_linearize", "head_dict",
        "mlp_mode", "attn_mode",
        "num_layers", "original_accuracy", "accuracy", "delta_acc", "num_samples",
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
                continue

            embeddings = DatasetDict.load_from_disk(str(cfg_dir))
            embeddings.set_format("torch")

            # Set training seed after loading embeddings (which were generated with seed=0)
            seed_everything(seed)
        except Exception as e:
            print(f"  ✗ Error loading embeddings: {e}")
            continue

        try:
            row_label_col   = DATASET2LABEL_COLUMN[row_dataset]
            row_num_classes = DATASET2NUM_CLASSES[row_dataset]

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

            if hidden_size is None:
                hidden_size = embeddings["train"][0]["embeddings"].shape[-1]

            batch_size = 256
            train_loader = DataLoader(hf_train, shuffle=True,  batch_size=batch_size, num_workers=2, pin_memory=True)
            test_loader  = DataLoader(hf_test,  shuffle=False, batch_size=batch_size, num_workers=2, pin_memory=True)

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

            print("  Starting classifier training...")
            _, _, _, eval_accuracies, _ = train_classifier(
                model=model,
                train_data_loader=train_loader,
                test_data_loader=test_loader,
                optimizer=optimizer,
                criterion=nn.CrossEntropyLoss(),
                label_column_name="labels",
                num_epochs=num_epochs,
            )
            accuracy = eval_accuracies[-1]
            print(f"  Done. Accuracy: {accuracy:.4f}")
        except Exception as e:
            print(f"  ✗ Error during training: {e}")
            import traceback
            traceback.print_exc()
            continue

        try:
            is_baseline = not skip and not mlp_skip and not attn_skip and not head_dict
            if is_baseline:
                original_accuracy = accuracy
            else:
                filtered = results_df[
                    (results_df["approx_layer"]  == str([]))
                    & (results_df["mlp_linearize"] == str([]))
                    & (results_df["attn_linearize"] == str([]))
                    & (results_df["head_dict"]     == str({}))
                    & (results_df["dataset"]       == row_dataset)
                    & (results_df["model"]         == row_encoder)
                    & (results_df["classifier"]    == classifier.__class__.__name__)
                    & (results_df["translator"]    == row_translator)
                    & (results_df["seed"]          == seed)
                    & (results_df["num_samples"]   == samples_to_extract)
                ]
                original_accuracy = filtered["accuracy"].iloc[0] if not filtered.empty else 0.0

            delta_acc = original_accuracy - accuracy if original_accuracy != 0.0 else 0.0
            num_layers_skipped = sum(end - start for start, end in skip) if skip else 0

            row = {
                "seed":              seed,
                "dataset":           row_dataset,
                "model":             row_encoder,
                "optimizer":         optimizer.__class__.__name__,
                "lr":                lr,
                "classifier":        classifier.__class__.__name__,
                "translator":        row_translator,
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
                "num_samples":       samples_to_extract,
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
            continue


if __name__ == "__main__":
    try:
        fire.Fire(skip_and_train_full_run)
    except Exception as e:
        print(f"\n✗ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
