import os
from typing import List

import fire
import numpy as np
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
    MODEL2CONFIGS,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def skip_and_train_run(
    dataset_name: str,
    model_name: str,
    layers_to_approximate: List,
    seed: int,
    classifier_type: str,
    translator_name: str,
    samples_to_extract: int,
    save_checkpoint: bool = False,
):

    print(
        f"Dataset: {dataset_name}, model: {model_name}, approximating:{layers_to_approximate}, "
        f"seed: {seed}, classifier_type: {classifier_type}, translator_name: {translator_name}"
    )

    seed_everything(seed)

    if model_name.startswith("open_clip:"):
        model_name_slug = model_name.split(":")[-1].split("/")[-1]
    else:
        model_name_slug = model_name.split("/")[-1]

    EMBEDDINGS_DIR = str(
        PROJECT_ROOT
        / "data"
        / f"{translator_name}_skipped_embeddings"
        / dataset_name
        / model_name_slug
        / str(samples_to_extract)
    )

    print(f"Loading embeddings from: {EMBEDDINGS_DIR}")

    if not os.path.exists(EMBEDDINGS_DIR):
        raise FileNotFoundError(f"Embeddings not found: {EMBEDDINGS_DIR}.")
    embeddings = DatasetDict.load_from_disk(EMBEDDINGS_DIR)
    embeddings.set_format("torch")

    if model_name not in MODEL2CONFIGS:
        raise ValueError(f"Model configuration not found for '{model_name}' in MODEL2CONFIGS.")

    embedding_col_name = str(layers_to_approximate)

    if (embedding_col_name not in embeddings["train"].column_names) or (
        embedding_col_name not in embeddings["test"].column_names
    ):
        raise KeyError(f"Skip '{embedding_col_name}' not found in loaded embeddings.")

    label_col_name = DATASET2LABEL_COLUMN[dataset_name]

    hf_train_embeddings = (
        embeddings["train"]
        .select_columns([embedding_col_name, label_col_name])
        .rename_column(embedding_col_name, "images")
        .rename_column(label_col_name, "labels")
    )

    hf_test_embeddings = (
        embeddings["test"]
        .select_columns([embedding_col_name, label_col_name])
        .rename_column(embedding_col_name, "images")
        .rename_column(label_col_name, "labels")
    )

    batch_size = 256
    num_workers = 8
    num_classes = DATASET2NUM_CLASSES[dataset_name]

    train_dataloader = DataLoader(
        hf_train_embeddings, shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True
    )
    test_dataloader = DataLoader(
        hf_test_embeddings, shuffle=False, batch_size=batch_size, num_workers=num_workers, pin_memory=True
    )

    sample_embedding = embeddings["train"][0][embedding_col_name]
    hidden_size = sample_embedding.shape[-1]

    if classifier_type == "MLP":
        classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Dropout(0.5),
            nn.LayerNorm(hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, num_classes),
        )
    elif classifier_type == "linear":
        classifier = nn.Linear(hidden_size, num_classes)
    else:
        raise ValueError(f"Unsupported classifier_type: {classifier_type}")

    no_encoder = NoEncoder(embeddings=None)
    skip_model = HFwrapper(encoder=no_encoder, classifier=classifier)
    skip_model.to(device)
    skip_model.freeze_encoder()

    if classifier_type == "MLP":
        lr = 0.001
        num_epochs = 50
        optimizer = optim.Adam(skip_model.parameters(), lr=lr, weight_decay=1e-5)
    elif classifier_type == "linear":
        lr = 0.01
        num_epochs = 5
        optimizer = optim.Adam(skip_model.parameters(), lr=lr)

    print("Starting classifier training...")
    _, _, _, eval_accuracies, _ = train_classifier(
        model=skip_model,
        train_data_loader=train_dataloader,
        test_data_loader=test_dataloader,
        optimizer=optimizer,
        criterion=nn.CrossEntropyLoss(),
        label_column_name="labels",
        num_epochs=num_epochs,
    )
    accuracy = eval_accuracies[-1]
    print(f"Training finished. Final accuracy: {accuracy:.4f}")

    columns = [
        "seed",
        "dataset",
        "model",
        "optimizer",
        "lr",
        "classifier",
        "translator",
        "batch_size",
        "num_epochs",
        "approx_layer",
        "num_layers",
        "original_accuracy",
        "accuracy",
        "delta_acc",
        "num_samples",
    ]

    results_path = PROJECT_ROOT / "results" / "results.csv"

    if os.path.exists(results_path):
        try:
            results_df = pd.read_csv(results_path)
        except pd.errors.EmptyDataError:
            print(f"Results file {results_path} is empty. Initializing DataFrame.")
            results_df = pd.DataFrame(columns=columns)
        except Exception as e:
            print(f"Error reading results file {results_path}: {e}. Initializing DataFrame.")
            results_df = pd.DataFrame(columns=columns)
    else:
        results_path.parent.mkdir(parents=True, exist_ok=True)
        results_df = pd.DataFrame(columns=columns)

    results_list = []
    results = {}
    original_accuracy = 0.0
    baseline_skip_repr = str([])

    if str(layers_to_approximate) == baseline_skip_repr:
        original_accuracy = accuracy
    else:
        filtered_df = results_df[
            (results_df["approx_layer"] == "[]")
            & (results_df["dataset"] == dataset_name)
            & (results_df["model"] == model_name)
            & (results_df["classifier"] == classifier.__class__.__name__)
            & (results_df["translator"] == translator_name)
            & (results_df["num_epochs"] == num_epochs)
            & (results_df["seed"] == seed)
            & (results_df["batch_size"] == batch_size)
            & (results_df["lr"] == lr)
            & (results_df["num_samples"] == samples_to_extract)
        ]
        original_accuracy = filtered_df["accuracy"].iloc[0] if not filtered_df.empty else 0.0

    delta_acc = (
        original_accuracy - accuracy if original_accuracy is not None and original_accuracy != 0.0 else 0.0
    )

    results = {
        "seed": seed,
        "dataset": dataset_name,
        "model": model_name,
        "optimizer": optimizer.__class__.__name__,
        "lr": lr,
        "classifier": classifier.__class__.__name__,
        "translator": translator_name,
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "approx_layer": layers_to_approximate,
        "original_accuracy": original_accuracy,
        "num_layers": sum([i[1] - i[0] for i in layers_to_approximate]),
        "accuracy": accuracy,
        "delta_acc": delta_acc,
        "num_samples": samples_to_extract,
    }

    results_list.append(results)

    new_results_df = pd.DataFrame(results_list)
    results_df = pd.concat([results_df, new_results_df])
    results_df.to_csv(results_path, index=False)

    if save_checkpoint:
        if model_name.startswith("open_clip:"):
            model_name_for_path = model_name.split(":")[-1].split("/")[-1]
        else:
            model_name_for_path = model_name.split("/")[-1]

        model_dir_path = PROJECT_ROOT / "models" / model_name_for_path
        model_dir_path.mkdir(parents=True, exist_ok=True)
        checkpoint_path = model_dir_path / f"{dataset_name}_classifier.ckpt"

        torch.save(skip_model.classifier, checkpoint_path)
        print(f"Checkpoint saved to: {checkpoint_path}")


if __name__ == "__main__":
    fire.Fire(skip_and_train_run)
