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
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def skip_and_train_mlp_run(
    dataset_name: str,
    model_name: str,
    layers_to_approximate: List,
    mlp_layers_to_linearize: List,
    seed: int,
    classifier_type: str,
    translator_name: str,
    samples_to_extract: int,
    mode: int = 1,
    save_checkpoint: bool = False,
):
    import ast

    if isinstance(layers_to_approximate, str):
        layers_to_approximate = ast.literal_eval(layers_to_approximate)
    if isinstance(mlp_layers_to_linearize, str):
        mlp_layers_to_linearize = ast.literal_eval(mlp_layers_to_linearize)

    print(
        f"Dataset: {dataset_name} | Model: {model_name} | "
        f"Skip: {layers_to_approximate} | MLP: {mlp_layers_to_linearize} | "
        f"Seed: {seed} | Classifier: {classifier_type} | Translator: {translator_name}"
    )

    seed_everything(seed)

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

    embedding_col_name = f"skip={layers_to_approximate}_mlp={mlp_layers_to_linearize}"

    if (embedding_col_name not in embeddings["train"].column_names) or (
        embedding_col_name not in embeddings["test"].column_names
    ):
        available = embeddings["train"].column_names
        raise KeyError(
            f"Column '{embedding_col_name}' not found. Available: {available}"
        )

    label_col_name = DATASET2LABEL_COLUMN[dataset_name]
    num_classes = DATASET2NUM_CLASSES[dataset_name]

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
    num_workers = 2
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
        lr = 0.001
        num_epochs = 50
        optimizer = optim.Adam(classifier.parameters(), lr=lr, weight_decay=1e-5)
    elif classifier_type == "linear":
        classifier = nn.Linear(hidden_size, num_classes)
        lr = 0.01
        num_epochs = 5
        optimizer = optim.Adam(classifier.parameters(), lr=lr)
    else:
        raise ValueError(f"Unsupported classifier_type: {classifier_type}")

    no_encoder = NoEncoder()
    skip_model = HFwrapper(encoder=no_encoder, classifier=classifier)
    skip_model.to(device)
    skip_model.freeze_encoder()

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
        "seed", "dataset", "model", "optimizer", "lr", "classifier",
        "translator", "batch_size", "num_epochs", "approx_layer",
        "mlp_linearize", "num_layers", "original_accuracy", "accuracy",
        "delta_acc", "num_samples",
    ]

    results_path = PROJECT_ROOT / "results" / "results_mlp_test.csv"
    if os.path.exists(results_path):
        try:
            results_df = pd.read_csv(results_path)
        except Exception:
            results_df = pd.DataFrame(columns=columns)
    else:
        results_path.parent.mkdir(parents=True, exist_ok=True)
        results_df = pd.DataFrame(columns=columns)

    original_accuracy = 0.0
    if layers_to_approximate or mlp_layers_to_linearize:
        filtered = results_df[
            (results_df["approx_layer"] == str([]))
            & (results_df["mlp_linearize"] == str([]))
            & (results_df["dataset"] == dataset_name)
            & (results_df["model"] == model_name)
            & (results_df["classifier"] == classifier.__class__.__name__)
            & (results_df["translator"] == translator_name)
            & (results_df["seed"] == seed)
            & (results_df["num_samples"] == samples_to_extract)
        ]
        original_accuracy = filtered["accuracy"].iloc[0] if not filtered.empty else 0.0
    else:
        original_accuracy = accuracy

    delta_acc = original_accuracy - accuracy if original_accuracy != 0.0 else 0.0
    num_layers_skipped = sum(end - start for start, end in layers_to_approximate) if layers_to_approximate else 0

    row = {
        "seed": seed,
        "dataset": dataset_name,
        "model": model_name,
        "optimizer": optimizer.__class__.__name__,
        "lr": lr,
        "classifier": classifier.__class__.__name__,
        "translator": translator_name,
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "approx_layer": str(layers_to_approximate),
        "mlp_linearize": str(mlp_layers_to_linearize),
        "num_layers": num_layers_skipped,
        "original_accuracy": original_accuracy,
        "accuracy": accuracy,
        "delta_acc": delta_acc,
        "num_samples": samples_to_extract,
    }

    results_df = pd.concat([results_df, pd.DataFrame([row])], ignore_index=True)
    results_df.to_csv(results_path, index=False)
    print(f"Results saved to: {results_path}")

    if save_checkpoint:
        model_dir_path = PROJECT_ROOT / "models" / model_name_slug
        model_dir_path.mkdir(parents=True, exist_ok=True)
        checkpoint_path = model_dir_path / f"{dataset_name}_mlp_classifier.ckpt"
        torch.save(skip_model.classifier, checkpoint_path)
        print(f"Checkpoint saved to: {checkpoint_path}")


if __name__ == "__main__":
    fire.Fire(skip_and_train_mlp_run)
