import copy
import functools

import pandas as pd
import torch
import torch.nn as nn
import wandb
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

import fire
from datasets import DownloadConfig, VerificationMode, load_dataset
from pytorch_lightning import seed_everything
from transformers import (
    AutoConfig,
    AutoImageProcessor,
    AutoModel,
    AutoModelForImageClassification,
)

from toast import PROJECT_ROOT
from toast.modules.module import HFwrapper, SkipModel
from toast.utils.dictionaries import (
    DATASET2INPUT_COLUMN,
    DATASET2LABEL_COLUMN,
    DATASET2NUM_CLASSES,
    DATASET_NAME2HF_NAME,
    MODEL2CONFIGS,
)
from toast.utils.timm_wrapper import is_timm_model, load_timm_encoder, load_timm_processor, get_timm_classifier_head
from toast.utils.utils import extract_representations, image_encode
from typing import List

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_CLF_ENCODER_ATTR = {
    "DeiTForImageClassification": "deit",
    "ViTForImageClassification": "vit",
    "DeiTForImageClassificationWithTeacher": "deit",
}

def train_one_epoch(model, loader, optimizer, criterion, grad_clip: float = 1.0):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for batch in tqdm(loader, desc="  Train", leave=False):
        pixel_values = batch["images"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model({"images": pixel_values})
        loss = criterion(logits, labels)
        loss.backward()

        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        total_loss += loss.item()
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.size(0)

    return total_loss / max(len(loader), 1), correct / max(total, 1)


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for batch in tqdm(loader, desc="  Eval ", leave=False):
        pixel_values = batch["images"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        logits = model({"images": pixel_values})
        loss = criterion(logits, labels)

        total_loss += loss.item()
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.size(0)

    return total_loss / max(len(loader), 1), correct / max(total, 1)



def finetune_e2e_run(
    dataset_name: str,
    model_name: str,
    layers_to_approximate: List,
    seed: int,
    # Head strategy
    use_pretrained_head: bool = True,
    classifier_type: str = "linear",   # used only when use_pretrained_head=False
    # TOAST translator
    translator_name: str = "linear",
    samples_to_extract: int = 500,
    mode: int = 1,
    # Optimisation
    lr: float = 1e-4,
    encoder_lr: float = None,  # If set, use discriminative LR (encoder_lr for encoder, lr for head)
    num_epochs: int = 10,
    batch_size: int = 64,
    weight_decay: float = 1e-4,
    grad_clip: float = 1.0,
    # Output
    results_file: str = "finetune_e2e.csv",
    # Logging
    use_wandb: bool = True,
    wandb_project: str = "toast",
):
    head_tag = "pretrained" if use_pretrained_head else classifier_type
    print(
        f"\n[E2E] Dataset={dataset_name} | Model={model_name} | "
        f"Skip={layers_to_approximate} | Translator={translator_name} | "
        f"Head={head_tag} | Seed={seed}"
    )

    seed_everything(seed)

    model_name_slug = model_name.split("/")[-1]
    run_name = f"{dataset_name}|{model_name_slug}|skip={layers_to_approximate}|{translator_name}|{head_tag}|s{seed}"
    wandb_run = None
    if use_wandb:
        wandb_run = wandb.init(
            project=wandb_project,
            name=run_name,
            config={
                "dataset": dataset_name,
                "model": model_name,
                "skip": str(layers_to_approximate),
                "translator": translator_name,
                "head_init": head_tag,
                "seed": seed,
                "lr": lr,
                "encoder_lr": encoder_lr if encoder_lr is not None else lr,
                "num_epochs": num_epochs,
                "batch_size": batch_size,
                "weight_decay": weight_decay,
                "grad_clip": grad_clip,
                "samples_to_extract": samples_to_extract,
                "mode": mode,
            },
            reinit=True,
        )

    if model_name not in MODEL2CONFIGS:
        raise ValueError(f"Model config not found for '{model_name}' in MODEL2CONFIGS.")
    model_config = MODEL2CONFIGS[model_name]

    needs_finetuning = not use_pretrained_head
    needs_translator_fitting = bool(layers_to_approximate)
    needs_train_data = needs_finetuning or needs_translator_fitting

    print("Loading dataset...")
    if dataset_name == "imagenet-1k":
        test_data = load_dataset(
            DATASET_NAME2HF_NAME[dataset_name],
            split="validation",
            data_files={"val": "data/val_images.tar.gz"},
            revision="refs/pr/20",
            trust_remote_code=True,
            download_config=DownloadConfig(resume_download=True),
            verification_mode=VerificationMode.NO_CHECKS,
        )
        print(f"  Test: {len(test_data)} (full ImageNet validation)")

        if needs_train_data:
            train_data = load_dataset(
                DATASET_NAME2HF_NAME[dataset_name],
                split="train",
                streaming=True,
                download_config=DownloadConfig(resume_download=True),
                verification_mode=VerificationMode.NO_CHECKS,
            )
        else:
            train_data = None
    else:
        test_data = load_dataset(DATASET_NAME2HF_NAME[dataset_name], split="test")
        train_data = load_dataset(DATASET_NAME2HF_NAME[dataset_name], split="train")
        print(f"  Train: {len(train_data)} | Test: {len(test_data)}")

    image_col = DATASET2INPUT_COLUMN[dataset_name]
    label_col = DATASET2LABEL_COLUMN[dataset_name]
    num_classes = DATASET2NUM_CLASSES[dataset_name]

    print("Loading model...")
    if is_timm_model(model_name):
        processor = load_timm_processor(model_name)
    else:
        processor = AutoImageProcessor.from_pretrained(model_name)

    collate_fn = functools.partial(
        image_encode, processor=processor, image_name=image_col, label_name=label_col
    )

    if use_pretrained_head:
        if is_timm_model(model_name):
            encoder = load_timm_encoder(model_name)
            classifier = get_timm_classifier_head(model_name)
            skip_model_config = {**model_config, "pooler_path": None}
        else:
            clf_model = AutoModelForImageClassification.from_pretrained(model_name)
            clf_class = clf_model.__class__.__name__

            if clf_class not in _CLF_ENCODER_ATTR:
                raise ValueError(
                    f"Don't know how to extract encoder from {clf_class}. "
                    f"Add it to _CLF_ENCODER_ATTR."
                )

            encoder = getattr(clf_model, _CLF_ENCODER_ATTR[clf_class])
            classifier = copy.deepcopy(clf_model.classifier)
            skip_model_config = {**model_config, "pooler_path": None}

        print(
            f"  Pretrained head loaded: {classifier} "
            f"(trained on {num_classes} classes)"
        )

    else:
        if is_timm_model(model_name):
            encoder = load_timm_encoder(model_name)
            hidden_size = encoder.config.hidden_size
        else:
            config = AutoConfig.from_pretrained(
                model_name, output_hidden_states=True, return_dict=True
            )
            encoder = AutoModel.from_pretrained(model_name, config=config)
            hidden_size = encoder.config.hidden_size

        if classifier_type == "linear":
            classifier = nn.Linear(hidden_size, num_classes)
        elif classifier_type == "MLP":
            classifier = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_size, num_classes),
            )
        else:
            raise ValueError(f"Unknown classifier_type: {classifier_type}")

        skip_model_config = model_config

    encoder.eval().to(device)

    test_loader = DataLoader(
        test_data, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True, collate_fn=collate_fn,
    )

    if needs_translator_fitting:
        fit_loader = DataLoader(
            train_data, batch_size=batch_size, shuffle=False,
            num_workers=4, pin_memory=True, collate_fn=collate_fn,
        )

    if needs_finetuning:
        train_loader = DataLoader(
            train_data, batch_size=batch_size, shuffle=True,
            num_workers=4, pin_memory=True, collate_fn=collate_fn,
        )

    if needs_translator_fitting:
        print(f"Extracting {samples_to_extract} samples for translator fitting...")
        with torch.no_grad():
            all_layer_embeddings = extract_representations(
                encoder=encoder,
                max_samples=samples_to_extract,
                loader=fit_loader,
                model_config=model_config,
                model_is_open_clip=False,
                use_hooks=is_timm_model(model_name),
                seed=seed,
            )
        print(f"  Captured layers: {sorted(all_layer_embeddings.keys())}")
    else:
        all_layer_embeddings = {}

    print(f"Building SkipModel with skip={layers_to_approximate}...")
    skip_model = SkipModel(
        encoder=encoder,
        skips=layers_to_approximate,
        mode=mode,
        precomputed_embeddings=all_layer_embeddings,
        translator_factory_name=translator_name,
        **skip_model_config,
    )
    skip_model.to(device)

    full_model = HFwrapper(encoder=skip_model, classifier=classifier)
    full_model.to(device)

    if use_pretrained_head:
        full_model.eval()

        criterion = nn.CrossEntropyLoss()
        test_loss, accuracy = evaluate(full_model, test_loader, criterion)
        best_test_acc = accuracy
        print(f"\n[Inference] Test loss={test_loss:.4f} | acc={accuracy:.4f}")

        if wandb_run is not None:
            wandb_run.log({"test/loss": test_loss, "test/acc": accuracy})
            wandb_run.summary["final/acc"] = accuracy
            wandb_run.summary["best/acc"] = accuracy

    else:
        if layers_to_approximate:
            skip_indices = set()
            for start, end in layers_to_approximate:
                skip_indices.update(range(start + 1, end + 1))
            for i, layer in enumerate(skip_model.encoder_layers_list):
                if i in skip_indices:
                    for param in layer.parameters():
                        param.requires_grad = False
            n_frozen = sum(
                p.numel()
                for i, layer in enumerate(skip_model.encoder_layers_list)
                if i in skip_indices
                for p in layer.parameters()
            )
            print(f"  Froze {n_frozen:,} params in skipped layers {skip_indices}")

        trainable = sum(p.numel() for p in full_model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in full_model.parameters())
        print(f"  Trainable: {trainable:,} / {total_params:,} params")

        if encoder_lr is not None:
            param_groups = [
                {'params': full_model.encoder.parameters(), 'lr': encoder_lr},
                {'params': full_model.classifier.parameters(), 'lr': lr},
            ]
            print(f"  Using discriminative LRs: encoder={encoder_lr:.2e}, head={lr:.2e}")
        else:
            param_groups = filter(lambda p: p.requires_grad, full_model.parameters())

        optimizer = optim.AdamW(param_groups, lr=lr, weight_decay=weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=(encoder_lr if encoder_lr is not None else lr) * 0.01)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

        print(f"\nStarting E2E fine-tuning ({num_epochs} epochs)...")
        best_test_acc = 0.0
        eval_accuracies = []

        for epoch in range(num_epochs):
            train_loss, train_acc = train_one_epoch(
                full_model, train_loader, optimizer, criterion, grad_clip=grad_clip
            )
            test_loss, test_acc = evaluate(full_model, test_loader, criterion)
            current_lr = scheduler.get_last_lr()[0]
            scheduler.step()

            eval_accuracies.append(test_acc)
            best_test_acc = max(best_test_acc, test_acc)

            print(
                f"  Epoch {epoch+1:02d}/{num_epochs} | "
                f"Train loss={train_loss:.4f} acc={train_acc:.4f} | "
                f"Test  loss={test_loss:.4f} acc={test_acc:.4f} | "
                f"LR={current_lr:.2e}"
            )

            if wandb_run is not None:
                wandb_run.log({
                    "epoch": epoch + 1,
                    "train/loss": train_loss,
                    "train/acc": train_acc,
                    "test/loss": test_loss,
                    "test/acc": test_acc,
                    "lr": current_lr,
                })

        accuracy = eval_accuracies[-1]
        best_test_acc = max(best_test_acc, accuracy)
        print(f"\nDone. Final acc={accuracy:.4f} | Best acc={best_test_acc:.4f}")

        if wandb_run is not None:
            wandb_run.summary["final/acc"] = accuracy
            wandb_run.summary["best/acc"] = best_test_acc

    results_path = PROJECT_ROOT / "results" / results_file
    columns = [
        "seed", "dataset", "model", "optimizer", "lr", "weight_decay",
        "head_init", "classifier", "translator", "batch_size", "num_epochs",
        "approx_layer", "num_layers", "original_accuracy", "accuracy",
        "best_accuracy", "delta_acc", "num_samples", "mode",
    ]

    if results_path.exists():
        try:
            results_df = pd.read_csv(results_path)
        except Exception:
            results_df = pd.DataFrame(columns=columns)
    else:
        results_path.parent.mkdir(parents=True, exist_ok=True)
        results_df = pd.DataFrame(columns=columns)

    approx_layer_str = str(layers_to_approximate)
    original_accuracy = 0.0
    if layers_to_approximate:
        filtered = results_df[
            (results_df["approx_layer"] == "[]")
            & (results_df["dataset"] == dataset_name)
            & (results_df["model"] == model_name)
            & (results_df["head_init"] == head_tag)
            & (results_df["translator"] == translator_name)
            & (results_df["seed"] == seed)
            & (results_df["num_samples"] == samples_to_extract)
        ]
        original_accuracy = filtered["accuracy"].iloc[0] if not filtered.empty else 0.0
    else:
        original_accuracy = accuracy

    delta_acc = original_accuracy - accuracy if original_accuracy != 0.0 else 0.0
    num_layers_skipped = (
        sum(end - start for start, end in layers_to_approximate) if layers_to_approximate else 0
    )

    row = {
        "seed": seed,
        "dataset": dataset_name,
        "model": model_name,
        "optimizer": "AdamW",
        "lr": lr,
        "weight_decay": weight_decay,
        "head_init": head_tag,
        "classifier": classifier.__class__.__name__,
        "translator": translator_name,
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "approx_layer": approx_layer_str,
        "num_layers": num_layers_skipped,
        "original_accuracy": original_accuracy,
        "accuracy": accuracy,
        "best_accuracy": best_test_acc,
        "delta_acc": delta_acc,
        "num_samples": samples_to_extract,
        "mode": mode,
    }

    results_df = pd.concat([results_df, pd.DataFrame([row])], ignore_index=True)
    results_df.to_csv(results_path, index=False)
    print(f"Results saved to: {results_path}")

    if wandb_run is not None:
        wandb_run.summary["delta_acc"] = delta_acc
        wandb_run.finish()


if __name__ == "__main__":
    fire.Fire(finetune_e2e_run)
