"""Writes the s5 translator-transfer sweep configs. Run from the project root:
    python thesis/s5_translator_transfer/make_configs.py

One evaluation x donor combo per pair of datasets within a group: translators are fitted on the
donor dataset and reused, unchanged, on the evaluation dataset. Single-layer skips only.

Two heads are scored on every skipped row:
  setup 1: a head retrained on that row's own embeddings (accuracy/auroc/auprc)
  setup 2: the head trained on the dataset's unskipped baseline, frozen (frozen_head_*)
Every config (and every shard) starts with a baseline row for BOTH datasets, so setup 2 has a
frozen head for the evaluation dataset's transfer rows and for the donor's own fitting rows
(the own-translator comparison).

A transfer row (dataset=eval, fit_dataset=donor) can only load a translator that a fitting row
(dataset=donor) in the SAME config saved (see toast.utils.translator_keys), so every config holds
both. Each fitting row is followed directly by its transfer row, so the runner can delete each
translator as soon as it has been used.

Every combo is split into one config per model. Group 2 models (24 layers, ~45h each) are split
again into two halves of the skip range so each job stays under its time limit;
analysis/collect_accuracies.py merges the shards back into one CSV per combo.

CIFAR-100 is the out-of-domain donor for both groups (group{g}_crossfit_medical_from_cifar100_*).
One CIFAR-100 fitting row serves both datasets of the group: each is followed by a transfer row
on each of them. Baselines are the two medical datasets only; CIFAR-100's own head is not used.
"""
import csv
from pathlib import Path

CONFIGS = Path("thesis/s5_translator_transfer/configs")
HEADER = ["dataset", "encoder", "skip", "mlp_skip", "attn_skip", "head_dict",
          "skip_translator", "mlp_mode", "attn_mode", "fit_dataset"]

# Dataset name -> slug used in file names (same slugs as the s4 configs).
SLUGS = {"chestxray14_multilabel": "chestxray14multilabel",
         "chestxray_pneumonia": "chestxraypneumonia",
         "patchcamelyon": "patchcamelyon", "tcga_ut": "tcgaut"}

# group -> (datasets, {encoder: (slug, number of layers)}, shards per model or None for one job)
GROUPS = {
    1: (["chestxray14_multilabel", "chestxray_pneumonia"],
        {"microsoft/rad-dino": ("raddino", 12),
         "microsoft/rad-dino-maira-2": ("raddinomaira2", 12),
         "facebook/dinov2-base": ("dinov2base", 12)},
        None),
    2: (["patchcamelyon", "tcga_ut"],
        {"google/vit-large-patch16-224": ("vitlarge", 24),
         "owkin/phikon-v2": ("phikonv2", 24)},
        2),
}

OOD_DONOR = "cifar100"
# Shards per model for the CIFAR-100 donor configs: they hold 3 rows per skip instead of 2, so
# pathology is split into thirds (~18-20h each) to pack into launch_s5.sh's lanes.
OOD_SHARDS = {1: None, 2: 3}


def baseline_row(dataset, encoder):
    return [dataset, encoder, "[]", "[]", "[]", "{}", "linear", "identity", "identity", ""]


def pair_rows(eval_ds, donor, encoder, skips):
    # Baselines first: they save the frozen heads the skipped rows below score (setup 2), and
    # fill original_accuracy so delta_acc is defined for every row.
    rows = [baseline_row(ds, encoder) for ds in (eval_ds, donor)]
    for i in skips:
        skip = f"[({i}, {i + 1})]"
        base = [encoder, skip, "[]", "[]", "{}", "linear", "identity", "identity"]
        rows.append([donor, *base, ""])        # fits the translator on the donor
        rows.append([eval_ds, *base, donor])   # reuses it on the evaluation dataset
    return rows


def ood_rows(datasets, encoder, skips):
    rows = [baseline_row(ds, encoder) for ds in datasets]
    for i in skips:
        skip = f"[({i}, {i + 1})]"
        base = [encoder, skip, "[]", "[]", "{}", "linear", "identity", "identity"]
        rows.append([OOD_DONOR, *base, ""])                    # fits on CIFAR-100
        rows += [[ds, *base, OOD_DONOR] for ds in datasets]   # reused on each medical dataset
    return rows


def write_sharded(name, shards, n_layers, make_rows):
    skips = list(range(n_layers - 1))
    if shards is None:
        write(name, make_rows(skips))
        return
    size = -(-len(skips) // shards)
    for part in range(shards):
        write(f"{name}_part{part + 1}", make_rows(skips[part * size:(part + 1) * size]))


def write(name, rows):
    path = CONFIGS / f"experiments_{name}.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f, quoting=csv.QUOTE_ALL)
        w.writerow(HEADER)
        w.writerows(rows)
    print(f"{path}: {len(rows)} rows")


def main():
    CONFIGS.mkdir(parents=True, exist_ok=True)
    for group, (datasets, encoders, shards) in GROUPS.items():
        for eval_ds in datasets:
            for donor in datasets:
                if donor == eval_ds:
                    continue   # self-fit results are s4
                combo = f"group{group}_crossfit_{SLUGS[eval_ds]}_from_{SLUGS[donor]}"
                for encoder, (enc_slug, n_layers) in encoders.items():
                    write_sharded(f"{combo}_{enc_slug}", shards, n_layers,
                                  lambda s: pair_rows(eval_ds, donor, encoder, s))
        for encoder, (enc_slug, n_layers) in encoders.items():
            write_sharded(f"group{group}_crossfit_medical_from_{OOD_DONOR}_{enc_slug}", OOD_SHARDS[group],
                          n_layers, lambda s: ood_rows(datasets, encoder, s))


if __name__ == "__main__":
    main()
