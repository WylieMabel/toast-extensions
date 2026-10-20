"""
Report per-split class balance (and majority-class rate) for every dataset this repo trains
or evaluates on, reading straight from the local on-disk copies in DATASET2LOCAL_PATH -- no
download, no HF Hub access.

The majority-class rate is the accuracy a constant predictor gets for free: any trained probe
landing at or below it has learned nothing on that dataset (see download_medmnist.py, which
reports the same number for the MedMNIST variants).

Usage:
    python src/toast/scripts/report_class_balance.py                     # every dataset
    python src/toast/scripts/report_class_balance.py chestxray14 tcga_ut # just these
"""

import sys
from collections import Counter
from pathlib import Path

from datasets import load_from_disk

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from toast.utils.dictionaries import (
    DATASET2LABEL_COLUMN,
    DATASET2LOCAL_PATH,
    MULTILABEL_DATASETS,
)


def report_one(ds_name: str):
    path = DATASET2LOCAL_PATH.get(ds_name)
    if path is None:
        print(f"[{ds_name}] no local path in DATASET2LOCAL_PATH -- skipping")
        return

    print(f"\n{'=' * 60}")
    print(f"  {ds_name}  ({path})")
    print(f"{'=' * 60}")

    try:
        data = load_from_disk(path)
    except Exception as e:
        print(f"  ERROR: could not load from disk: {e}")
        return

    label_col = DATASET2LABEL_COLUMN.get(ds_name, "label")
    is_multilabel = ds_name in MULTILABEL_DATASETS

    for split in data:
        labels = data[split][label_col]
        n = len(labels)
        if is_multilabel:
            n_labels = len(labels[0])
            pos_rate = [sum(row[j] for row in labels) / n for j in range(n_labels)]
            print(f"  {split}: {n} samples, multi-label ({n_labels} labels)")
            print("    positive rate per label: " + ", ".join(f"{p:.3f}" for p in pos_rate))
        else:
            counts = Counter(int(v) for v in labels)
            majority = max(counts.values()) / n
            print(f"  {split}: {n} samples, {len(counts)} classes")
            print("    distribution: " + ", ".join(f"{k}:{counts[k]}" for k in sorted(counts)))
            print(f"    majority-class rate: {majority:.4f} "
                  f"(a probe scoring this has learned nothing)")


if __name__ == "__main__":
    targets = sys.argv[1:] if len(sys.argv) > 1 else sorted(DATASET2LOCAL_PATH)
    for name in targets:
        report_one(name)
