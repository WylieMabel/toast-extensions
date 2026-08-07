"""Generate the translator-transferability config CSV.

Tests whether a closed-form translator fitted on one dataset still works on another. If it
does, the translator is capturing structure intrinsic to the *model* rather than statistics of
the dataset it was fitted on.

    python src/configs/make_transfer_sweep.py > src/configs/experiments_transfer.csv

ROW ORDER IS LOAD-BEARING
    A transfer row loads the translator saved by the row that fitted it, keyed on the fitting
    dataset. So every donor row (fit_dataset == dataset) must appear *before* any row that
    transfers from it -- run_pipeline_row_by_row.sh works through the CSV top to bottom, and
    translators persist in data/translators between rows.

    Get this wrong and load_translator raises FileNotFoundError naming the missing key, so it
    fails loudly rather than silently, but the run still dies partway. Hence the two explicit
    phases below.

    Each dataset also needs its own all-empty baseline row before its skip rows, or delta_acc
    comes out NaN for that dataset.

WHAT THE COMPARISON SHOWS
    For each eval dataset the sweep produces: an identity floor (no translator), a
    same-dataset control (the normal TOAST setting), and three transfers at increasing domain
    distance. The control is also the correctness check -- it must reproduce the accuracy of
    the equivalent run with no fit_dataset column at all.
"""

import argparse
import csv
import sys

ENCODER = "microsoft/rad-dino"

# Lowest-influence adjacent block for rad-dino, from the layer_priority block scores
# (blocks_dropped=1 -> skip_from=9, skip_to=10). Re-derive if the scores are recomputed.
SPAN = "[(9, 10)]"

# Datasets that translators are fitted on and transferred between, ordered by distance from
# dermamnist: itself, another medical imaging task, a different medical modality, then
# natural images as the far-domain control.
EVAL_DATASETS = ["dermamnist", "pneumoniamnist"]
DONOR_DATASETS = ["dermamnist", "pneumoniamnist", "pathmnist", "cifar100"]

COLUMNS = [
    "dataset", "encoder", "skip", "mlp_skip", "attn_skip",
    "head_dict", "skip_translator", "mlp_mode", "attn_mode", "fit_dataset",
]


def row(dataset, skip, translator, fit_dataset=""):
    return {
        "dataset": dataset, "encoder": ENCODER, "skip": skip,
        "mlp_skip": "[]", "attn_skip": "[]", "head_dict": "{}",
        "skip_translator": translator, "mlp_mode": "identity", "attn_mode": "identity",
        "fit_dataset": fit_dataset,
    }


def build(translator):
    rows = []

    # Phase 1 -- donors. Each fits its own translator and saves it under its dataset's key.
    # These must all run before phase 2. They double as ordinary results in their own right.
    for ds in DONOR_DATASETS:
        rows.append(row(ds, "[]", "identity"))                 # baseline, first for this ds
        if ds in EVAL_DATASETS:
            rows.append(row(ds, SPAN, "identity"))             # no-translator floor
        rows.append(row(ds, SPAN, translator, fit_dataset=ds))  # fits + saves

    # Phase 2 -- transfers. Every translator they need is on disk by now.
    for eval_ds in EVAL_DATASETS:
        for donor in DONOR_DATASETS:
            if donor == eval_ds:
                continue  # already covered as the same-dataset control in phase 1
            rows.append(row(eval_ds, SPAN, translator, fit_dataset=donor))

    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--translator", default="linear",
                   help="must round-trip through save/load: linear, lowrank_*, rrr_*, lora_*")
    args = p.parse_args()

    rows = build(args.translator)
    w = csv.DictWriter(sys.stdout, fieldnames=COLUMNS, quoting=csv.QUOTE_NONNUMERIC)
    w.writeheader()
    w.writerows(rows)

    n_transfer = sum(1 for r in rows if r["fit_dataset"] not in ("", r["dataset"]))
    print(f"# {len(rows)} rows: {len(rows) - n_transfer} donor/baseline + {n_transfer} transfer",
          file=sys.stderr)


if __name__ == "__main__":
    main()
