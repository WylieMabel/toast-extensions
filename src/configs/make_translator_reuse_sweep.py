"""Generate cifar100 -> group translator-reuse config CSVs, one file per skip.

For each encoder in the chosen group and each single-block skip span, fits a translator on
cifar100 and reuses that exact translator -- same encoder, same span, same skip_translator --
to evaluate the classifier trained on each other dataset: imagenet-1k (in-domain control) and
that group's two domain-shifted datasets. Answers: does a translator estimated on cheap public
natural-image data (cifar100) transfer to a classifier trained on a different, and in two cases
domain-shifted, dataset?

    python src/configs/make_translator_reuse_sweep.py --group group1
    python src/configs/make_translator_reuse_sweep.py --group group2

GROUPS mirrors make_health_skip_grid.py's model/dataset pairing:
    group1 (12-layer dino family): dinov2-base, rad-dino, rad-dino-maira-2
        eval datasets: imagenet-1k, chestxray_pneumonia, chestxray14 -- 11 single-block spans
    group2 (24-layer ViT-L family): vit-large-patch16-224, phikon-v2
        eval datasets: imagenet-1k, patchcamelyon, tcga_ut -- 23 single-block spans
    Only length-1 skips are generated for either group (matching the group1 sweep this was
    modeled on), not the full length-1..4 grid make_health_skip_grid.py uses for the regular
    skip-grid -- a full-length group2 sweep would be 4x the spans and GPU jobs for a family
    that already hit OOM issues at length-1.

FILE LAYOUT: ONE PER SPAN-GROUP, ALL OF A GROUP'S MODELS INSIDE, MODEL-AND-SPAN-CONTIGUOUS
    A translator's fit row and every row that reuses it must share one sweep CSV -- see
    run_pipeline_row_by_row.sh, which deletes a translator as soon as no *later* row in that
    same file still needs it (translators_ready_for_cleanup in translator_keys.py), rather than
    only at the very end. That incremental cleanup is what makes bundling several spans and all
    of a group's models into one file safe: ordering each (span, model) pair's fit row
    immediately before its consumer rows (rather than fitting everything first) means only one
    translator is ever on disk at a time within a file, regardless of how many spans or models
    that file covers.

    --spans-per-file controls how many consecutive spans go in each file (default 1, one span
    per file as before). Raising it trades job-count for a smaller number of concurrent sbatch
    jobs / concurrently-open embedding directories -- e.g. --spans-per-file 2 on group2's 23
    spans gives 11 two-span files + 1 one-span leftover = 12 files instead of 23, each still
    only ever holding one translator on disk at a time. Files are named by their overall span
    range, e.g. experiments_skip_0_2.csv for spans (0,1) and (1,2) together.

    One more file holds the skip="[]" baselines (all models x all datasets), needed once per
    (model, dataset) for delta_acc; skip="[]" rows touch no translator, so bundling every model
    into one baseline file costs nothing.

Writes src/configs/translator_reuse/<group>/experiments_baseline.csv and
experiments_skip_<a>_<b>.csv (one per span-group).
"""

import argparse
import csv
from pathlib import Path

DEFAULT_DONOR_DATASET = "cifar100"

GROUPS = {
    "group1": {
        "encoders": ["facebook/dinov2-base", "microsoft/rad-dino", "microsoft/rad-dino-maira-2"],
        "n_blocks": 12,
        # imagenet-1k is the in-domain (natural-image) control; the other two are group1's
        # medical, domain-shifted datasets -- same trio make_health_skip_grid.py pairs with
        # this model family.
        "eval_datasets": ["imagenet-1k", "chestxray_pneumonia", "chestxray14"],
    },
    "group2": {
        "encoders": ["google/vit-large-patch16-224", "owkin/phikon-v2"],
        "n_blocks": 24,
        # Same logic as group1: imagenet-1k is the in-domain control, patchcamelyon and
        # tcga_ut are group2's domain-shifted pair in make_health_skip_grid.py.
        "eval_datasets": ["imagenet-1k", "patchcamelyon", "tcga_ut"],
    },
    # Both chest datasets are in-domain (radiograph) data for these two encoders, unlike
    # group1's cifar100-donor sweep where the donor is out-of-domain for everyone. Use
    # --donor-dataset chestxray_pneumonia or --donor-dataset chestxray14 to fit on one and
    # reuse the translator on the other -- run both directions to compare. rad-dino and
    # rad-dino-maira-2 were pretrained on chestxray14 (it's part of their public pretraining
    # corpus) but not on chestxray_pneumonia (the Kermany/Kaggle pediatric pneumonia set),
    # so the two directions are not a symmetric control -- see conversation notes.
    "raddino_cross": {
        "encoders": ["microsoft/rad-dino", "microsoft/rad-dino-maira-2"],
        "n_blocks": 12,
        "eval_datasets": ["chestxray_pneumonia", "chestxray14"],
    },
}

ALIAS2ENCODER = {
    "dinov2base": "facebook/dinov2-base",
    "raddino": "microsoft/rad-dino",
    "raddinomaira2": "microsoft/rad-dino-maira-2",
    "vitlarge": "google/vit-large-patch16-224",
    "phikonv2": "owkin/phikon-v2",
}

COLUMNS = [
    "dataset", "encoder", "skip", "mlp_skip", "attn_skip",
    "head_dict", "skip_translator", "mlp_mode", "attn_mode", "fit_dataset",
]

OUT_ROOT = Path("src/configs/translator_reuse")


def row(dataset, encoder, skip, translator, fit_dataset=""):
    return {
        "dataset": dataset, "encoder": encoder, "skip": skip,
        "mlp_skip": "[]", "attn_skip": "[]", "head_dict": "{}",
        "skip_translator": translator, "mlp_mode": "identity", "attn_mode": "identity",
        "fit_dataset": fit_dataset,
    }


def single_block_spans(n_blocks):
    """Every length-1 contiguous span, e.g. n_blocks=12 -> [(0,1) .. (10,11)], 11 total."""
    return [(i, i + 1) for i in range(n_blocks - 1)]


def baseline_rows(translator, encoders, donor_dataset, eval_datasets):
    datasets = dict.fromkeys([donor_dataset, *eval_datasets])  # de-dupe, keep order
    return [
        row(ds, encoder, "[]", translator)
        for encoder in encoders
        for ds in datasets
    ]


def chunk(seq, size):
    return [seq[i:i + size] for i in range(0, len(seq), size)]


def skip_file_rows(spans, encoders, translator, donor_dataset, eval_datasets):
    """Each span's fit row immediately followed by its consumers, one model at a time.

    Nesting spans inside the model loop (rather than the other way around) keeps only one
    translator on disk at a time regardless of how many spans this file covers: model A's
    span-1 translator is fit and fully consumed before model A's span-2 fit even runs.
    """
    consumer_datasets = [ds for ds in eval_datasets if ds != donor_dataset]
    rows = []
    for encoder in encoders:
        for skip_from, skip_to in spans:
            span = f"[({skip_from}, {skip_to})]"
            rows.append(row(donor_dataset, encoder, span, translator, fit_dataset=donor_dataset))
            rows += [
                row(ds, encoder, span, translator, fit_dataset=donor_dataset)
                for ds in consumer_datasets
            ]
    return rows


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, quoting=csv.QUOTE_NONNUMERIC)
        w.writeheader()
        w.writerows(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--group", default="group1", choices=list(GROUPS))
    p.add_argument("--translator", default="linear",
                   help="must round-trip through save/load: linear, lowrank_*, rrr_*, lora_*")
    p.add_argument("--donor-dataset", default=None,
                   help="dataset the translator is fit on; default is group1/group2's usual "
                        "cifar100 donor, but raddino_cross requires an explicit choice of "
                        "chestxray_pneumonia or chestxray14 (run both directions separately)")
    p.add_argument("--out-dir", default=None,
                   help="default: src/configs/translator_reuse/<group>/, or "
                        "<group>_fit_<donor>/ when --donor-dataset overrides the group default")
    p.add_argument("--exclude-encoders", nargs="+", default=[], metavar="ALIAS",
                   choices=list(ALIAS2ENCODER),
                   help="skip these encoders' rows entirely, e.g. to re-run only the models "
                        "that failed a previous attempt without repeating ones that already "
                        "completed: --exclude-encoders dinov2base")
    p.add_argument("--spans-per-file", type=int, default=1,
                   help="bundle this many consecutive skip spans into each file, to trade "
                        "job-count for fewer concurrent sbatch jobs / open embedding dirs")
    args = p.parse_args()

    group = GROUPS[args.group]
    eval_datasets = group["eval_datasets"]
    donor_dataset = args.donor_dataset or DEFAULT_DONOR_DATASET
    if args.group == "raddino_cross" and not args.donor_dataset:
        raise SystemExit("raddino_cross requires --donor-dataset chestxray_pneumonia|chestxray14")
    if donor_dataset not in eval_datasets and args.group == "raddino_cross":
        raise SystemExit(f"--donor-dataset must be one of {eval_datasets} for {args.group}")

    if args.out_dir:
        out_dir = Path(args.out_dir)
    elif donor_dataset == DEFAULT_DONOR_DATASET:
        out_dir = OUT_ROOT / args.group
    else:
        out_dir = OUT_ROOT / f"{args.group}_fit_{donor_dataset}"

    exclude = {ALIAS2ENCODER[a] for a in args.exclude_encoders}
    encoders = [e for e in group["encoders"] if e not in exclude]
    if not encoders:
        raise SystemExit("--exclude-encoders left no encoders to generate rows for")

    print(f"{'file':65s} {'rows':>4s}")
    total_files = total_rows = 0

    path = out_dir / "experiments_baseline.csv"
    rows = baseline_rows(args.translator, encoders, donor_dataset, eval_datasets)
    write_csv(path, rows)
    print(f"{str(path):65s} {len(rows):4d}")
    total_files += 1
    total_rows += len(rows)

    all_spans = single_block_spans(group["n_blocks"])
    for spans in chunk(all_spans, args.spans_per_file):
        range_from, range_to = spans[0][0], spans[-1][1]
        path = out_dir / f"experiments_skip_{range_from}_{range_to}.csv"
        rows = skip_file_rows(spans, encoders, args.translator, donor_dataset, eval_datasets)
        write_csv(path, rows)
        print(f"{str(path):65s} {len(rows):4d}")
        total_files += 1
        total_rows += len(rows)

    print(f"\n{total_files} files, {total_rows} rows total, under {out_dir}/")


if __name__ == "__main__":
    main()
