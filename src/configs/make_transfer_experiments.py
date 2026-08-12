"""Generate transfer learning experiments: train on source datasets, fit translators on target datasets.

This script extends the alternating window grid to create:
1. Baseline experiments for source datasets (chestmnist, pneumoniamnist, cifar100, dermamnist, imagenet-1k)
2. Transfer variants where translators are fitted on target datasets (chestmnist, pneumoniamnist)

Usage:
    python src/configs/make_transfer_experiments.py \\
        > src/configs/experiments_transfer_learning.csv

    # Split into N cost-balanced files for parallel execution
    python src/configs/make_transfer_experiments.py --split 4 \\
        --out-prefix src/configs/experiments_transfer_learning
"""

import argparse
import csv
import sys

from make_window_grid import (
    ALIASES,
    COLUMNS,
    MODEL2COST,
    MODEL2NUM_LAYERS,
    WINDOWS,
    existing_skips,
    partition,
    row,
    write_csv,
)

# Source datasets: these are used as the training/encoder dataset
SOURCE_DATASETS = [
    "chestmnist",
    "pneumoniamnist",
    "cifar100",
    "dermamnist",
    "imagenet-1k",
]

# Target datasets: these are where we fit the translators (for transfer assessment)
TARGET_DATASETS = ["chestmnist", "pneumoniamnist"]

DEFAULT_ENCODERS = list(WINDOWS)
DEFAULT_OUT_PREFIX = "src/configs/experiments_transfer_learning"


def patterns(windows, n_blocks, min_drops, stride):
    """Alternating skip specs inside each window, as strings, fewest drops first."""
    out = []
    for lo, hi in windows:
        if not 0 <= lo < hi <= n_blocks - 1:
            raise SystemExit(
                f"Window ({lo}, {hi}) is not a legal endpoint range for a {n_blocks}-block "
                f"encoder: need 0 <= lo < hi <= {n_blocks - 1}"
            )
        n_drops = min_drops
        while True:
            reach = stride * (n_drops - 1) + 1
            starts = range(lo, hi - reach + 1)
            if not starts:
                break
            for start in starts:
                spans = [(start + stride * i, start + stride * i + 1) for i in range(n_drops)]
                out.append(str(spans))
            n_drops += 1
    return out


def build_transfer_experiments(datasets, target_datasets, encoders, translator, have, dedup_rows, min_drops, stride):
    """Build source dataset experiments + transfer variants.

    For each (dataset, encoder) combination:
      - Emit baseline and all skip patterns as-is (fit_dataset empty)
      - Then emit the same patterns again with fit_dataset set to each target dataset
    """
    blocks, report = [], []

    for ds in datasets:
        for enc in encoders:
            grid = patterns(WINDOWS[enc], MODEL2NUM_LAYERS[enc], min_drops, stride)
            if not grid:
                report.append((ds, enc, 0, 0, "no pattern fits the window"))
                continue

            seen = have.get((ds, enc), set())
            dup = [s for s in grid if (s.replace(" ", ""), translator) in seen]
            emit = [s for s in grid if s not in dup] if dedup_rows else grid
            if not emit:
                report.append((ds, enc, 0, len(grid), "fully covered"))
                continue

            # Source experiments (fit_dataset empty - no transfer)
            rows = [row(ds, enc, "[]", "identity")]
            rows += [row(ds, enc, s, translator) for s in emit]

            # Transfer variants (fit_dataset set to each target)
            # Skip transfer if source == target (would be the same as above)
            for target_ds in target_datasets:
                if target_ds != ds:
                    for s in emit:
                        r = row(ds, enc, s, translator)
                        r["fit_dataset"] = target_ds
                        rows.append(r)
                    # Also add baseline with transfer target
                    r = row(ds, enc, "[]", "identity")
                    r["fit_dataset"] = target_ds
                    rows.append(r)

            blocks.append((ds, enc, rows))
            n_transfer_patterns = len(emit) * (len([t for t in target_datasets if t != ds]))
            report.append(
                (ds, enc, len(rows), len(grid),
                 f"{len(dup)} already present; +{n_transfer_patterns} transfer variants" if dup
                 else f"+{n_transfer_patterns} transfer variants")
            )

    return blocks, report


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--datasets", nargs="+", default=SOURCE_DATASETS,
                   help="source datasets (training/encoder datasets)")
    p.add_argument("--target-datasets", nargs="+", default=TARGET_DATASETS,
                   help="target datasets for transfer learning (fit_dataset values)")
    p.add_argument("--encoders", nargs="+", default=DEFAULT_ENCODERS,
                   help="aliases or HF ids; see ALIASES. Must have a window in WINDOWS")
    p.add_argument("--translator", default="linear",
                   help="translator for the pattern rows (baselines stay identity)")
    p.add_argument("--stride", type=int, default=2, metavar="S",
                   help="blocks between consecutive drop starts; 2 keeps one block between "
                        "drops, 3 keeps two")
    p.add_argument("--min-drops", type=int, default=2, metavar="N",
                   help="smallest pattern to emit; 1 would re-emit single spans already in "
                        "experiments_window_grid.csv")
    p.add_argument("--config-dir", default="src/configs",
                   help="directory scanned for already-covered rows")
    p.add_argument("--dedup-rows", action="store_true",
                   help="drop patterns that already exist in another config")
    p.add_argument("--no-exclude", action="store_true",
                   help="do not scan existing configs at all")
    p.add_argument("--dry-run", action="store_true",
                   help="print the per-combination report only, write no CSV")
    p.add_argument("--split", type=int, default=1, metavar="N",
                   help="write N cost-balanced files instead of one CSV on stdout")
    p.add_argument("--out-prefix", default=DEFAULT_OUT_PREFIX,
                   help="path prefix for --split output; files are <prefix>_partK.csv")
    args = p.parse_args()

    if args.split < 1:
        p.error("--split must be at least 1")
    if args.stride < 2:
        p.error("--stride must be at least 2")
    if args.min_drops < 1:
        p.error("--min-drops must be at least 1")

    encoders = [ALIASES.get(e, e) for e in args.encoders]
    unknown = [e for e in encoders if e not in WINDOWS]
    if unknown:
        raise SystemExit(f"No window for: {unknown}. Known: {sorted(WINDOWS)}")

    have = {} if args.no_exclude else existing_skips(args.config_dir, args.out_prefix)
    blocks, report = build_transfer_experiments(
        args.datasets, args.target_datasets, encoders, args.translator,
        have, args.dedup_rows, args.min_drops, args.stride
    )
    total = sum(len(rows) for _, _, rows in blocks)

    print(
        f"# {total} rows over {len(args.datasets)} source datasets x {len(encoders)} encoders "
        f"x (1 + {len(args.target_datasets)} transfer targets) "
        f"(stride {args.stride}, >={args.min_drops} drops, {args.translator} translator, "
        f"{len(blocks)} (dataset, encoder) blocks)",
        file=sys.stderr
    )
    for ds, enc, emitted, wanted, note in report:
        windows = " | ".join(f"{lo}-{hi}" for lo, hi in WINDOWS[enc])
        suffix = f"  [{note}]" if note else ""
        print(
            f"#   {ds:16s} {enc:34s} window {windows:12s} "
            f"{emitted:5d} rows / {wanted} patterns{suffix}",
            file=sys.stderr
        )

    if args.dry_run:
        return

    if args.split == 1:
        write_csv(sys.stdout, blocks)
        return

    parts, loads = partition(blocks, args.split)
    for i, (part, load) in enumerate(zip(parts, loads), start=1):
        path = f"{args.out_prefix}_part{i}.csv"
        with open(path, "w", newline="") as f:
            write_csv(f, part)
        n = sum(len(rows) for _, _, rows in part)
        print(f"# wrote {path}: {n} rows, cost {load:.0f}", file=sys.stderr)


if __name__ == "__main__":
    main()
