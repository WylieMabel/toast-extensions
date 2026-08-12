"""Generate two-phase transfer learning configs: fit first, then transfer.

Phase 1: Fitting — train on source datasets, translators fitted on same dataset
Phase 2: Transfer — use fitted translators on target datasets

This split ensures translators are available before transfer rows run, and allows
independent scheduling of fitting (can use fewer GPUs) vs transfer evaluation.

Usage:
    python src/configs/make_transfer_experiments_two_phase.py

This generates:
    - experiments_transfer_learning_phase1_fit_part{1,2}.csv (fitting configs)
    - experiments_transfer_learning_phase2_transfer_part{1,2}.csv (transfer configs)
"""

import argparse
import csv
import sys
from pathlib import Path

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

SOURCE_DATASETS = [
    "chestmnist",
    "pneumoniamnist",
    "cifar100",
    "dermamnist",
    "imagenet-1k",
]

TARGET_DATASETS = ["chestmnist", "pneumoniamnist"]

DEFAULT_ENCODERS = list(WINDOWS)


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


def build_phase1_fitting(datasets, encoders, translator, have, dedup_rows, min_drops, stride):
    """Build fitting configs: source dataset rows with fit_dataset empty."""
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

            rows = [row(ds, enc, "[]", "identity")]
            rows += [row(ds, enc, s, translator) for s in emit]
            blocks.append((ds, enc, rows))
            report.append(
                (ds, enc, len(rows), len(grid),
                 f"{len(dup)} already present" if dup else "")
            )

    return blocks, report


def build_phase2_transfer(datasets, target_datasets, encoders, translator, have, dedup_rows, min_drops, stride):
    """Build transfer configs: cross-dataset translator evaluation."""
    blocks, report = [], []

    for ds in datasets:
        for enc in encoders:
            grid = patterns(WINDOWS[enc], MODEL2NUM_LAYERS[enc], min_drops, stride)
            if not grid:
                continue

            seen = have.get((ds, enc), set())
            dup = [s for s in grid if (s.replace(" ", ""), translator) in seen]
            emit = [s for s in grid if s not in dup] if dedup_rows else grid
            if not emit:
                continue

            rows = []
            for target_ds in target_datasets:
                if target_ds != ds:
                    # Baseline with transfer
                    r = row(ds, enc, "[]", "identity")
                    r["fit_dataset"] = target_ds
                    rows.append(r)

                    # Patterns with transfer
                    for s in emit:
                        r = row(ds, enc, s, translator)
                        r["fit_dataset"] = target_ds
                        rows.append(r)

            if rows:
                blocks.append((ds, enc, rows))
                n_patterns = len(emit)
                n_targets = len([t for t in target_datasets if t != ds])
                report.append(
                    (ds, enc, len(rows), n_patterns * n_targets,
                     f"transfer to {n_targets} target(s)")
                )

    return blocks, report


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--datasets", nargs="+", default=SOURCE_DATASETS)
    p.add_argument("--target-datasets", nargs="+", default=TARGET_DATASETS)
    p.add_argument("--encoders", nargs="+", default=DEFAULT_ENCODERS)
    p.add_argument("--translator", default="linear")
    p.add_argument("--stride", type=int, default=2)
    p.add_argument("--min-drops", type=int, default=2)
    p.add_argument("--config-dir", default="src/configs")
    p.add_argument("--dedup-rows", action="store_true")
    p.add_argument("--no-exclude", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--split-fit", type=int, default=2, metavar="N",
                   help="split phase 1 into N files")
    p.add_argument("--split-transfer", type=int, default=2, metavar="N",
                   help="split phase 2 into N files")
    args = p.parse_args()

    if args.stride < 2:
        p.error("--stride must be at least 2")
    if args.min_drops < 1:
        p.error("--min-drops must be at least 1")

    encoders = [ALIASES.get(e, e) for e in args.encoders]
    unknown = [e for e in encoders if e not in WINDOWS]
    if unknown:
        raise SystemExit(f"No window for: {unknown}")

    have = {} if args.no_exclude else existing_skips(args.config_dir, "experiments_transfer_learning")

    # Build phase 1: fitting
    phase1_blocks, phase1_report = build_phase1_fitting(
        args.datasets, encoders, args.translator, have, args.dedup_rows,
        args.min_drops, args.stride
    )
    phase1_total = sum(len(rows) for _, _, rows in phase1_blocks)

    # Build phase 2: transfer
    phase2_blocks, phase2_report = build_phase2_transfer(
        args.datasets, args.target_datasets, encoders, args.translator, have, args.dedup_rows,
        args.min_drops, args.stride
    )
    phase2_total = sum(len(rows) for _, _, rows in phase2_blocks)

    print(f"# Phase 1 (Fitting): {phase1_total} rows", file=sys.stderr)
    for ds, enc, emitted, wanted, note in phase1_report:
        windows = " | ".join(f"{lo}-{hi}" for lo, hi in WINDOWS[enc])
        suffix = f"  [{note}]" if note else ""
        print(f"#   {ds:16s} {enc:34s} {emitted:4d} rows{suffix}", file=sys.stderr)

    print(f"# Phase 2 (Transfer): {phase2_total} rows", file=sys.stderr)
    for ds, enc, emitted, wanted, note in phase2_report:
        windows = " | ".join(f"{lo}-{hi}" for lo, hi in WINDOWS[enc])
        suffix = f"  [{note}]" if note else ""
        print(f"#   {ds:16s} {enc:34s} {emitted:4d} rows{suffix}", file=sys.stderr)

    if args.dry_run:
        return

    # Write phase 1
    if args.split_fit == 1:
        path = "src/configs/experiments_transfer_learning_phase1_fit.csv"
        with open(path, "w", newline="") as f:
            write_csv(f, phase1_blocks)
        print(f"# wrote {path}: {phase1_total} rows", file=sys.stderr)
    else:
        parts, loads = partition(phase1_blocks, args.split_fit)
        for i, (part, load) in enumerate(zip(parts, loads), start=1):
            path = f"src/configs/experiments_transfer_learning_phase1_fit_part{i}.csv"
            with open(path, "w", newline="") as f:
                write_csv(f, part)
            n = sum(len(rows) for _, _, rows in part)
            print(f"# wrote {path}: {n} rows, cost {load:.0f}", file=sys.stderr)

    # Write phase 2
    if args.split_transfer == 1:
        path = "src/configs/experiments_transfer_learning_phase2_transfer.csv"
        with open(path, "w", newline="") as f:
            write_csv(f, phase2_blocks)
        print(f"# wrote {path}: {phase2_total} rows", file=sys.stderr)
    else:
        parts, loads = partition(phase2_blocks, args.split_transfer)
        for i, (part, load) in enumerate(zip(parts, loads), start=1):
            path = f"src/configs/experiments_transfer_learning_phase2_transfer_part{i}.csv"
            with open(path, "w", newline="") as f:
                write_csv(f, part)
            n = sum(len(rows) for _, _, rows in part)
            print(f"# wrote {path}: {n} rows, cost {load:.0f}", file=sys.stderr)


if __name__ == "__main__":
    main()
