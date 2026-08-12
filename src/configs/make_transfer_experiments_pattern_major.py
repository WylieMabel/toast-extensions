"""Generate transfer learning configs grouped by skip pattern to limit translator storage.

This organizes experiments so only one skip pattern's translators are on disk at a time.
For each skip pattern, we:
  1. Run fitting rows (fit_dataset="")
  2. Run transfer rows (fit_dataset="target")
  3. Delete all translators for that pattern
  4. Move to next pattern

This limits peak disk usage to ~20-30 translators at once.

Usage:
    python src/configs/make_transfer_experiments_pattern_major.py

Generates: experiments_transfer_learning_pattern_{baseline,pattern1,...,patternN}.csv
where each file contains all datasets/encoders for one skip pattern.
"""

import argparse
import csv
import sys
from collections import defaultdict
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


def build_pattern_major(datasets, target_datasets, encoders, translator, have, dedup_rows, min_drops, stride):
    """Build all experiments grouped by skip pattern (pattern-major order).

    Returns list of (pattern_name, rows) tuples. Each tuple contains all datasets/encoders
    for that pattern, both fitting and transfer rows.
    """
    pattern_rows = defaultdict(list)

    # Collect all patterns
    all_patterns = {}  # pattern_str -> pattern_name
    pattern_order = []

    # Baseline pattern
    baseline_name = "baseline"
    all_patterns["[]"] = baseline_name
    pattern_order.append("[]")

    # Data patterns
    for enc in encoders:
        grid = patterns(WINDOWS[enc], MODEL2NUM_LAYERS[enc], min_drops, stride)
        for i, s in enumerate(grid):
            if s not in all_patterns:
                all_patterns[s] = f"pattern{len(all_patterns)}"
                pattern_order.append(s)

    # Build rows for each pattern
    for pattern_idx, pattern_str in enumerate(pattern_order):
        pattern_name = all_patterns[pattern_str]

        for ds in datasets:
            for enc in encoders:
                grid = patterns(WINDOWS[enc], MODEL2NUM_LAYERS[enc], min_drops, stride)

                # Check if this pattern exists for this encoder
                if pattern_str not in grid and pattern_str != "[]":
                    continue

                seen = have.get((ds, enc), set())

                # Fitting row (fit_dataset="")
                r = row(ds, enc, pattern_str, "identity" if pattern_str == "[]" else translator)
                pattern_rows[pattern_name].append(r)

                # Transfer rows (fit_dataset=target)
                for target_ds in target_datasets:
                    if target_ds != ds:
                        r = row(ds, enc, pattern_str, "identity" if pattern_str == "[]" else translator)
                        r["fit_dataset"] = target_ds
                        pattern_rows[pattern_name].append(r)

    # Convert to list of (pattern_name, rows) with counts
    blocks = [
        (all_patterns[pattern_str], pattern_rows[all_patterns[pattern_str]])
        for pattern_str in pattern_order
    ]

    # Generate report
    report = []
    total_rows = 0
    total_translators = 0

    for pattern_name, rows in blocks:
        n_fitting = sum(1 for r in rows if r["fit_dataset"] == "")
        n_transfer = len(rows) - n_fitting

        # Rough estimate of translators for this pattern:
        # Each (dataset, encoder) pair with fit_dataset="" creates ~1 translator (one per span per encoder)
        # But we're only counting unique (dataset, encoder) combinations, not duplicate patterns
        n_encoder_patterns = 0
        for enc in encoders:
            grid = patterns(WINDOWS[enc], MODEL2NUM_LAYERS[enc], min_drops, stride)
            # Find what datasets have rows for this pattern
            datasets_with_pattern = set(r["dataset"] for r in rows if r["fit_dataset"] == "")
            n_encoder_patterns += len(datasets_with_pattern)

        report.append({
            "pattern": pattern_name,
            "total_rows": len(rows),
            "fitting_rows": n_fitting,
            "transfer_rows": n_transfer,
            "est_translators": n_encoder_patterns,  # Rough estimate
        })
        total_rows += len(rows)
        total_translators += n_encoder_patterns

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

    blocks, report = build_pattern_major(
        args.datasets, args.target_datasets, encoders, args.translator, have,
        args.dedup_rows, args.min_drops, args.stride
    )

    print("# Pattern-Major Transfer Learning Experiments", file=sys.stderr)
    print("# Run patterns sequentially to keep translator storage ~20-30 at a time", file=sys.stderr)
    print("#", file=sys.stderr)

    total_rows = 0
    total_translators = 0
    for item in report:
        total_rows += item["total_rows"]
        total_translators += item["est_translators"]
        print(
            f"# {item['pattern']:12s} {item['total_rows']:4d} rows "
            f"({item['fitting_rows']:3d} fit + {item['transfer_rows']:3d} transfer) "
            f"~{item['est_translators']:2d} translators",
            file=sys.stderr
        )

    print(f"#", file=sys.stderr)
    print(f"# Total: {total_rows} rows across {len(blocks)} patterns", file=sys.stderr)
    print(f"# Peak disk usage: ~{max(item['est_translators'] for item in report)} translators", file=sys.stderr)
    print(f"# Storage: ~{max(item['est_translators'] for item in report) * 2.3:.0f}MB per pattern", file=sys.stderr)

    if args.dry_run:
        return

    # Write one file per pattern
    for pattern_name, rows in blocks:
        path = f"src/configs/experiments_transfer_learning_pattern_{pattern_name}.csv"
        with open(path, "w", newline="") as f:
            # Write header + rows
            f.write(",".join(COLUMNS) + "\n")
            for r in rows:
                f.write(",".join(f'"{r[col]}"' for col in COLUMNS) + "\n")

        n = len(rows)
        print(f"# wrote {path}: {n} rows", file=sys.stderr)

    print(f"#", file=sys.stderr)
    print(f"# Run with:", file=sys.stderr)
    for pattern_name, _ in blocks:
        print(
            f"# CONFIG_CSV=src/configs/experiments_transfer_learning_pattern_{pattern_name}.csv "
            f"RESULTS_CSV_NAME=results_transfer_learning_pattern_{pattern_name}.csv "
            f"sbatch --output=logs/transfer_learning_pattern_{pattern_name}.out.txt "
            f"src/run_scripts/run_pipeline_row_by_row.sh",
            file=sys.stderr
        )


if __name__ == "__main__":
    main()
