"""Generate the all-spans-within-a-window skip grid: every legal span inside each model's
hand-picked block window, linear translator only, over imagenet-1k / pneumoniamnist / cifar100.

    python src/configs/make_window_grid.py > src/configs/experiments_window_grid.csv

    # split into 3 independently runnable jobs
    python src/configs/make_window_grid.py --split 3 \
        --out-prefix src/configs/experiments_window_grid

    # what would be emitted, and what already exists elsewhere
    python src/configs/make_window_grid.py --dry-run

THE WINDOWS
    One or two windows per encoder (WINDOWS below), given as inclusive ranges of *skip
    endpoints*, not of dropped blocks. A span (a, b) is emitted for every lo <= a < b <= hi, so
    the window's first block is never dropped -- it is the source the translator maps from.

        vit-large   6-23          vit-base   5-11        deit-small  4-7  | 8-11
        rad-dino    0-3           deit-base  0-4 | 6-11  dinov2-base 8-10

    Two-window models get each window's grid separately and never a span straddling the gap:
    no (7, 8) on deit-small, no (4, 6) on deit-base. That is the "don't mix and match" rule,
    and it falls out for free because every emitted skip list holds exactly one span.

WHAT EACH (dataset, encoder) EXPANDS TO
    The all-empty baseline first -- train_skipped_full.py resolves delta_acc against it inside
    its OWN results CSV, so every file needs one per (dataset, model) -- then one row per span
    with skip_translator=linear. The baseline carries translator=identity by repo convention;
    with nothing to bridge the translator is never invoked, and the baseline lookup in
    train_skipped_full.py deliberately does not match on it.

    Spans per window of k endpoints is k*(k-1)/2: 153 for vit-large, 21 for vit-base, 6+6 for
    deit-small, 6 for rad-dino, 10+15 for deit-base, 3 for dinov2-base. 220 spans per dataset,
    so 678 rows over the three datasets including the 18 baselines.

OVERLAP WITH EXISTING CONFIGS
    The three imagenet-1k spacing sweeps already ran distance-1 and distance-2 spans with a
    linear translator, so a few dozen rows here duplicate them. They are emitted anyway, which
    keeps each (dataset, encoder) block self-contained against its own baseline and results
    file; the report on stderr counts the duplicates, and --dedup-rows drops them if the GPU
    time matters more (the baseline is always kept).

SPLITTING
    --split N writes N cost-balanced files, whole (dataset, encoder) blocks never sliced, since
    each block needs its baseline in the same results CSV and concurrent jobs must write to
    different ones:

        CONFIG_CSV=src/configs/experiments_window_grid_part1.csv \
        RESULTS_CSV_NAME=results_window_grid_part1.csv \
            sbatch src/run_scripts/run_pipeline_row_by_row.sh
"""

import argparse
import csv
import sys
from pathlib import Path

# Inclusive ranges of legal skip endpoints, per encoder. Endpoints are layer-output keys
# 0 .. n_blocks-1, so a window's upper bound can never exceed that.
WINDOWS = {
    "google/vit-large-patch16-224": [(6, 23)],
    "google/vit-base-patch16-224": [(5, 11)],
    "facebook/deit-small-patch16-224": [(4, 7), (8, 11)],
    "microsoft/rad-dino": [(0, 3)],
    "facebook/deit-base-patch16-224": [(0, 4), (6, 11)],
    "facebook/dinov2-base": [(8, 10)],
}

# Mirrors MODEL2NUM_LAYERS in src/toast/utils/dictionaries.py, copied rather than imported so
# this generator stays runnable without the torch/latentis stack (same as make_skip_grid).
MODEL2NUM_LAYERS = {
    "facebook/deit-small-patch16-224": 12,
    "facebook/deit-base-patch16-224": 12,
    "facebook/dinov2-base": 12,
    "google/vit-base-patch16-224": 12,
    "google/vit-large-patch16-224": 24,
    "microsoft/rad-dino": 12,
}

ALIASES = {
    "deitsmall": "facebook/deit-small-patch16-224",
    "deitbase": "facebook/deit-base-patch16-224",
    "dinobase": "facebook/dinov2-base",
    "vitbase": "google/vit-base-patch16-224",
    "vitlarge": "google/vit-large-patch16-224",
    "raddino": "microsoft/rad-dino",
}

# Rough per-row GPU cost relative to deit-small, used only to balance --split; see MODEL2COST in
# make_skip_grid.py for why rad-dino is as expensive as vit-large (518px input, ~1370 tokens).
MODEL2COST = {
    "facebook/deit-small-patch16-224": 1.0,
    "facebook/deit-base-patch16-224": 2.0,
    "facebook/dinov2-base": 2.0,
    "google/vit-base-patch16-224": 2.0,
    "google/vit-large-patch16-224": 4.0,
    "microsoft/rad-dino": 4.0,
}

DEFAULT_ENCODERS = list(WINDOWS)
DEFAULT_DATASETS = ["imagenet-1k", "pneumoniamnist", "cifar100"]
DEFAULT_OUT_PREFIX = "src/configs/experiments_window_grid"

COLUMNS = [
    "dataset", "encoder", "skip", "mlp_skip", "attn_skip",
    "head_dict", "skip_translator", "mlp_mode", "attn_mode", "fit_dataset",
]


def row(dataset, encoder, skip, translator):
    return {
        "dataset": dataset, "encoder": encoder, "skip": skip,
        "mlp_skip": "[]", "attn_skip": "[]", "head_dict": "{}",
        "skip_translator": translator, "mlp_mode": "identity", "attn_mode": "identity",
        "fit_dataset": "",
    }


def spans(windows, n_blocks):
    """Every legal single span inside each window, as skip strings, shallowest first.

    Ordered by start then by end, so a run cut short has covered the early positions at all
    widths rather than one width at all positions.
    """
    out = []
    for lo, hi in windows:
        if not 0 <= lo < hi <= n_blocks - 1:
            raise SystemExit(
                f"Window ({lo}, {hi}) is not a legal endpoint range for a {n_blocks}-block "
                f"encoder: need 0 <= lo < hi <= {n_blocks - 1}"
            )
        out += [f"[({a}, {b})]" for a in range(lo, hi) for b in range(a + 1, hi + 1)]
    return out


def existing_skips(config_dir, out_prefix):
    """(dataset, encoder) -> {(normalised skip, translator)} over every other experiments*.csv.

    Only whole-block-skip rows count: a row that also sets mlp_skip / attn_skip / head_dict is a
    different experiment even when its `skip` column matches. This generator's own output is
    excluded -- it writes into the directory it scans, so a second run would otherwise read back
    the first and report the whole grid as already covered.
    """
    own = {Path(out_prefix).name, Path(DEFAULT_OUT_PREFIX).name}
    have = {}
    for path in sorted(Path(config_dir).glob("experiments*.csv")):
        if any(path.name == f"{o}.csv" or path.name.startswith(f"{o}_part") for o in own):
            continue
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                if not r.get("dataset") or not r.get("encoder"):
                    continue
                if r.get("mlp_skip", "[]") != "[]" or r.get("attn_skip", "[]") != "[]":
                    continue
                if r.get("head_dict", "{}") not in ("{}", ""):
                    continue
                have.setdefault((r["dataset"], r["encoder"]), set()).add(
                    (r["skip"].replace(" ", ""), r["skip_translator"])
                )
    return have


def build(datasets, encoders, translator, have, dedup_rows):
    """One block of rows per (dataset, encoder), plus a report of what was emitted or skipped.

    Blocks are kept whole rather than flattened: --split assigns them one at a time, and a block
    sliced across two files would leave its rows without the baseline they are compared against.
    """
    blocks, report = [], []
    # Dataset-major so each dataset's block is contiguous: a run cut short still yields complete
    # results for the datasets it reached (same ordering rule as make_skip_grid.py).
    for ds in datasets:
        for enc in encoders:
            grid = spans(WINDOWS[enc], MODEL2NUM_LAYERS[enc])
            seen = have.get((ds, enc), set())
            dup = [s for s in grid if (s.replace(" ", ""), translator) in seen]

            emit = [s for s in grid if s not in dup] if dedup_rows else grid
            if not emit:
                report.append((ds, enc, 0, len(grid), "fully covered"))
                continue

            # The baseline leads every block regardless of coverage elsewhere: delta_acc is
            # resolved against it within this file's own results.
            rows = [row(ds, enc, "[]", "identity")]
            rows += [row(ds, enc, s, translator) for s in emit]
            blocks.append((ds, enc, rows))
            report.append((ds, enc, len(rows), len(grid),
                           f"{len(dup)} already present" if dup else ""))
    return blocks, report


def partition(blocks, n_parts):
    """Assign whole blocks to n_parts, balancing estimated cost (longest-processing-time).

    Heaviest block first into whichever part is currently lightest. Ties break on the blocks'
    original order, so regenerating gives the same files.
    """
    parts = [[] for _ in range(n_parts)]
    loads = [0.0] * n_parts
    order = sorted(
        range(len(blocks)),
        key=lambda i: (-len(blocks[i][2]) * MODEL2COST.get(blocks[i][1], 1.0), i),
    )
    for i in order:
        p = loads.index(min(loads))
        parts[p].append(i)
        loads[p] += len(blocks[i][2]) * MODEL2COST.get(blocks[i][1], 1.0)
    # Restore the dataset-major order within each part; the greedy pass visits blocks by size.
    return [[blocks[i] for i in sorted(idxs)] for idxs in parts], loads


def write_csv(handle, blocks):
    w = csv.DictWriter(handle, fieldnames=COLUMNS, quoting=csv.QUOTE_NONNUMERIC)
    w.writeheader()
    for _, _, rows in blocks:
        w.writerows(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    p.add_argument("--encoders", nargs="+", default=DEFAULT_ENCODERS,
                   help="aliases or HF ids; see ALIASES. Must have a window in WINDOWS")
    p.add_argument("--translator", default="linear",
                   help="translator for the span rows (baselines stay identity)")
    p.add_argument("--config-dir", default="src/configs",
                   help="directory scanned for already-covered rows")
    p.add_argument("--dedup-rows", action="store_true",
                   help="drop spans that already exist in another config")
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

    encoders = [ALIASES.get(e, e) for e in args.encoders]
    unknown = [e for e in encoders if e not in WINDOWS]
    if unknown:
        raise SystemExit(f"No window for: {unknown}. Known: {sorted(WINDOWS)}")

    have = {} if args.no_exclude else existing_skips(args.config_dir, args.out_prefix)
    blocks, report = build(args.datasets, encoders, args.translator, have, args.dedup_rows)
    total = sum(len(rows) for _, _, rows in blocks)

    print(f"# {total} rows over {len(args.datasets)} datasets x {len(encoders)} encoders "
          f"({args.translator} translator, {len(blocks)} baselines)", file=sys.stderr)
    for ds, enc, emitted, wanted, note in report:
        windows = " | ".join(f"{lo}-{hi}" for lo, hi in WINDOWS[enc])
        suffix = f"  [{note}]" if note else ""
        print(f"#   {ds:16s} {enc:34s} window {windows:12s} "
              f"{emitted:4d} rows / {wanted} spans{suffix}", file=sys.stderr)

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
