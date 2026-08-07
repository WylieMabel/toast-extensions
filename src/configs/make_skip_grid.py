"""Generate the full no-skip / 1-block / 2-block skip grid across encoders x datasets.

This is the `position` and `width` half of make_spacing_sweep.py (the quadratic `spacing`
group is deliberately left out) run over every encoder x dataset combination, minus the
combinations that are already covered by an existing config.

    # the default grid, written to the file the run scripts expect
    python src/configs/make_skip_grid.py > src/configs/experiments_skip_grid.csv

    # the same grid split into two independently runnable halves
    python src/configs/make_skip_grid.py --split 2 \
        --out-prefix src/configs/experiments_skip_grid

    # see what is skipped and why, without writing anything
    python src/configs/make_skip_grid.py --dry-run

WHAT EACH COMBINATION EXPANDS TO
    Emitted once per translator (identity first, then linear), baseline first:

      1. baseline   -- skip=[]        reference row for delta_acc
      2. position   -- [(i, i+1)]     one block dropped, slid along the depth
      3. width      -- [(i, i+2)]     two adjacent blocks dropped as one span

    A skip (a, b) keeps block a and drops a+1 .. b, and both indices are layer-output keys
    0 .. n_blocks-1, so the last legal single skip is (n-2, n-1) and the last legal double
    is (n-3, n-1). That gives 1 + (n-1) + (n-2) rows per translator: 22 at depth 12, 46 at
    depth 24.

WHAT IS EXCLUDED
    Every experiments*.csv in this directory is scanned for whole-block-skip rows (no
    mlp_skip / attn_skip / head_dict), and any (dataset, encoder) whose entire grid is
    already present is dropped -- currently the three imagenet-1k spacing sweeps.

    Combinations with only partial coverage are emitted in full: the handful of overlapping
    rows live in other configs with their own baselines and result files, and re-running them
    keeps each combination here a complete, self-contained block. Pass --dedup-rows to drop
    those individual duplicates instead (the baseline row is always kept, since
    train_skipped_full.py resolves delta_acc against it).

SPLITTING FOR PARALLEL RUNS
    --split N writes N files that can be submitted as N concurrent jobs. Whole (dataset,
    encoder) combinations are kept together, never sliced across files, because
    train_skipped_full.py looks the baseline up inside its OWN results CSV -- and the two
    jobs must write to different ones, since each row rewrites that file in full and
    concurrent writers would clobber each other:

        CONFIG_CSV=src/configs/experiments_skip_grid_part1.csv \
        RESULTS_CSV_NAME=results_skip_grid_part1.csv \
            sbatch src/run_scripts/run_pipeline_row_by_row.sh

    Embeddings are safe to run concurrently: the directory is hashed from the whole row
    config, so files holding disjoint rows never touch the same one.

    Files are balanced on estimated GPU cost (MODEL2COST below), not row count, so the jobs
    finish at roughly the same time rather than one idling while the other grinds through the
    expensive encoders.
"""

import argparse
import csv
import sys
from pathlib import Path

# Mirrors MODEL2NUM_LAYERS in src/toast/utils/dictionaries.py, copied rather than imported so
# this generator stays runnable without the torch/latentis stack (same as make_spacing_sweep).
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

# Rough per-row GPU cost relative to deit-small, used only to balance --split. Driven by what
# actually dominates a row: encoder width/depth, and for rad-dino the 518px input (~1370 tokens
# against 197 for the 224px models), which costs far more than its 12 blocks suggest. The
# dataset barely enters into it -- encode_vision_full caps every row at 250 samples regardless
# of dataset size. Adjust if the balance comes out wrong in practice; it changes nothing but
# which file a combination lands in.
MODEL2COST = {
    "facebook/deit-small-patch16-224": 1.0,
    "facebook/deit-base-patch16-224": 2.0,
    "facebook/dinov2-base": 2.0,
    "google/vit-base-patch16-224": 2.0,
    "google/vit-large-patch16-224": 4.0,
    "microsoft/rad-dino": 4.0,
}

DEFAULT_OUT_PREFIX = "src/configs/experiments_skip_grid"

DEFAULT_ENCODERS = list(MODEL2NUM_LAYERS)
DEFAULT_DATASETS = ["imagenet-1k", "cifar100", "pneumoniamnist", "dermamnist"]

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


def spans(n_blocks):
    """The skip specs of the grid, in file order, as strings."""
    last = n_blocks - 1  # highest block index a skip may land on
    return (
        ["[]"]
        + [f"[({i}, {i + 1})]" for i in range(0, last)]
        + [f"[({i}, {i + 2})]" for i in range(0, last - 1)]
    )


def existing_skips(config_dir, out_prefix):
    """(dataset, encoder) -> {(normalised skip, translator)} over every experiments*.csv.

    Only whole-block-skip rows count: a row that also sets mlp_skip / attn_skip / head_dict
    is a different experiment even when its `skip` column matches.

    This generator's own output is excluded. It writes into the directory it scans, so a
    second run would otherwise read back the previous one and report the entire grid as
    already covered. Both the requested prefix and the default one are excluded: writing a
    split elsewhere must not make the parts already sitting in src/configs look like
    independent prior coverage.
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
                key = (r["dataset"], r["encoder"])
                have.setdefault(key, set()).add(
                    (r["skip"].replace(" ", ""), r["skip_translator"])
                )
    return have


def build(datasets, encoders, translators, have, dedup_rows):
    """One block of rows per (dataset, encoder), plus a report of what was emitted or skipped.

    Blocks are kept whole rather than flattened: --split assigns them one at a time, and a
    block sliced across two files would leave its rows without the baseline they are compared
    against.
    """
    blocks, report = [], []
    # Dataset-major so each dataset's block is contiguous: a run cut short still yields
    # complete results for the datasets it reached (same ordering rule as make_table3.py).
    for ds in datasets:
        for enc in encoders:
            n_blocks = MODEL2NUM_LAYERS[enc]
            grid = spans(n_blocks)
            seen = have.get((ds, enc), set())

            wanted = [(s, t) for t in translators for s in grid]
            missing = [(s, t) for (s, t) in wanted if (s.replace(" ", ""), t) not in seen]
            if not missing:
                report.append((ds, enc, 0, len(wanted), "fully covered"))
                continue

            if dedup_rows:
                # Keep the baseline regardless: delta_acc is resolved against it per
                # (dataset, model) within this file's own results.
                emit = [(s, t) for (s, t) in wanted if (s, t) in missing or s == "[]"]
            else:
                emit = wanted

            blocks.append((ds, enc, [row(ds, enc, s, t) for (s, t) in emit]))
            report.append((ds, enc, len(emit), len(wanted),
                           f"{len(wanted) - len(missing)} already present"))
    return blocks, report


def partition(blocks, n_parts):
    """Assign whole blocks to n_parts, balancing estimated cost (longest-processing-time).

    Heaviest block first into whichever part is currently lightest. Ties break on the blocks'
    original order, so the split is deterministic -- regenerating gives the same files.
    """
    parts = [[] for _ in range(n_parts)]
    loads = [0.0] * n_parts
    order = sorted(
        range(len(blocks)),
        key=lambda i: (-len(blocks[i][2]) * MODEL2COST.get(blocks[i][1], 1.0), i),
    )
    for i in order:
        ds, enc, rows = blocks[i]
        p = loads.index(min(loads))
        parts[p].append(i)
        loads[p] += len(rows) * MODEL2COST.get(enc, 1.0)
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
                   help="aliases or HF ids; see ALIASES")
    p.add_argument("--translators", nargs="+", default=["identity", "linear"],
                   help="one full grid per translator, in the order given")
    p.add_argument("--config-dir", default="src/configs",
                   help="directory scanned for already-covered rows")
    p.add_argument("--dedup-rows", action="store_true",
                   help="also drop individual rows that exist in another config")
    p.add_argument("--no-exclude", action="store_true",
                   help="emit the whole grid, ignoring existing configs")
    p.add_argument("--dry-run", action="store_true",
                   help="print the per-combination report only, write no CSV")
    p.add_argument("--split", type=int, default=1, metavar="N",
                   help="write N cost-balanced files instead of one CSV on stdout")
    p.add_argument("--out-prefix", default="src/configs/experiments_skip_grid",
                   help="path prefix for --split output; files are <prefix>_partK.csv")
    args = p.parse_args()

    if args.split < 1:
        p.error("--split must be at least 1")

    encoders = [ALIASES.get(e, e) for e in args.encoders]
    unknown = [e for e in encoders if e not in MODEL2NUM_LAYERS]
    if unknown:
        raise SystemExit(f"No block count for: {unknown}. Known: {sorted(MODEL2NUM_LAYERS)}")

    have = {} if args.no_exclude else existing_skips(args.config_dir, args.out_prefix)
    blocks, report = build(args.datasets, encoders, args.translators, have, args.dedup_rows)
    total = sum(len(rows) for _, _, rows in blocks)

    print(f"# {total} rows over {len(args.datasets)} datasets x {len(encoders)} encoders "
          f"x {len(args.translators)} translators", file=sys.stderr)
    for ds, enc, emitted, wanted, note in report:
        mark = "  " if emitted else "--"
        print(f"# {mark} {ds:16s} {enc:33s} {emitted:3d}/{wanted:3d} rows  ({note})",
              file=sys.stderr)

    if args.dry_run:
        return

    if args.split == 1:
        write_csv(sys.stdout, blocks)
        return

    parts, loads = partition(blocks, args.split)
    print("#", file=sys.stderr)

    # Drop parts left over from a previous split at a higher N. They are a stale copy of rows
    # this run has just reassigned, so leaving them behind means running some rows twice.
    written = {Path(f"{args.out_prefix}_part{k}.csv") for k in range(1, args.split + 1)}
    prefix = Path(args.out_prefix)
    for path in sorted(prefix.parent.glob(f"{prefix.name}_part*.csv")):
        if path not in written:
            path.unlink()
            print(f"# removed stale {path}", file=sys.stderr)

    for k, (part, load) in enumerate(zip(parts, loads), 1):
        path = Path(f"{args.out_prefix}_part{k}.csv")
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="") as f:
            write_csv(f, part)
        n_rows = sum(len(rows) for _, _, rows in part)
        print(f"# {path}: {n_rows} rows, {len(part)} combinations, cost {load:.0f}",
              file=sys.stderr)
        for ds, enc, rows in part:
            print(f"#     {ds:16s} {enc:33s} {len(rows):3d}", file=sys.stderr)
    print(f"# Give each part its own RESULTS_CSV_NAME when running them concurrently.",
          file=sys.stderr)


if __name__ == "__main__":
    main()
