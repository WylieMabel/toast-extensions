"""Generate the alternating (drop-one, keep-one) skip grid inside each model's block window.

The interleaved counterpart to make_window_grid.py, which sweeps contiguous spans. Same windows,
same datasets, same linear translator -- the only thing that changes is that the dropped blocks
are spread out with a surviving block between each pair:

    [(5, 10)]                 drops 6,7,8,9,10   one translator, nothing in between
    [(5, 6), (7, 8), (9, 10)] drops 6,8,10       three translators, blocks 7 and 9 survive

    python src/configs/make_alternating_window_grid.py \
        > src/configs/experiments_alternating_window_grid.csv

    # the 4-way split the window grid uses, so the two studies run the same shape of job
    python src/configs/make_alternating_window_grid.py --split 4 \
        --out-prefix src/configs/experiments_alternating_window_grid

    python src/configs/make_alternating_window_grid.py --dry-run

WHY -- THE RECOVERY HYPOTHESIS
    A surviving block between two drops sees a *translated* representation and may pull it back
    toward the true manifold, so n scattered drops could cost less than n contiguous ones. The
    counter-pressure: every translator is fit on clean precomputed embeddings
    (module.fit_translators), so at inference each one after the first receives input that is
    already perturbed, and the errors may compound instead of cancelling. The sweep is what
    decides it.

    docs/cka_skip_distance_findings.md leans toward interleaving: skip *distance* is the
    strongest predictor of damage (+0.739 vs identity drop), and distance-1 linear spans lose
    ~1-1.4pp where distance-2 loses 5.7pp. Three distance-1 hops may well beat one distance-3
    span at the same number of blocks removed -- at the cost of three translator matrices
    instead of one, which is the trade this grid measures.

WHAT IS EMITTED
    Every (start, n_drops) pattern that fits in a window: n_drops width-1 spans at stride 2,
    starting from each legal position. A pattern needs start + 2*n_drops - 1 <= hi, so a window
    of k endpoints admits n_drops up to k//2, with k - 2*n_drops + 1 starts at each size.

        n_drops=2, window 6-23  ->  [(6, 7), (8, 9)], [(7, 8), (9, 10)], ... 15 patterns
        n_drops=9, window 6-23  ->  [(6, 7), (8, 9), ..., (22, 23)], 1 pattern -- the whole
                                    window consumed, 9 of 24 blocks gone, 9 translators

    Rows are ordered by n_drops ascending, then by start: the question is where interleaving
    breaks down as more blocks come out, so a run cut short still answers it completely for
    every drop count it reached.

    n_drops=1 is deliberately excluded -- a single width-1 span is a contiguous span, and every
    one of them is already in experiments_window_grid.csv. dinov2-base drops out entirely for
    the same reason: its window (8-10, k=3) admits no pattern with two drops.

    Per dataset: 64 vit-large, 6 vit-base, 2 deit-small, 1 rad-dino, 6 deit-base = 79 patterns,
    so 252 rows over the three datasets including the 15 baselines.

COMPARING AGAINST THE CONTIGUOUS GRID
    The controls live in experiments_window_grid.csv and do not need re-running. For a pattern
    starting at `a` with n drops, the two matched contiguous rows there are:

        same blocks removed   [(a, a + n)]         n dropped, 1 translator
        same extent covered   [(a, a + 2n - 1)]    2n - 1 dropped, 1 translator

    Both configs share the (dataset, encoder) baselines, so delta_acc is on the same scale.

Everything else -- window semantics, baseline placement, overlap reporting, --split cost
balancing -- is inherited from make_window_grid.py; see its docstring.
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

DEFAULT_ENCODERS = list(WINDOWS)
DEFAULT_DATASETS = ["imagenet-1k", "pneumoniamnist", "cifar100"]
DEFAULT_OUT_PREFIX = "src/configs/experiments_alternating_window_grid"


def patterns(windows, n_blocks, min_drops, stride):
    """Alternating skip specs inside each window, as strings, fewest drops first.

    Each pattern is n_drops width-1 spans spaced `stride` apart, so stride 2 leaves exactly one
    surviving block between consecutive drops. The last dropped block is start + stride*(n-1) + 1
    and must land inside the window, which is what bounds both n_drops and start.
    """
    out = []
    for lo, hi in windows:
        if not 0 <= lo < hi <= n_blocks - 1:
            raise SystemExit(
                f"Window ({lo}, {hi}) is not a legal endpoint range for a {n_blocks}-block "
                f"encoder: need 0 <= lo < hi <= {n_blocks - 1}"
            )
        n_drops = min_drops
        while True:
            reach = stride * (n_drops - 1) + 1  # start -> last dropped block
            starts = range(lo, hi - reach + 1)
            if not starts:
                break
            for start in starts:
                spans = [(start + stride * i, start + stride * i + 1) for i in range(n_drops)]
                out.append(str(spans))
            n_drops += 1
    return out


def build(datasets, encoders, translator, have, dedup_rows, min_drops, stride):
    """One block of rows per (dataset, encoder), plus a report; mirrors make_window_grid.build.

    Encoders whose windows admit no pattern are dropped rather than emitted as a lone baseline.
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

            rows = [row(ds, enc, "[]", "identity")]
            rows += [row(ds, enc, s, translator) for s in emit]
            blocks.append((ds, enc, rows))
            report.append((ds, enc, len(rows), len(grid),
                           f"{len(dup)} already present" if dup else ""))
    return blocks, report


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
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
        p.error("--stride must be at least 2; stride 1 makes consecutive spans touch, which is "
                "a chain of translators with no surviving block between them")
    if args.min_drops < 1:
        p.error("--min-drops must be at least 1")

    encoders = [ALIASES.get(e, e) for e in args.encoders]
    unknown = [e for e in encoders if e not in WINDOWS]
    if unknown:
        raise SystemExit(f"No window for: {unknown}. Known: {sorted(WINDOWS)}")

    # existing_skips also treats make_window_grid's own output as "own" and skips it. Harmless
    # here: that file holds single spans only, and --min-drops 2 never emits one, so there is
    # nothing of ours it could have covered.
    have = {} if args.no_exclude else existing_skips(args.config_dir, args.out_prefix)
    blocks, report = build(args.datasets, encoders, args.translator, have, args.dedup_rows,
                           args.min_drops, args.stride)
    total = sum(len(rows) for _, _, rows in blocks)

    print(f"# {total} rows over {len(args.datasets)} datasets x {len(encoders)} encoders "
          f"(stride {args.stride}, >={args.min_drops} drops, {args.translator} translator, "
          f"{len(blocks)} baselines)", file=sys.stderr)
    for ds, enc, emitted, wanted, note in report:
        windows = " | ".join(f"{lo}-{hi}" for lo, hi in WINDOWS[enc])
        suffix = f"  [{note}]" if note else ""
        print(f"#   {ds:16s} {enc:34s} window {windows:12s} "
              f"{emitted:4d} rows / {wanted} patterns{suffix}", file=sys.stderr)

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
