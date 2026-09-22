"""Generate the medical-domain skip grid: baseline + contiguous skip spans, one CSV per
(model group, dataset) combination, each in its own subfolder under src/configs/health/.

Same row shape and span convention as make_skip_grid.py (skip (a, b) keeps block a and drops
a+1..b, both 0-indexed layer-output keys, last legal skip of length L is (n-1-L, n-1) so the
final block is never dropped) -- extended to an arbitrary max span length per group, since the
two model families here use different depths and different max lengths:

    Group 1 (12-layer dino family): dinov2-base, rad-dino, rad-dino-maira-2 -- max length 2
    Group 2 (24-layer ViT-L family): vitlarge, phikon-v2                    -- max length 4

    python src/configs/make_health_skip_grid.py

Only the "linear" translator is emitted (matching the baseline debug configs already run for
these models), not identity+linear like make_skip_grid.py's grid -- there was no request here
to compare translators, only skip lengths. Pass --translators identity linear for that.

Writes src/configs/health/<group>_<dataset>/experiments.csv, one file per combination:
6 files total (2 groups x 3 datasets each).

Pass --split-by-model to ALSO write one file per (group, dataset, encoder) in the same
subfolder, named experiments_<model_alias>.csv -- lets each model in a group run as its own
concurrent sbatch job instead of sharing one sequential file. Useful when a group's per-row
cost is dominated by test-set-size x num-configs (every skip span re-runs a full, uncapped
pass over the test split) and splitting by dataset alone isn't enough parallelism -- this was
needed for group2 (vitlarge/phikon-v2, 87 configs/model over large test splits) after
NUM_WORKERS/BATCH_SIZE alone weren't enough to bring patchcamelyon/tcga_ut down to group1's
wall-clock. Does not replace the combined experiments.csv; both exist side by side.
"""

import argparse
import csv
from pathlib import Path

MODEL2NUM_LAYERS = {
    "facebook/dinov2-base": 12,
    "microsoft/rad-dino": 12,
    "microsoft/rad-dino-maira-2": 12,
    "google/vit-large-patch16-224": 24,
    "owkin/phikon-v2": 24,
}

MODEL2ALIAS = {
    "facebook/dinov2-base": "dinov2base",
    "microsoft/rad-dino": "raddino",
    "microsoft/rad-dino-maira-2": "raddinomaira2",
    "google/vit-large-patch16-224": "vitlarge",
    "owkin/phikon-v2": "phikonv2",
}

GROUPS = {
    "group1": {
        "encoders": ["facebook/dinov2-base", "microsoft/rad-dino", "microsoft/rad-dino-maira-2"],
        "datasets": ["imagenet-1k", "chestxray_pneumonia", "chestxray14"],
        "max_span_length": 2,
    },
    "group2": {
        "encoders": ["google/vit-large-patch16-224", "owkin/phikon-v2"],
        "datasets": ["imagenet-1k", "patchcamelyon", "tcga_ut"],
        "max_span_length": 4,
    },
}

COLUMNS = [
    "dataset", "encoder", "skip", "mlp_skip", "attn_skip",
    "head_dict", "skip_translator", "mlp_mode", "attn_mode", "fit_dataset",
]

OUT_ROOT = Path("src/configs/health")


def row(dataset, encoder, skip, translator):
    return {
        "dataset": dataset, "encoder": encoder, "skip": skip,
        "mlp_skip": "[]", "attn_skip": "[]", "head_dict": "{}",
        "skip_translator": translator, "mlp_mode": "identity", "attn_mode": "identity",
        "fit_dataset": "",
    }


def spans_by_length(n_blocks, max_length):
    """{length: [span strings]} for length 1..max_length, never touching the last block."""
    last = n_blocks - 1  # highest block index a skip may land on
    return {
        length: [f"[({i}, {i + length})]" for i in range(0, last - length + 1)]
        for length in range(1, max_length + 1)
    }


def spans(n_blocks, max_length):
    """Baseline + every contiguous span of length 1..max_length, in file order."""
    out = ["[]"]
    by_length = spans_by_length(n_blocks, max_length)
    for length in range(1, max_length + 1):
        out += by_length[length]
    return out


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, quoting=csv.QUOTE_NONNUMERIC)
        w.writeheader()
        w.writerows(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--translators", nargs="+", default=["linear"])
    p.add_argument("--out-root", default=str(OUT_ROOT))
    p.add_argument("--split-by-model", action="store_true",
                   help="also write one experiments_<alias>.csv per encoder in each subfolder")
    p.add_argument("--length-groups", type=int, default=0, metavar="N",
                   help="split each model's spans into N contiguous length-range shards "
                        "(e.g. N=2 at max_span_length=4 -> lengths {1,2} and {3,4}). ONLY "
                        "the first group's file gets the baseline row -- cfg_embedding_dir() "
                        "(utils.py) hashes the baseline config the same way regardless of "
                        "which file it came from and ignores seed entirely, so two shards "
                        "for the same model both carrying their own 'baseline' copy would "
                        "run_pipeline_row_by_row.sh-concurrently race on the exact same "
                        "on-disk embedding path (one job's save/delete stomping another's "
                        "mid-write). Groups after the first have no baseline of their own, so "
                        "their delta_acc comes back NaN until backfilled from group 1's "
                        "results (same accuracy, same dataset/model/seed) in a merge step.")
    args = p.parse_args()
    out_root = Path(args.out_root)

    print(f"{'group':8s} {'dataset':22s} {'rows':>6s}  path")
    total = 0
    for group_name, cfg in GROUPS.items():
        for ds in cfg["datasets"]:
            per_model_rows = {}
            per_model_by_length = {}
            for enc in cfg["encoders"]:
                n_blocks = MODEL2NUM_LAYERS[enc]
                grid = spans(n_blocks, cfg["max_span_length"])
                per_model_rows[enc] = [
                    row(ds, enc, s, t) for t in args.translators for s in grid
                ]
                per_model_by_length[enc] = spans_by_length(n_blocks, cfg["max_span_length"])

            folder = out_root / f"{group_name}_{ds}"
            combined_rows = [r for rows in per_model_rows.values() for r in rows]
            path = folder / "experiments.csv"
            write_csv(path, combined_rows)
            print(f"{group_name:8s} {ds:22s} {len(combined_rows):6d}  {path}")
            total += len(combined_rows)

            if args.split_by_model:
                for enc, rows in per_model_rows.items():
                    model_path = folder / f"experiments_{MODEL2ALIAS[enc]}.csv"
                    write_csv(model_path, rows)
                    print(f"{'':8s} {'':22s} {len(rows):6d}  {model_path}")

            if args.length_groups > 0:
                max_len = cfg["max_span_length"]
                n_groups = min(args.length_groups, max_len)
                # Contiguous, near-even chunks of the length range, e.g. max_len=4, n=2 ->
                # [[1,2],[3,4]]; max_len=4, n=3 -> [[1,2],[3],[4]].
                lengths = list(range(1, max_len + 1))
                chunk_size, remainder = divmod(len(lengths), n_groups)
                chunks, start = [], 0
                for i in range(n_groups):
                    size = chunk_size + (1 if i < remainder else 0)
                    chunks.append(lengths[start:start + size])
                    start += size

                for enc, by_length in per_model_by_length.items():
                    alias = MODEL2ALIAS[enc]
                    for group_idx, chunk_lengths in enumerate(chunks):
                        group_spans = [s for l in chunk_lengths for s in by_length[l]]
                        # Only the first chunk gets the baseline row -- see --length-groups
                        # help text for why duplicating it across concurrently-run shards is
                        # unsafe (shared, seed-blind embedding cache path).
                        prefix = ["[]"] if group_idx == 0 else []
                        group_rows = [
                            row(ds, enc, s, t)
                            for t in args.translators
                            for s in prefix + group_spans
                        ]
                        label = f"len{chunk_lengths[0]}-{chunk_lengths[-1]}" if len(chunk_lengths) > 1 \
                            else f"len{chunk_lengths[0]}"
                        group_path = folder / f"experiments_{alias}_{label}.csv"
                        write_csv(group_path, group_rows)
                        note = "" if group_idx == 0 else "  (no baseline -- delta_acc will be NaN until backfilled)"
                        print(f"{'':8s} {'':22s} {len(group_rows):6d}  {group_path}{note}")

    print(f"\n{total} rows across {sum(len(c['datasets']) for c in GROUPS.values())} "
          f"combined files"
          + (" (plus per-model splits)" if args.split_by_model else "") + ".")


if __name__ == "__main__":
    main()
