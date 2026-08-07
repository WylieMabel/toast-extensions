"""Generate table-3-style ablations for any dataset/encoder combination.

Table 3 compares three ways of removing the same blocks, at matched parameter savings:
dropping whole blocks (with and without a translator), replacing just their MLP sub-blocks,
and replacing just their attention sub-blocks. Each configuration therefore expands to five
rows:

    skip=<spans>       skip_translator=identity     whole block, no bridge
    skip=<spans>       skip_translator=linear       whole block, linear bridge
    mlp_skip=<blocks>  mlp_mode=identity            MLP dropped
    mlp_skip=<blocks>  mlp_mode=linear              MLP linearised
    attn_skip=<blocks> attn_mode=identity           attention dropped

<blocks> is derived from <spans>, not specified separately: skip (a, b) removes blocks
a+1 .. b, so the sub-block rows always target exactly the blocks the whole-block row removes.
That is what makes the three arms comparable.

    # reproduce the original table exactly (used as the generator's own test)
    python src/configs/make_table3.py --datasets cifar100 --encoders deitsmall dinobase vitlarge

    # the medical version
    python src/configs/make_table3.py > src/configs/experiments_table3_medical.csv

SPAN SETS
    Taken verbatim from src/configs/experiments_table3.csv, which replicates the prior paper.
    rad-dino reuses DINOv2-base's spans: it is a DINOv2-base architecture (12 blocks, d=768)
    with medical pretraining, so the block indices transfer directly and the pair becomes a
    controlled comparison -- same architecture, same spans, different pretraining domain.

    Holding the spans fixed while changing the dataset is the experiment: do blocks that were
    redundant on natural images stay redundant on medical images?
"""

import argparse
import csv
import sys

# Short name -> HF id, mirroring layer_priority.MODEL_ALIASES.
ALIASES = {
    "deitsmall": "facebook/deit-small-patch16-224",
    "dinobase": "facebook/dinov2-base",
    "vitlarge": "google/vit-large-patch16-224",
    "raddino": "microsoft/rad-dino",
}

# Span sets per encoder, in the order the original table lists them.
SPAN_SETS = {
    "facebook/deit-small-patch16-224": [
        [(3, 4), (9, 11)],
        [(9, 11)],
        [(8, 9)],
        [(9, 10)],
    ],
    "facebook/dinov2-base": [
        [(0, 4)],
        [(0, 1), (2, 3), (4, 5)],
        [(0, 1), (2, 3)],
        [(0, 1)],
        [(2, 3)],
    ],
    "google/vit-large-patch16-224": [
        [(2, 4), (18, 23)],
        [(17, 23)],
        [(3, 4), (19, 23)],
        [(3, 4), (20, 23)],
        [(20, 23)],
        [(3, 4), (21, 23)],
        [(20, 22)],
        [(3, 4), (21, 22)],
        [(20, 21)],
        [(21, 22)],
    ],
}
# rad-dino is DINOv2-base's architecture, so it inherits DINOv2-base's spans.
SPAN_SETS["microsoft/rad-dino"] = SPAN_SETS["facebook/dinov2-base"]

COLUMNS = [
    "dataset", "encoder", "skip", "mlp_skip", "attn_skip",
    "head_dict", "skip_translator", "mlp_mode", "attn_mode", "fit_dataset",
]


def blocks_removed(spans):
    """Blocks a set of spans deletes: skip (a, b) keeps a and drops a+1 .. b."""
    out = set()
    for a, b in spans:
        out.update(range(a + 1, b + 1))
    return sorted(out)


def row(dataset, encoder, skip="[]", mlp_skip="[]", attn_skip="[]",
        skip_translator="identity", mlp_mode="identity", attn_mode="identity"):
    return {
        "dataset": dataset, "encoder": encoder, "skip": skip,
        "mlp_skip": mlp_skip, "attn_skip": attn_skip, "head_dict": "{}",
        "skip_translator": skip_translator, "mlp_mode": mlp_mode,
        "attn_mode": attn_mode, "fit_dataset": "",
    }


def build(dataset, encoder):
    # Baseline first: train_skipped_full.py resolves delta_acc against the all-empty row for
    # this dataset/model, and a missing one makes every delta NaN.
    rows = [row(dataset, encoder, mlp_mode="linear")]

    for spans in SPAN_SETS[encoder]:
        skip_str = "[" + ", ".join(f"({a}, {b})" for a, b in spans) + "]"
        blocks_str = "[" + ",".join(str(b) for b in blocks_removed(spans)) + "]"

        rows.append(row(dataset, encoder, skip=skip_str, skip_translator="identity"))
        rows.append(row(dataset, encoder, skip=skip_str, skip_translator="linear"))
        rows.append(row(dataset, encoder, mlp_skip=blocks_str, mlp_mode="identity"))
        rows.append(row(dataset, encoder, mlp_skip=blocks_str, mlp_mode="linear"))
        rows.append(row(dataset, encoder, attn_skip=blocks_str, attn_mode="identity"))

    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--datasets", nargs="+", default=["dermamnist", "pneumoniamnist"])
    p.add_argument("--encoders", nargs="+", default=["dinobase", "raddino", "deitsmall"],
                   help="aliases or HF ids; see ALIASES")
    args = p.parse_args()

    encoders = [ALIASES.get(e, e) for e in args.encoders]
    unknown = [e for e in encoders if e not in SPAN_SETS]
    if unknown:
        raise SystemExit(f"No span set for: {unknown}. Known: {sorted(SPAN_SETS)}")

    rows = []
    # Dataset-major so each dataset's block of rows is contiguous and self-contained: a run
    # that is cut short still yields complete results for the datasets it reached.
    for ds in args.datasets:
        for enc in encoders:
            rows.extend(build(ds, enc))

    w = csv.DictWriter(sys.stdout, fieldnames=COLUMNS, quoting=csv.QUOTE_NONNUMERIC)
    w.writeheader()
    w.writerows(rows)

    print(f"# {len(rows)} rows = {len(args.datasets)} datasets x {len(encoders)} encoders",
          file=sys.stderr)
    for enc in encoders:
        n = 1 + 5 * len(SPAN_SETS[enc])
        print(f"#   {enc}: {n} rows/dataset ({len(SPAN_SETS[enc])} configs x 5 + baseline)",
              file=sys.stderr)


if __name__ == "__main__":
    main()
