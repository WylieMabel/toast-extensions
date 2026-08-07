"""Generate the low-rank translator sweep config CSV.

Emits rows crossing {lowrank, rrr, lora} x rank against a few skip spans, plus the identity
floor and the full-rank linear control. Everything is expressed in the normal config schema,
so the sweep runs through run_pipeline_row_by_row.sh with no special-casing.

    python src/configs/make_lowrank_sweep.py > src/configs/experiments_lowrank.csv

SPAN CHOICE
    Spans are picked from block_distance/results_imgnet_dinov2.csv by how much work the
    linear translator is actually doing (its accuracy gain over identity). The hypothesis
    under test is that the rank a translator needs scales with that gain, so the sweep needs
    spans at both ends:

        (5, 6)  identity 0.294 -> linear 0.731   gain 0.437   translator does the heavy lifting
        (2, 4)  identity 0.097 -> linear 0.687   gain 0.590   hardest span, 2 blocks dropped
        (0, 1)  identity 0.654 -> linear 0.738   gain 0.084   control: rank should barely matter

    (dinov2-base / imagenet-1k, no-skip baseline 0.7431)

ROW ORDER
    The all-empty baseline row comes first because train_skipped_full.py looks the baseline up
    out of the partially-written results CSV -- if it is missing, original_accuracy silently
    becomes 0.0 and every delta_acc is forced to 0.0.

    After that, rows are grouped span-by-span rather than interleaved. Results are written
    incrementally, so if the job runs out of time the completed prefix is still a complete
    story for the first span instead of a scattering of holes across all three.
"""

import argparse
import csv
import sys

# Kept in step with lowrank_translator.DEFAULT_RANKS / METHOD2CLASS by hand rather than
# imported, so this generator stays runnable without torch or latentis installed -- the same
# reason src/toast/scripts/get_embed_dir.py inlines its helper.
DEFAULT_RANKS = (8, 16, 32, 64, 128, 256)
METHODS = ("lowrank", "rrr", "lora")

COLUMNS = [
    "dataset", "encoder", "skip", "mlp_skip", "attn_skip",
    "head_dict", "skip_translator", "mlp_mode", "attn_mode",
]

# Spans ordered most-informative first (see module docstring).
IMAGENET_SPANS = ["[(5, 6)]", "[(2, 4)]", "[(0, 1)]"]

# rad-dino has no published spans to inherit; these are the adjacent-block spans that
# layer_priority ranks lowest-influence on the medical sets. Re-derive from the block scores
# once layer_priority has run on the new datasets.
MEDICAL_SPANS = ["[(9, 10)]", "[(8, 10)]"]


def row(dataset, encoder, skip, translator):
    return {
        "dataset": dataset, "encoder": encoder, "skip": skip,
        "mlp_skip": "[]", "attn_skip": "[]", "head_dict": "{}",
        "skip_translator": translator, "mlp_mode": "identity", "attn_mode": "identity",
    }


def build(dataset, encoder, spans, ranks, methods):
    rows = [row(dataset, encoder, "[]", "identity")]  # baseline MUST be first
    for skip in spans:
        # Floor and full-rank control first, so each span's block is interpretable alone.
        rows.append(row(dataset, encoder, skip, "identity"))
        rows.append(row(dataset, encoder, skip, "linear"))
        for method in methods:
            for r in ranks:
                rows.append(row(dataset, encoder, skip, f"{method}_{r}"))
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="imagenet-1k")
    p.add_argument("--encoder", default="facebook/dinov2-base")
    p.add_argument("--spans", nargs="*", default=None,
                   help="Skip spans as literal strings, e.g. '[(5, 6)]'. Defaults by dataset.")
    p.add_argument("--ranks", nargs="*", type=int, default=list(DEFAULT_RANKS))
    p.add_argument("--methods", nargs="*", default=list(METHODS), choices=list(METHODS))
    args = p.parse_args()

    spans = args.spans
    if spans is None:
        spans = MEDICAL_SPANS if "mnist" in args.dataset else IMAGENET_SPANS

    rows = build(args.dataset, args.encoder, spans, args.ranks, args.methods)

    writer = csv.DictWriter(sys.stdout, fieldnames=COLUMNS, quoting=csv.QUOTE_NONNUMERIC)
    writer.writeheader()
    writer.writerows(rows)

    n_per_span = 2 + len(args.methods) * len(args.ranks)
    print(
        f"# {len(rows)} rows: 1 baseline + {len(spans)} spans x {n_per_span} "
        f"({len(args.methods)} methods x {len(args.ranks)} ranks, + identity + linear)",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
