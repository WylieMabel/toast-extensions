"""Generate a skip-spacing sweep config CSV for a given encoder.

Reproduces the structure of experiments_dinobase_imagenet1k_spacing.csv, parameterised by the
encoder's block count so the same study can be run at other depths.

    python src/configs/make_spacing_sweep.py facebook/deit-small-patch16-224 \
        > src/configs/experiments_deitsmall_imagenet1k_spacing.csv

WHAT THE SWEEP VARIES
    Three groups, emitted once per translator (identity first, then the fitted one) and led by
    the all-empty baseline so delta_acc has a reference:

      1. position   -- (i, i+1) for every i: which single block is cheapest to drop
      2. width      -- (i, i+2) for every i: cost of dropping two adjacent blocks at once
      3. spacing    -- two width-1 skips separated by a gap g, for every g and every start.
                       Isolates whether two drops interfere when close together.

    Row order matches the dinov2-base file: all identity rows, then all translator rows, so a
    run can be cut short after the identity half and still be self-consistent.

SIZE
    Rows grow quadratically with depth (the spacing group is every ordered pair of skips):
    12 blocks -> 134 rows, 24 blocks -> 554. Use --stride to subsample start positions and
    gaps when the full grid is too expensive.
"""

import argparse
import csv
import sys

# Mirrors MODEL2NUM_LAYERS in src/toast/utils/dictionaries.py, which is the source of truth --
# copied rather than imported so this generator stays runnable without the torch/latentis stack.
# Anything not listed here needs --blocks.
MODEL2NUM_LAYERS = {
    "WinKawaks/vit-small-patch16-224": 12,
    "WinKawaks/vit-tiny-patch16-224": 12,
    "facebook/deit-small-patch16-224": 12,
    "facebook/deit-base-patch16-224": 12,
    "google/vit-base-patch16-224": 12,
    "google/vit-large-patch16-224": 24,
    "facebook/dinov2-small": 12,
    "facebook/dinov2-base": 12,
    "microsoft/rad-dino": 12,
    "openai/clip-vit-base-patch32": 12,
    "timm/vit_tiny_patch16_224.augreg_in21k_ft_in1k": 12,
    "timm/vit_small_patch16_224.augreg_in21k_ft_in1k": 12,
    "timm/vit_large_patch16_224.augreg_in21k_ft_in1k": 24,
    "timm/deit_base_patch16_224.fb_in1k": 12,
    "answerdotai/ModernBERT-base": 22,
}

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


def spans(n_blocks, stride):
    """The skip specs of the sweep, in file order, as strings."""
    last = n_blocks - 1  # highest block index a skip may land on
    out = ["[]"]

    # position: one block dropped, slid along the depth
    out += [f"[({i}, {i + 1})]" for i in range(0, last, stride)]

    # width: two adjacent blocks dropped as a single span
    out += [f"[({i}, {i + 2})]" for i in range(0, last - 1, stride)]

    # spacing: two width-1 skips, gap g apart, slid along the depth
    for gap in range(2, last, stride):
        for i in range(0, last - gap, stride):
            out.append(f"[({i}, {i + 1}), ({i + gap}, {i + gap + 1})]")

    return out


def build(dataset, encoder, n_blocks, translators, stride):
    return [
        row(dataset, encoder, skip, translator)
        for translator in translators
        for skip in spans(n_blocks, stride)
    ]


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("encoder", help="HuggingFace encoder id, e.g. google/vit-large-patch16-224")
    p.add_argument("--dataset", default="imagenet-1k")
    p.add_argument("--blocks", type=int, default=None,
                   help="block count; defaults to the encoder's entry in dictionaries.py")
    p.add_argument("--translators", nargs="+", default=["identity", "linear"],
                   help="one full sweep per translator, in the order given")
    p.add_argument("--stride", type=int, default=1,
                   help="subsample start positions and gaps; 1 is the full grid")
    args = p.parse_args()

    n_blocks = args.blocks
    if n_blocks is None:
        if args.encoder not in MODEL2NUM_LAYERS:
            p.error(f"{args.encoder} has no block count in dictionaries.py; pass --blocks")
        n_blocks = MODEL2NUM_LAYERS[args.encoder]

    rows = build(args.dataset, args.encoder, n_blocks, args.translators, args.stride)
    w = csv.DictWriter(sys.stdout, fieldnames=COLUMNS, quoting=csv.QUOTE_NONNUMERIC)
    w.writeheader()
    w.writerows(rows)

    print(f"# {args.encoder}: {n_blocks} blocks, {len(rows)} rows "
          f"({len(rows) // len(args.translators)} per translator)", file=sys.stderr)


if __name__ == "__main__":
    main()
