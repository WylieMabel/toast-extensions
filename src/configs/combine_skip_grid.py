"""Merge the skip-grid shards and the pre-existing imagenet-1k sweeps into one results table.

The grid study is 6 encoders x 4 datasets x {no skip, one block, two adjacent blocks} x
{identity, linear}. It was run in two eras: three imagenet-1k combinations were already
covered by earlier sweeps (which is why make_skip_grid.py excludes them), and the remaining 21
were run as six shards. This joins the two into a single table.

    python src/configs/combine_skip_grid.py
    python src/configs/combine_skip_grid.py --all-seeds --out results/results_skip_grid_all.csv

WHAT IS TAKEN FROM WHERE
    The shards contribute every row they hold. The earlier sweeps are filtered down to the
    grid: they also carry the quadratic two-skip spacing group, so they are cut to rows whose
    skip is [], (i, i+1) or (i, i+2), with no sublayer or head edits and a translator of
    identity or linear. Everything else in those files is left alone.

SEEDS
    The shards ran seeds 1-3; dinov2-base on imagenet-1k carries 1-5. The merge keeps 1-3 by
    default so every cell of the grid is averaged over the same seeds -- an encoder silently
    averaged over five while its neighbours use three is not comparable, and comparing across
    encoders is the entire point of the table. --all-seeds keeps them.

DELTA_ACC
    Recomputed from each row's own baseline (same dataset, model and seed) rather than trusted
    from the source files, and any row whose stored value disagreed is reported. train_
    skipped_full.py historically matched the baseline on translator too, which silently gave
    every non-identity row delta_acc = 0.0; that bug is fixed, but these files span both eras
    and a merged table should not depend on which era a row came from.
"""

import argparse
import ast
from pathlib import Path

import pandas as pd

SHARD_GLOB = "results/results_skip_grid_part*.csv"

# (results file, dataset, encoder, block count) for the combinations make_skip_grid.py skipped
# because an earlier sweep already covered them.
PRIOR_SOURCES = [
    ("results/results_deitsmall_imagenet1k_spacing.csv",
     "imagenet-1k", "facebook/deit-small-patch16-224", 12),
    # dinov2-base's imagenet-1k sweep lives here, not under results/. Do not substitute
    # results_new.csv: it is not the authoritative source for this combination.
    ("block_distance/results_imgnet_dinov2.csv",
     "imagenet-1k", "facebook/dinov2-base", 12),
    ("results/results_vitlarge_imagenet1k_spacing.csv",
     "imagenet-1k", "google/vit-large-patch16-224", 24),
]

KEY = ["dataset", "model", "approx_layer", "mlp_linearize", "attn_linearize",
       "head_dict", "translator", "seed"]


def norm(value):
    """Compare spans by parsed value: the files disagree on whitespace, not content."""
    try:
        return str(ast.literal_eval(str(value)))
    except (ValueError, SyntaxError):
        return str(value)


def grid_spans(n_blocks):
    last = n_blocks - 1
    return {norm(s) for s in
            ["[]"]
            + [f"[({i}, {i + 1})]" for i in range(last)]
            + [f"[({i}, {i + 2})]" for i in range(last - 1)]}


def select_grid(df, dataset, model, n_blocks):
    """The grid rows of one (dataset, encoder) inside a possibly much larger results file."""
    sel = df[(df["dataset"] == dataset) & (df["model"] == model)]
    sel = sel[(sel["mlp_linearize"].map(norm) == "[]")
              & (sel["attn_linearize"].map(norm) == "[]")
              & (sel["head_dict"].map(norm) == "{}")]
    sel = sel[sel["approx_layer"].map(norm).isin(grid_spans(n_blocks))]
    return sel[sel["translator"].isin(["identity", "linear"])]


def recompute_delta(df):
    """Return df with delta_acc rebuilt from each (dataset, model, seed) baseline.

    The baseline is the unmodified encoder, so it is matched WITHOUT translator: an identity
    and a linear row over the same span share one baseline (nothing is bridged when nothing is
    skipped). Returns the frame and the number of rows whose stored value disagreed.
    """
    is_base = ((df["approx_layer"].map(norm) == "[]")
               & (df["mlp_linearize"].map(norm) == "[]")
               & (df["attn_linearize"].map(norm) == "[]")
               & (df["head_dict"].map(norm) == "{}"))

    base = (df[is_base]
            .drop_duplicates(subset=["dataset", "model", "seed"])
            .set_index(["dataset", "model", "seed"])["accuracy"])
    ref = df.set_index(["dataset", "model", "seed"]).index.map(base)

    out = df.copy()
    out["original_accuracy"] = ref.to_numpy()
    new_delta = out["original_accuracy"] - out["accuracy"]
    changed = int((~new_delta.round(9).eq(out["delta_acc"].round(9))
                   & new_delta.notna()).sum())
    out["delta_acc"] = new_delta
    return out, changed


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="results/results_skip_grid.csv")
    p.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    p.add_argument("--all-seeds", action="store_true",
                   help="keep every seed each source has, not just --seeds")
    args = p.parse_args()

    frames = []

    shards = sorted(Path().glob(SHARD_GLOB))
    if not shards:
        raise SystemExit(f"No shard results matched {SHARD_GLOB}")
    for path in shards:
        df = pd.read_csv(path)
        df["source_file"] = path.name
        frames.append(df)
        print(f"  {path.name}: {len(df)} rows")

    for path, dataset, model, n_blocks in PRIOR_SOURCES:
        df = pd.read_csv(path)
        sel = select_grid(df, dataset, model, n_blocks).copy()
        sel["source_file"] = Path(path).name
        frames.append(sel)
        print(f"  {Path(path).name}: {len(sel)} grid rows for {model} "
              f"(of {len(df)} in file)")

    combined = pd.concat(frames, ignore_index=True, sort=False)

    if not args.all_seeds:
        before = len(combined)
        combined = combined[combined["seed"].isin(args.seeds)]
        if before != len(combined):
            print(f"\n  Dropped {before - len(combined)} rows outside seeds {args.seeds}.")

    dupes = combined.duplicated(subset=KEY).sum()
    if dupes:
        print(f"  WARNING: {dupes} duplicate (config, seed) rows; keeping the first of each.")
        combined = combined.drop_duplicates(subset=KEY)

    combined, changed = recompute_delta(combined)
    if changed:
        print(f"  Recomputed delta_acc: {changed} row(s) disagreed with the stored value.")
    orphans = int(combined["original_accuracy"].isna().sum())
    if orphans:
        print(f"  WARNING: {orphans} row(s) have no baseline for their (dataset, model, seed).")

    combined = combined.sort_values(["dataset", "model", "translator", "approx_layer", "seed"],
                                    kind="stable")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(args.out, index=False)

    combos = combined.groupby(["dataset", "model"]).size()
    print(f"\n  {len(combined)} rows, {len(combos)} (dataset, encoder) combinations "
          f"-> {args.out}")
    for (ds, mdl), n in combos.items():
        print(f"    {ds:16s} {mdl:33s} {n:4d}")


if __name__ == "__main__":
    main()
