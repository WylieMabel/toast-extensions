"""Cheaper alternative to the sum-of-drops baseline (composability_check.py): instead of summing
the two single-block accuracy-drop *magnitudes*, sum their *ranks* in the "skip each layer in
turn" sweep (rank 1 = safest single block, smallest drop) and correlate that rank-sum against the
true ranking of 2-block spans by actual accuracy drop.

This only needs the ORDER of the single-block sweep, not calibrated drop magnitudes -- a dumber,
cheaper signal than composability_check.py's block_drop_sum. If it correlates almost as well,
that undercuts the case for the fitted regression's extra machinery (CKA, kept-separate features
in fit_skip_regression.py); if it does notably worse, that's evidence the regression's features
earn their keep.

Checked separately for the two skip_count=2 shapes (see composability_check.py for why):
    contiguous : one span dropping two adjacent blocks, e.g. [(0, 2)]
    disjoint   : two separate single-block skips, e.g. [(0, 1), (2, 3)]

    python block_distance/rank_sum_baseline.py
"""

import argparse
import ast

import pandas as pd


def dropped_blocks(approx_layer_str):
    spans = ast.literal_eval(approx_layer_str)
    blocks = []
    for start, end in spans:
        blocks.extend(range(start + 1, end + 1))
    return blocks


def spearman(a, b):
    return a.rank().corr(b.rank())


def analyze_group(label, rows, single_rank, baseline, model, dataset, results):
    if len(rows) == 0:
        print(f"  {label}: no rows")
        return

    actual_drop, rank_sum = [], []
    for _, row in rows.iterrows():
        blocks = dropped_blocks(row["approx_layer_str"])
        if len(blocks) != 2 or not all(b in single_rank for b in blocks):
            continue
        rank_sum.append(single_rank[blocks[0]] + single_rank[blocks[1]])
        actual_drop.append(baseline - row["accuracy_mean_linear"])

    if len(actual_drop) < 2:
        print(f"  {label}: n={len(actual_drop)}, not enough matched rows")
        return

    actual_s = pd.Series(actual_drop)
    rank_sum_s = pd.Series(rank_sum)
    spearman_r = spearman(actual_s, rank_sum_s)
    print(f"  {label}: n={len(actual_s)}  spearman r(rank_sum, actual_drop)={spearman_r:+.3f}")
    if results is not None:
        results.append({
            "model": model, "dataset": dataset, "shape": label, "n_spans": len(actual_s),
            "spearman_r": spearman_r,
        })


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--distance-csv", default="block_distance/distance_analysis.csv")
    p.add_argument("--out", default=None, help="Optional CSV path to save per-(model,dataset,shape) correlation stats")
    args = p.parse_args()

    df = pd.read_csv(args.distance_csv)
    df["skip_count"] = df["skip_count"].astype(int)

    results = [] if args.out else None
    for (model, dataset), g in df.groupby(["model", "dataset"]):
        baseline_rows = g[g["skip_count"] == 0]
        if baseline_rows.empty:
            continue
        baseline = baseline_rows["accuracy_mean_linear"].iloc[0]

        single = g[g["skip_count"] == 1]
        single_drop = {}
        for _, row in single.iterrows():
            blocks = dropped_blocks(row["approx_layer_str"])
            if len(blocks) == 1:
                single_drop[blocks[0]] = baseline - row["accuracy_mean_linear"]

        # rank 1 = safest (smallest drop)
        ordered_blocks = sorted(single_drop, key=lambda b: single_drop[b])
        single_rank = {b: i + 1 for i, b in enumerate(ordered_blocks)}

        two = g[g["skip_count"] == 2]
        contiguous = two[two["skip_distance"] == 1]
        disjoint = two[two["skip_distance"] > 1]

        print(f"{model} / {dataset}  (baseline={baseline:.4f}, n_single={len(single_drop)})")
        analyze_group("contiguous 2-block spans", contiguous, single_rank, baseline, model, dataset, results)
        analyze_group("disjoint single-block pairs", disjoint, single_rank, baseline, model, dataset, results)
        print()

    if args.out:
        out_df = pd.DataFrame(results)
        out_df.to_csv(args.out, index=False)
        print(f"Saved {len(out_df)} rows -> {args.out}")


if __name__ == "__main__":
    main()
