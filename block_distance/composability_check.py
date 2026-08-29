"""Step 0 of the skip-candidate predictor plan: is skip damage additive?

Tests whether the accuracy drop of a 2-block skip can be predicted by summing the
accuracy drops of the individual single-block skips it's made of, using the
single-block ("skip each layer in turn") and 2-block sweeps already in
distance_analysis.csv. Checked separately for the two shapes skip_count=2 takes:

    contiguous : one span dropping two adjacent blocks, e.g. [(0, 2)]
    disjoint   : two separate single-block skips, e.g. [(0, 1), (2, 3)]

since a contiguous span shares intermediate activations in a way a disjoint pair
doesn't, and additivity may hold for one shape and not the other.

Tuple convention (SkipModel, see recommend_runs.py): (skip_from, skip_to) keeps
block skip_from and drops blocks skip_from+1 .. skip_to.

    python block_distance/composability_check.py
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


def r_squared(actual, predicted):
    ss_res = ((actual - predicted) ** 2).sum()
    ss_tot = ((actual - actual.mean()) ** 2).sum()
    return 1 - ss_res / ss_tot if ss_tot else float("nan")


def analyze_group(label, rows, single_drop, baseline):
    if len(rows) == 0:
        print(f"  {label}: no rows")
        return

    actual_drop, predicted_drop = [], []
    for _, row in rows.iterrows():
        blocks = dropped_blocks(row["approx_layer_str"])
        if len(blocks) != 2 or not all(b in single_drop for b in blocks):
            continue
        predicted_drop.append(single_drop[blocks[0]] + single_drop[blocks[1]])
        actual_drop.append(baseline - row["accuracy_mean_linear"])

    if len(actual_drop) < 2:
        print(f"  {label}: n={len(actual_drop)}, not enough matched rows")
        return

    actual_s = pd.Series(actual_drop)
    predicted_s = pd.Series(predicted_drop)
    print(
        f"  {label}: n={len(actual_s)}  "
        f"pearson r={actual_s.corr(predicted_s):+.3f}  "
        f"spearman r={spearman(actual_s, predicted_s):+.3f}  "
        f"R^2={r_squared(actual_s, predicted_s):+.3f}"
    )


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--distance-csv", default="block_distance/distance_analysis.csv")
    args = p.parse_args()

    df = pd.read_csv(args.distance_csv)
    df["skip_count"] = df["skip_count"].astype(int)

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

        two = g[g["skip_count"] == 2]
        contiguous = two[two["skip_distance"] == 1]
        disjoint = two[two["skip_distance"] > 1]

        print(f"{model} / {dataset}  (baseline={baseline:.4f}, n_single={len(single_drop)})")
        analyze_group("contiguous 2-block spans", contiguous, single_drop, baseline)
        analyze_group("disjoint single-block pairs", disjoint, single_drop, baseline)
        print()


if __name__ == "__main__":
    main()
