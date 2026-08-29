"""Step 3 of the skip-candidate predictor plan: for a fixed model, do accuracy-drop patterns
and CKA patterns look similar across datasets?

Two checks, both per model, comparing every dataset pair:

  accuracy-drop pattern   Spearman correlation of the single_drop[block] vector (from the
                          "skip each layer in turn" sweep, skip_count == 1 rows) across
                          datasets. If block 7 is safe to skip on cifar100, is it also safe
                          on imagenet-1k? Only 11-23 blocks per model, so raw r alone is not
                          enough to call -- each correlation is permutation-tested (shuffle
                          one vector, rebuild the null distribution of r, 20000 trials) to
                          get a p-value, since with this few points a "weak" r may just be
                          underpowered rather than genuinely zero.

  CKA pattern             Pearson correlation between the two datasets' flattened CKA
                          matrices (src/layers/outputs/*_cka.npy), plus the same comparison
                          restricted to the distance-1 diagonal (cka[i, i+1] for consecutive
                          blocks) since that's the value Step 1's regressions actually used.

Also compares Step 1's fitted contiguous-regression coefficients (coef_cka,
coef_block_drop_sum) across datasets for the same model, as a second, more direct read on
whether the same relationship holds across datasets rather than just the raw numbers.

    python block_distance/cross_dataset_consistency.py
"""

import argparse
import ast
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd


def dropped_blocks(spans):
    blocks = []
    for start, end in spans:
        blocks.extend(range(start + 1, end + 1))
    return blocks


def spearman(a, b):
    return pd.Series(a).rank().corr(pd.Series(b).rank())


def build_single_drop_table(distance_csv):
    df = pd.read_csv(distance_csv)
    df["skip_count"] = df["skip_count"].astype(int)
    df["spans"] = df["approx_layer_str"].map(ast.literal_eval)

    table = {}
    for (model, dataset), g in df.groupby(["model", "dataset"]):
        baseline_rows = g[g["skip_count"] == 0]
        if baseline_rows.empty:
            continue
        baseline = baseline_rows["accuracy_mean_linear"].iloc[0]
        single = g[g["skip_count"] == 1]
        drops = {}
        for _, row in single.iterrows():
            blocks = dropped_blocks(row["spans"])
            if len(blocks) == 1:
                drops[blocks[0]] = baseline - row["accuracy_mean_linear"]
        table[(model, dataset)] = drops
    return table


def permutation_p(v1, v2, r_obs, n_perm, rng):
    v2 = np.asarray(v2, dtype=float)
    null_rs = np.empty(n_perm)
    for i in range(n_perm):
        null_rs[i] = spearman(v1, rng.permutation(v2))
    return (np.abs(null_rs) >= abs(r_obs)).mean()


def accuracy_pattern_report(single_drop_table, n_perm, seed):
    rng = np.random.default_rng(seed)
    rows = []
    models = sorted({m for m, _ in single_drop_table})
    for model in models:
        datasets = sorted(d for m, d in single_drop_table if m == model)
        print(f"{model}:")
        for d1, d2 in combinations(datasets, 2):
            drops1, drops2 = single_drop_table[(model, d1)], single_drop_table[(model, d2)]
            blocks = sorted(set(drops1) & set(drops2))
            if len(blocks) < 3:
                continue
            v1 = [drops1[b] for b in blocks]
            v2 = [drops2[b] for b in blocks]
            r = spearman(v1, v2)
            p = permutation_p(v1, v2, r, n_perm, rng)
            sig = "significant" if p < 0.05 else "not significant"
            print(f"  {d1:14s} vs {d2:14s}  spearman r={r:+.3f}  p={p:.3f} ({sig}, n_blocks={len(blocks)})")
            rows.append({"model": model, "dataset_a": d1, "dataset_b": d2,
                         "spearman_r": r, "n_blocks": len(blocks), "perm_p": p, "significant": p < 0.05})
        print()
    return pd.DataFrame(rows)


def cka_pattern_report(cka_dir, models_datasets):
    rows = []
    outputs_dir = Path(cka_dir)
    by_model = {}
    for model, dataset in models_datasets:
        by_model.setdefault(model, set()).add(dataset)

    for model, datasets in sorted(by_model.items()):
        print(f"{model}:")
        cka_by_dataset = {}
        for dataset in sorted(datasets):
            f = outputs_dir / f"{model.replace('/', '_')}_{dataset}_cka.npy"
            if f.exists():
                cka_by_dataset[dataset] = np.load(f)
        for d1, d2 in combinations(sorted(cka_by_dataset), 2):
            m1, m2 = cka_by_dataset[d1], cka_by_dataset[d2]
            if m1.shape != m2.shape:
                continue
            full_r = np.corrcoef(m1.flatten(), m2.flatten())[0, 1]
            diag1 = np.diagonal(m1, offset=1)
            diag2 = np.diagonal(m2, offset=1)
            diag_r = spearman(diag1, diag2)
            print(f"  {d1:14s} vs {d2:14s}  full-matrix pearson r={full_r:+.3f}  "
                  f"distance-1 diagonal spearman r={diag_r:+.3f}")
            rows.append({"model": model, "dataset_a": d1, "dataset_b": d2,
                         "full_matrix_pearson_r": full_r, "distance1_diag_spearman_r": diag_r})
        print()
    return pd.DataFrame(rows)


def regression_coef_comparison(coefs_csv):
    coefs = pd.read_csv(coefs_csv)
    contiguous = coefs[(coefs["shape"] == "contiguous") & (~coefs["cka_screened"])]
    pivot = contiguous.pivot_table(index="model", columns="dataset",
                                    values=["coef_cka", "coef_block_drop_sum", "r2_loo_cv"])
    print("Step 1 contiguous-fit coefficients by model x dataset (full fits, not cka-screened):")
    print(pivot.round(3).to_string())
    print()
    return pivot


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--distance-csv", default="block_distance/distance_analysis.csv")
    p.add_argument("--cka-dir", default="src/layers/outputs")
    p.add_argument("--coefs", default="block_distance/skip_regression_coefs.csv")
    p.add_argument("--accuracy-out", default="block_distance/cross_dataset_accuracy_pattern.csv")
    p.add_argument("--cka-out", default="block_distance/cross_dataset_cka_pattern.csv")
    p.add_argument("--n-perm", type=int, default=20000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    single_drop_table = build_single_drop_table(args.distance_csv)

    print("=== Accuracy-drop pattern consistency across datasets (same model) ===\n")
    acc_df = accuracy_pattern_report(single_drop_table, args.n_perm, args.seed)
    acc_df.to_csv(args.accuracy_out, index=False)
    n_sig = acc_df["significant"].sum()
    n_sig_pos = ((acc_df["significant"]) & (acc_df["spearman_r"] > 0)).sum()
    n_sig_neg = ((acc_df["significant"]) & (acc_df["spearman_r"] < 0)).sum()
    print(f"{n_sig}/{len(acc_df)} dataset pairs significant at p<0.05 "
          f"({n_sig_pos} positive / {n_sig_neg} negative); "
          f"{len(acc_df) - n_sig} indistinguishable from chance at this sample size.\n")

    print("=== CKA pattern consistency across datasets (same model) ===\n")
    cka_df = cka_pattern_report(args.cka_dir, list(single_drop_table.keys()))
    cka_df.to_csv(args.cka_out, index=False)

    print("=== Step 1 regression coefficients across datasets (same model) ===\n")
    regression_coef_comparison(args.coefs)

    print(f"Saved -> {args.accuracy_out}, {args.cka_out}")


if __name__ == "__main__":
    main()
