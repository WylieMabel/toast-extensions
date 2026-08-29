"""Step 1 of the skip-candidate predictor plan: per (model, dataset) linear regression
predicting the accuracy drop of a 2-block skip from cheap features, using only single-block
skips as the source of truth. Fit as two separate regressions per (model, dataset), since
Step 0 (composability_check.py) found the two shapes compose very differently:

  contiguous (one span dropping two adjacent blocks, e.g. [(0, 2)])
      features: cka             CKA between the hidden states the span bridges
                block_drop_sum  single_drop[a] + single_drop[a+1] for the two blocks in the span

  disjoint (two separate single-block skips, e.g. [(0, 1), (2, 3)])
      features: cka_1, cka_2    CKA of each individual single-block skip, kept SEPARATE
                                 (not summed -- the two skips are distinct bridges, and summing
                                 would throw away which one is more responsible for the damage)
                block_drop_sum  single_drop[a] + single_drop[c] (summing this one is justified:
                                 Step 0 found disjoint-pair damage is close to additive,
                                 R^2 0.87-0.998)

single_drop comes from the "skip each layer in turn" sweep (skip_count == 1 rows). Scope is
deliberately narrow: ONLY 2-block-drop configs (skip_count == 2), no longer spans.

CKA-boundary screen (contiguous only): investigating why vit-large's contiguous fit failed
out-of-sample found its 22 spans split cleanly into 16 near-perfect-CKA spans (drop 0.01-0.05,
well-behaved) and 6 spans crossing into CKA <= 0.88 (drop up to 0.71, a floor/saturation effect
a 2-feature linear model can't capture with only 6 points). Fitting on the 16 safe spans alone
took LOO-CV R^2 from -0.57 to +0.73. Before fitting each contiguous regression, we now look for
that kind of gap adaptively per (model, dataset): sort CKA descending, find the largest gap
between consecutive values, and split there IF that gap is much bigger than the typical
(median) gap AND it only carves off a minority of spans as "unsafe" -- otherwise there's no real
boundary (e.g. deit-base/imagenet-1k's CKA ranges smoothly from 0.28-0.90 with nothing behaving
like a floor effect, and the ungated fit there is already good: LOO-CV R^2=0.95) and all spans
are kept. Both the full-data and (when found) safe-only fits are reported; spans excluded by the
screen are flagged in the predictions CSV rather than silently dropped.

Data source: block_distance/distance_analysis.csv (the one file with disjoint-pair rows).

Writes two CSVs: one row per (model, dataset, shape) fit with coefficients and R^2
(skip_regression_coefs.csv), and one row per individual span with its features, actual
delta_acc, in-sample prediction, LOO-CV prediction, and LOO-CV residual
(skip_regression_predictions.csv).

    python block_distance/fit_skip_regression.py
"""

import argparse
import ast
from pathlib import Path

import numpy as np
import pandas as pd


def dropped_blocks(spans):
    blocks = []
    for start, end in spans:
        blocks.extend(range(start + 1, end + 1))
    return blocks


def ols_with_intercept(X, y):
    design = np.column_stack([np.ones(len(X)), X])
    coefs, *_ = np.linalg.lstsq(design, y, rcond=None)
    return coefs


def predict(design_row, coefs):
    return coefs[0] + np.dot(design_row, coefs[1:])


def r_squared(actual, predicted):
    actual = np.asarray(actual)
    predicted = np.asarray(predicted)
    ss_res = ((actual - predicted) ** 2).sum()
    ss_tot = ((actual - actual.mean()) ** 2).sum()
    return 1 - ss_res / ss_tot if ss_tot else float("nan")


def loo_cv_predictions(X, y):
    n = len(y)
    preds = np.empty(n)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        coefs = ols_with_intercept(X[mask], y[mask])
        preds[i] = predict(X[i], coefs)
    return preds


def find_cka_gap_split(cka_values, min_gap_ratio=5.0, max_unsafe_fraction=0.4):
    """Adaptive CKA-boundary detector: look for a gap in sorted CKA values that's much
    bigger than the typical spacing and only carves off a minority as "unsafe". Returns
    the threshold (spans with cka >= threshold are "safe") or None if no such gap exists.
    """
    sorted_desc = sorted(cka_values, reverse=True)
    n = len(sorted_desc)
    gaps = [sorted_desc[i] - sorted_desc[i + 1] for i in range(n - 1)]
    if not gaps:
        return None

    max_gap = max(gaps)
    gap_idx = gaps.index(max_gap)
    n_safe = gap_idx + 1
    n_unsafe = n - n_safe

    median_gap = sorted(gaps)[len(gaps) // 2]
    ratio = float("inf") if median_gap == 0 else max_gap / median_gap
    unsafe_fraction = n_unsafe / n

    if n_unsafe == 0 or ratio < min_gap_ratio or unsafe_fraction > max_unsafe_fraction:
        return None
    return (sorted_desc[gap_idx] + sorted_desc[gap_idx + 1]) / 2


def fit_and_report(label, model, dataset, rows, feature_cols, results, predictions, min_spans,
                    cka_screened=False):
    n = len(rows)
    if n < min_spans:
        print(f"  {label}: n={n}, below --min-spans, skipped")
        return

    X = rows[feature_cols].to_numpy()
    y = rows["delta_acc"].to_numpy()

    coefs = ols_with_intercept(X, y)
    in_sample_preds = np.array([predict(x, coefs) for x in X])
    in_sample_r2 = r_squared(y, in_sample_preds)
    loo_preds = loo_cv_predictions(X, y)
    cv_r2 = r_squared(y, loo_preds)

    terms = "  ".join(f"{c:+.4f}*{name}" for c, name in zip(coefs[1:], feature_cols))
    screen_tag = " [cka-safe subset]" if cka_screened else ""
    print(
        f"  {label}{screen_tag}: n={n}  delta_acc = {coefs[0]:+.4f}  {terms}  "
        f"R^2(in-sample)={in_sample_r2:+.3f}  R^2(LOO-CV)={cv_r2:+.3f}"
    )

    row = {
        "model": model, "dataset": dataset, "shape": label, "cka_screened": cka_screened,
        "n_spans": n, "intercept": coefs[0], "r2_in_sample": in_sample_r2, "r2_loo_cv": cv_r2,
    }
    row.update({f"coef_{name}": c for c, name in zip(coefs[1:], feature_cols)})
    results.append(row)

    pred_df = rows[["approx_layer_str"] + feature_cols].copy()
    if "cka_safe" in rows.columns:
        pred_df["cka_safe"] = rows["cka_safe"].values
    pred_df.insert(0, "cka_screened_fit", cka_screened)
    pred_df.insert(0, "shape", label)
    pred_df.insert(0, "dataset", dataset)
    pred_df.insert(0, "model", model)
    pred_df["actual_delta_acc"] = y
    pred_df["predicted_delta_acc_in_sample"] = in_sample_preds
    pred_df["predicted_delta_acc_loo_cv"] = loo_preds
    pred_df["residual_loo_cv"] = y - loo_preds
    predictions.append(pred_df)


def build_contiguous(rows, single_drop, cka):
    def features(row):
        (start, end), = row["spans"]
        blocks = dropped_blocks(row["spans"])
        drops = [single_drop.get(b) for b in blocks]
        if any(d is None for d in drops):
            return pd.Series({"cka": None, "block_drop_sum": None})
        return pd.Series({
            "cka": cka[start + 1, end + 1],
            "block_drop_sum": sum(drops),
        })

    rows = rows.copy()
    if rows.empty:
        rows["cka"], rows["block_drop_sum"] = None, None
        return rows
    rows[["cka", "block_drop_sum"]] = rows.apply(features, axis=1)
    return rows.dropna(subset=["cka", "block_drop_sum"])


def build_disjoint(rows, single_drop, cka):
    def features(row):
        spans = sorted(row["spans"])
        (s1, e1), (s2, e2) = spans
        b1, b2 = dropped_blocks([(s1, e1)])[0], dropped_blocks([(s2, e2)])[0]
        d1, d2 = single_drop.get(b1), single_drop.get(b2)
        if d1 is None or d2 is None:
            return pd.Series({"cka_1": None, "cka_2": None, "block_drop_sum": None})
        return pd.Series({
            "cka_1": cka[s1 + 1, e1 + 1],
            "cka_2": cka[s2 + 1, e2 + 1],
            "block_drop_sum": d1 + d2,
        })

    rows = rows.copy()
    if rows.empty:
        rows["cka_1"], rows["cka_2"], rows["block_drop_sum"] = None, None, None
        return rows
    rows[["cka_1", "cka_2", "block_drop_sum"]] = rows.apply(features, axis=1)
    return rows.dropna(subset=["cka_1", "cka_2", "block_drop_sum"])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--distance-csv", default="block_distance/distance_analysis.csv")
    p.add_argument("--cka-dir", default="src/layers/outputs")
    p.add_argument("--min-spans", type=int, default=5)
    p.add_argument("--out", default="block_distance/skip_regression_coefs.csv")
    p.add_argument("--predictions-out", default="block_distance/skip_regression_predictions.csv")
    args = p.parse_args()

    outputs_dir = Path(args.cka_dir)
    df = pd.read_csv(args.distance_csv)
    df["skip_count"] = df["skip_count"].astype(int)
    df["spans"] = df["approx_layer_str"].map(ast.literal_eval)

    results = []
    predictions = []
    for (model, dataset), g in df.groupby(["model", "dataset"]):
        baseline_rows = g[g["skip_count"] == 0]
        if baseline_rows.empty:
            continue
        baseline = baseline_rows["accuracy_mean_linear"].iloc[0]

        single = g[g["skip_count"] == 1]
        single_drop = {}
        for _, row in single.iterrows():
            blocks = dropped_blocks(row["spans"])
            if len(blocks) == 1:
                single_drop[blocks[0]] = baseline - row["accuracy_mean_linear"]

        cka_file = outputs_dir / f"{model.replace('/', '_')}_{dataset}_cka.npy"
        if not cka_file.exists():
            print(f"{model} / {dataset}: missing {cka_file.name}, skipped")
            continue
        cka = np.load(cka_file)

        two = g[g["skip_count"] == 2].copy()
        two["delta_acc"] = baseline - two["accuracy_mean_linear"]

        contiguous = build_contiguous(two[two["skip_distance"] == 1], single_drop, cka)
        disjoint = build_disjoint(two[two["skip_distance"] > 1], single_drop, cka)

        print(f"{model} / {dataset}  (baseline={baseline:.4f})")

        threshold = find_cka_gap_split(contiguous["cka"].tolist()) if len(contiguous) else None
        contiguous["cka_safe"] = True if threshold is None else contiguous["cka"] >= threshold
        fit_and_report("contiguous", model, dataset, contiguous,
                        ["cka", "block_drop_sum"], results, predictions, args.min_spans)
        if threshold is not None:
            unsafe = contiguous[~contiguous["cka_safe"]]
            print(f"  cka-boundary gap found at cka={threshold:.3f}: "
                  f"{len(unsafe)} unsafe span(s) excluded from the safe-only fit: "
                  f"{unsafe['approx_layer_str'].tolist()}")
            fit_and_report("contiguous", model, dataset, contiguous[contiguous["cka_safe"]],
                            ["cka", "block_drop_sum"], results, predictions, args.min_spans,
                            cka_screened=True)

        fit_and_report("disjoint", model, dataset, disjoint,
                        ["cka_1", "cka_2", "block_drop_sum"], results, predictions, args.min_spans)

    out_df = pd.DataFrame(results)
    out_df.to_csv(args.out, index=False)
    print(f"\nSaved {len(out_df)} model/dataset/shape fits -> {args.out}")

    pred_df = pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()
    pred_df.to_csv(args.predictions_out, index=False)
    print(f"Saved {len(pred_df)} per-span predictions -> {args.predictions_out}")


if __name__ == "__main__":
    main()
