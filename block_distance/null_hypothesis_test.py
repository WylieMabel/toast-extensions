"""Step 2 of the skip-candidate predictor plan: is the Step 1 regression's predicted-best
skip actually better than picking at random?

For each (model, dataset, shape) with a fit in skip_regression_coefs.csv:

  1. Pick the predictor: whichever of the "full" and "cka-safe" contiguous fits has the
     higher LOO-CV R^2 (Step 1 found the screen helps some model/dataset pairs and hurts
     others -- see fit_skip_regression.py docstring -- so this is decided per fit, not
     assumed). Disjoint only ever has one fit.
  2. The predictor's recommended span is whichever candidate it was trained on has the
     lowest LOO-CV predicted damage (predicted_delta_acc_loo_cv) -- using the LOO-CV
     prediction, not the in-sample one, so the "recommendation" is the same kind of
     out-of-sample guess the predictor would make on an unseen span.
  3. Null hypothesis: instead of using the predictor, try a few (--k, default 3) random
     spans from the SAME model/dataset/shape's full candidate pool (unfiltered by any CKA
     screen -- a random guesser wouldn't know to apply that) and take the best of them.
     Repeat --trials times to build a distribution of "best-of-k-random" damage, and take
     its 5th/95th percentile as the null CI.
  4. Compare: does the predictor's actual damage beat the CI's lower bound? Also report the
     exact single-random-pick p-value: the fraction of the full candidate pool that is at
     least as good as the predictor's pick (i.e. what a single uniform-random draw would
     have beaten it with).

    python block_distance/null_hypothesis_test.py
"""

import argparse

import numpy as np
import pandas as pd


def best_of_k_random(pool_damage, k, n_trials, rng):
    n = len(pool_damage)
    k = min(k, n)
    trial_mins = np.empty(n_trials)
    for t in range(n_trials):
        draw = rng.choice(pool_damage, size=k, replace=False)
        trial_mins[t] = draw.min()
    return trial_mins


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--coefs", default="block_distance/skip_regression_coefs.csv")
    p.add_argument("--predictions", default="block_distance/skip_regression_predictions.csv")
    p.add_argument("--k", type=int, default=3, help="'a few' random spans tried per trial")
    p.add_argument("--trials", type=int, default=2000)
    p.add_argument("--ci", type=float, default=0.90, help="central CI width, e.g. 0.90 = 5th-95th pct")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="block_distance/null_hypothesis_results.csv")
    args = p.parse_args()

    coefs = pd.read_csv(args.coefs)
    preds = pd.read_csv(args.predictions)
    rng = np.random.default_rng(args.seed)

    lo_pct = (1 - args.ci) / 2 * 100
    hi_pct = (1 + args.ci) / 2 * 100

    results = []
    for (model, dataset, shape), _ in coefs.groupby(["model", "dataset", "shape"]):
        fits = coefs[(coefs["model"] == model) & (coefs["dataset"] == dataset) & (coefs["shape"] == shape)]
        chosen = fits.loc[fits["r2_loo_cv"].idxmax()]

        chosen_preds = preds[
            (preds["model"] == model) & (preds["dataset"] == dataset) & (preds["shape"] == shape)
            & (preds["cka_screened_fit"] == chosen["cka_screened"])
        ]
        if chosen_preds.empty:
            continue

        best_row = chosen_preds.loc[chosen_preds["predicted_delta_acc_loo_cv"].idxmin()]
        predicted_span = best_row["approx_layer_str"]
        predicted_actual_damage = best_row["actual_delta_acc"]

        full_pool = preds[
            (preds["model"] == model) & (preds["dataset"] == dataset) & (preds["shape"] == shape)
            & (~preds["cka_screened_fit"])
        ]
        pool_damage = full_pool["actual_delta_acc"].to_numpy()
        n_pool = len(pool_damage)
        if n_pool < 2:
            continue

        trial_mins = best_of_k_random(pool_damage, args.k, args.trials, rng)
        ci_lo, ci_hi = np.percentile(trial_mins, [lo_pct, hi_pct])
        beats_ci = predicted_actual_damage < ci_lo

        rank = (pool_damage <= predicted_actual_damage).sum()
        single_pick_p = rank / n_pool
        min_achievable_p = 1 / n_pool
        # With a discrete pool of n_pool candidates, a single random pick can be "wrong" at
        # best 1/n_pool of the time -- p < 0.05 is mathematically unreachable when n_pool < 20,
        # regardless of how good the predictor is. Flag that explicitly rather than reporting
        # "not significant" as if it were evidence against the predictor.
        underpowered = min_achievable_p >= 0.05
        significant = (not underpowered) and single_pick_p < 0.05

        print(
            f"{model} / {dataset} / {shape} (n_pool={n_pool}, screened={bool(chosen['cka_screened'])}): "
            f"predicted best = {predicted_span}  actual damage={predicted_actual_damage:+.4f}  |  "
            f"best-of-{args.k}-random {int(args.ci*100)}% CI=[{ci_lo:+.4f}, {ci_hi:+.4f}]  "
            f"{'BEATS' if beats_ci else 'does not beat'} CI  |  "
            f"single-random-pick p={single_pick_p:.3f}"
            f"{'  [UNDERPOWERED: min possible p=' + f'{min_achievable_p:.2f}]' if underpowered else ''}"
        )

        results.append({
            "model": model, "dataset": dataset, "shape": shape,
            "cka_screened": bool(chosen["cka_screened"]), "n_pool": n_pool,
            "predicted_span": predicted_span, "predicted_actual_damage": predicted_actual_damage,
            "random_ci_lo": ci_lo, "random_ci_hi": ci_hi, "beats_ci": beats_ci,
            "single_random_pick_p": single_pick_p, "min_achievable_p": min_achievable_p,
            "underpowered": underpowered, "significant": significant,
        })

    out_df = pd.DataFrame(results)
    out_df.to_csv(args.out, index=False)
    powered = out_df[~out_df["underpowered"]]
    print(f"\n{len(out_df) - len(powered)}/{len(out_df)} combos are underpowered "
          f"(n_pool < 20, can never reach p<0.05) -- excluded from the significance count.")
    print(f"{powered['significant'].sum()}/{len(powered)} adequately-powered combos beat random "
          f"at p<0.05. Saved -> {args.out}")


if __name__ == "__main__":
    main()
