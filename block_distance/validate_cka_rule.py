"""Audit the CKA-based translator-selection rule against the skip-distance sweep.

recommend_runs.py:92-94 decides whether a skip needs a translator by looking at CKA between
the two hidden states it bridges:

    return "identity" if cka[skip_from + 1, skip_to + 1] >= CKA_IDENTITY else "linear"

That is a falsifiable prediction, and block_distance/results_imgnet_dinov2.csv contains the
runs to test it against. This script joins the two and reports whether the rule holds.

    python block_distance/validate_cka_rule.py

Needs the CKA matrix from layer_priority (src/layers/outputs/<stem>_cka.npy) and the sweep
results for the same model/dataset pair.
"""

import argparse
import ast
from pathlib import Path

import numpy as np
import pandas as pd


def load(results_csv, cka_npy):
    cka = np.load(cka_npy)
    df = pd.read_csv(results_csv)

    baseline = df[df["approx_layer"] == "[]"]["accuracy"].mean()

    d = df[df["approx_layer"] != "[]"].copy()
    d["span"] = d["approx_layer"].map(
        lambda s: (lambda v: v[0] if len(v) == 1 else None)(ast.literal_eval(s))
    )
    d = d[d["span"].notna()]

    g = d.groupby(["span", "translator"])["accuracy"].mean().unstack()
    g = g.dropna(subset=["identity", "linear"])
    g["cka"] = [cka[a + 1, b + 1] for a, b in g.index]
    g["dist"] = [b - a for a, b in g.index]
    g["depth"] = [a for a, b in g.index]
    # Drop from the unmodified encoder -- the number that actually matters. Comparing identity
    # against linear instead conflates "both fine" with "both useless".
    g["id_drop"] = baseline - g["identity"]
    g["lin_drop"] = baseline - g["linear"]
    return g, baseline


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", default="block_distance/results_imgnet_dinov2.csv")
    p.add_argument("--cka", default="src/layers/outputs/dinobase_imagenet-1k_cka.npy")
    p.add_argument("--threshold", type=float, default=0.90, help="CKA_IDENTITY in recommend_runs")
    p.add_argument("--safe-tol", type=float, default=0.02,
                   help="max drop from baseline for identity to count as safe")
    args = p.parse_args()

    g, baseline = load(args.results, Path(args.cka))
    print(f"n = {len(g)} single-span configs | no-skip baseline = {baseline:.4f}\n")

    g["identity_safe"] = g["id_drop"] < args.safe_tol
    g["rule_says_identity"] = g["cka"] >= args.threshold

    print(f"Is identity ever safe (within {args.safe_tol:.0%} of baseline)?")
    print(f"  {g['identity_safe'].sum()}/{len(g)} spans\n")

    fired = g[g["rule_says_identity"]]
    print(f"Rule (CKA >= {args.threshold}) recommended identity for {len(fired)} spans:")
    if len(fired):
        print(fired[["cka", "identity", "linear", "id_drop"]].round(4).to_string())
        print(f"\n  mean accuracy lost by following it: {fired['id_drop'].mean():.3f}")
        print(f"  correct calls: {(fired['identity_safe']).sum()}/{len(fired)}")

    print("\nWhat CKA actually predicts:")
    print(f"  CKA vs identity drop : {g['cka'].corr(g['id_drop']):+.3f}  "
          f"(what the rule assumes it predicts)")
    print(f"  CKA vs linear  drop  : {g['cka'].corr(g['lin_drop']):+.3f}  "
          f"(what it really predicts)")

    d1 = g[g["dist"] == 1]
    print(f"\n  controlling for span length (distance-1 only, n={len(d1)}):")
    print(f"    CKA vs identity drop : {d1['cka'].corr(d1['id_drop']):+.3f}")
    print(f"    CKA vs linear  drop  : {d1['cka'].corr(d1['lin_drop']):+.3f}")

    print("\nAll spans, ordered by identity's drop from baseline:")
    cols = ["cka", "dist", "identity", "linear", "id_drop", "lin_drop"]
    print(g.sort_values("id_drop")[cols].round(4).to_string())


if __name__ == "__main__":
    main()
