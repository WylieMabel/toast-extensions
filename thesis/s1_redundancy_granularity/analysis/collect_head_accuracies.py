"""Joins the s1 head-pruning accuracies to the thresholds that produced each head_dict.
Run from the project root:
    python thesis/s1_redundancy_granularity/analysis/collect_head_accuracies.py

One row per model x metric x threshold (0.05 .. 0.95 from results/head_similarity/head_keep_*.csv,
plus 1.00 = the unpruned baseline row from the s2 accuracies). A head_dict reached by several
thresholds or by both metrics shares one accuracy, since each distinct head_dict was run once.

Model size follows src/heads/notebooks/heads.ipynb (encoder_data.csv): heads kept against the
full model's layers x heads, and the parameters those pruned heads free. params_saved is read from
the pipeline's results (toast.utils.dictionaries.PARAM_SAVINGS: Q/K/V + output-projection slice
per head), and params_saved_pct is relative to the encoder body, as in the s2 results.

Also writes head_pruning_random_imagenet1k.csv: the random-pruning baseline
(make_random_head_configs.py), one row per model x level x draw, with the same size columns.
"""
import ast
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, "src")
from toast.utils.dictionaries import MODEL2NUM_LAYERS, _resolve_param_savings  # noqa: E402

S1 = Path("thesis/s1_redundancy_granularity")
S2_ACC = Path("thesis/s2_skip_cost_metrics/accuracies")
DATASET = "imagenet-1k"
# head_similarity model key -> slug used in the s2 accuracies file names.
S2_SLUGS = {"vitsmall": "vitsmall", "deitsmall": "deitsmall", "dinobase": "dinov2base", "vitlarge": "vitlarge"}
COLUMNS = ["metric", "threshold", "model", "dataset", "head_dict", "accuracy_mean", "accuracy_std",
           "total_heads_kept", "total_heads", "heads_kept_pct", "params_saved", "params_saved_pct"]
RANDOM_COLUMNS = ["model", "dataset", "level_pct", "draw", "head_dict", "accuracy_mean", "accuracy_std",
                  "heads_removed", "total_heads_kept", "total_heads", "heads_kept_pct",
                  "params_saved", "params_saved_pct"]


def _canon(hd):
    """Same head_dict string however it was formatted (key order, spacing)."""
    return str(dict(sorted(ast.literal_eval(hd).items())))


def _is_baseline(df):
    return ((df.approx_layer == "[]") & (df.mlp_linearize == "[]")
            & (df.attn_linearize == "[]") & (df.head_dict == "{}"))


def _accuracies_and_params(prefix, model):
    """Per-head_dict accuracy (mean/std over seeds) and params saved, keyed by canonical head_dict."""
    acc = pd.read_csv(S1 / f"accuracies/accuracies_results_{prefix}_{model}_imagenet1k.csv")
    acc["key"] = acc.head_dict.map(_canon)
    acc = acc.set_index("key")[["accuracy_mean", "accuracy_std"]]
    # params_saved is the same for every seed of a head_dict; take one per head_dict.
    res = pd.read_csv(S1 / f"results/results_{prefix}_{model}_imagenet1k.csv", keep_default_na=False)
    res["key"] = res.head_dict.map(_canon)
    params = res.groupby("key")[["params_saved", "params_saved_pct"]].first()
    return acc, params


def _join(df, acc, params, what):
    df = df.assign(key=df.head_dict.map(_canon))
    missing = set(df.key) - set(acc.index)
    if missing:
        raise ValueError(f"{what}: {len(missing)} head_dicts have no accuracy, e.g. {next(iter(missing))}")
    return df.join(acc, on="key").join(params, on="key")


def collect_random():
    draws = pd.read_csv(S1 / "results/head_similarity/random_heads_imagenet1k.csv")
    rows = []
    for model, g in draws.groupby("model", sort=False):
        acc, params = _accuracies_and_params("random_heads", model)
        rows.append(_join(g, acc, params, f"random {model}").assign(model=g.encoder))
    out = pd.concat(rows, ignore_index=True)
    out["heads_kept_pct"] = 100.0 * out.total_heads_kept / out.total_heads
    out = out[RANDOM_COLUMNS].sort_values(["model", "level_pct", "draw"], kind="stable")
    path = S1 / "analysis/head_pruning_random_imagenet1k.csv"
    out.to_csv(path, index=False, float_format="%.6g")
    print(f"{len(out)} rows -> {path}")


def main():
    keep = pd.read_csv(S1 / "results/head_similarity/head_keep_imagenet1k.csv")
    rows = []
    for model, g in keep.groupby("model", sort=False):
        encoder = g.encoder.iloc[0]
        total_heads = MODEL2NUM_LAYERS[encoder] * _resolve_param_savings(encoder)["heads"]

        acc, params = _accuracies_and_params("heads", model)
        rows.append(_join(g, acc, params, model).assign(model=encoder, total_heads=total_heads))

        s2 = pd.read_csv(S2_ACC / f"accuracies_results_{S2_SLUGS[model]}_imagenet1k.csv", keep_default_na=False)
        base = s2[_is_baseline(s2)]
        if len(base) != 1:
            raise ValueError(f"{model}: expected one s2 baseline row, found {len(base)}")
        for metric in g.metric.unique():
            rows.append(pd.DataFrame([{
                "metric": metric, "threshold": 1.0, "model": base.model.iloc[0], "dataset": DATASET,
                "head_dict": "{}", "accuracy_mean": base.accuracy_mean.iloc[0],
                "accuracy_std": base.accuracy_std.iloc[0],
                "total_heads_kept": total_heads, "total_heads": total_heads,
                "params_saved": 0, "params_saved_pct": 0.0,
            }]))

    out = pd.concat(rows, ignore_index=True)
    out["heads_kept_pct"] = 100.0 * out.total_heads_kept / out.total_heads
    out = out[COLUMNS]
    out = out.sort_values(["model", "metric", "threshold"], kind="stable")
    path = S1 / "analysis/head_pruning_accuracies_imagenet1k.csv"
    out.to_csv(path, index=False, float_format="%.6g")
    print(f"{len(out)} rows -> {path}")

    collect_random()


if __name__ == "__main__":
    main()
