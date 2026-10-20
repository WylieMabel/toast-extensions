"""Random head-pruning baseline for the s1 CKA / angular-distance head pruning. Run from the
project root:
    python thesis/s1_redundancy_granularity/make_random_head_configs.py

For each model, removes a fixed fraction of all attention heads (LEVELS) chosen uniformly at
random, DRAWS times per level, so the metric curves can be compared with chance at the same
number of heads removed. Like the metric rule (which always keeps head 0), every layer keeps at
least one head: draws are uniform over all such subsets (rejection sampling). Draws are seeded per
model / level / draw, so re-running gives the same head_dicts.

Writes, in configs/, one experiments CSV per model (baseline row first) in the same format as
experiments_heads_*.csv, and results/head_similarity/random_heads_imagenet1k.csv recording which
level and draw each head_dict came from (the pipeline results do not carry them).
"""
import csv
import sys
import zlib
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")
from toast.utils.dictionaries import MODEL2NUM_LAYERS, _resolve_param_savings  # noqa: E402

S1 = Path("thesis/s1_redundancy_granularity")
DATASET = "imagenet-1k"
MODELS = {
    "vitsmall": "WinKawaks/vit-small-patch16-224",
    "deitsmall": "facebook/deit-small-patch16-224",
    "dinobase": "facebook/dinov2-base",
    "vitlarge": "google/vit-large-patch16-224",
}
LEVELS = [2.5, 5, 7.5, 10, 15, 25, 50, 75]  # % of all heads removed
DRAWS = 3
CONFIG_COLUMNS = ["dataset", "encoder", "skip", "mlp_skip", "attn_skip", "head_dict",
                  "skip_translator", "mlp_mode", "attn_mode", "fit_dataset"]


def _draw(rng, num_layers, num_heads, n_remove):
    """Remove n_remove heads uniformly at random, leaving every layer at least one head."""
    while True:
        removed = rng.choice(num_layers * num_heads, size=n_remove, replace=False)
        per_layer = np.bincount(removed // num_heads, minlength=num_layers)
        if (per_layer < num_heads).all():
            break
    head_dict = {}
    for layer in range(num_layers):
        gone = {int(h) % num_heads for h in removed if h // num_heads == layer}
        if gone:
            head_dict[layer] = [h for h in range(num_heads) if h not in gone]
    return str(head_dict)


def main():
    meta = []
    for key, encoder in MODELS.items():
        num_layers = MODEL2NUM_LAYERS[encoder]
        num_heads = _resolve_param_savings(encoder)["heads"]
        total = num_layers * num_heads

        rows, seen = [], {"{}"}
        for level in LEVELS:
            n_remove = round(total * level / 100)
            for draw in range(DRAWS):
                rng = np.random.default_rng(zlib.crc32(f"{key}-{level}-{draw}".encode()))
                head_dict = _draw(rng, num_layers, num_heads, n_remove)
                while head_dict in seen:  # keep every row distinct (only possible at tiny counts)
                    head_dict = _draw(rng, num_layers, num_heads, n_remove)
                seen.add(head_dict)
                rows.append(head_dict)
                meta.append({"dataset": DATASET, "encoder": encoder, "model": key,
                             "level_pct": level, "draw": draw, "heads_removed": n_remove,
                             "total_heads_kept": total - n_remove, "total_heads": total,
                             "head_dict": head_dict})

        path = S1 / f"configs/experiments_random_heads_{key}_imagenet1k.csv"
        with open(path, "w", newline="") as f:
            w = csv.writer(f, quoting=csv.QUOTE_ALL)
            w.writerow(CONFIG_COLUMNS)
            for hd in ["{}"] + rows:
                w.writerow([DATASET, encoder, "[]", "[]", "[]", hd, "identity", "identity", "identity", ""])
        print(f"{key}: {len(rows) + 1} rows -> {path}")

    meta_path = S1 / "results/head_similarity/random_heads_imagenet1k.csv"
    with open(meta_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(meta[0]))
        w.writeheader()
        w.writerows(meta)
    print(f"draw metadata -> {meta_path}")


if __name__ == "__main__":
    main()
