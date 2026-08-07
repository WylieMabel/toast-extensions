"""
Linear-CKA heatmaps for every model x dataset combination run by run_layer_priority.sh.

Reads the ``{stem}_cka.npy`` matrices written by ``layer_priority.py`` and saves one
heatmap per combination, plus a single overview grid (models x datasets), into
``src/layers/plots/cka``. Axes are BLOCK-OUTPUT indices, not raw hidden-state indices:
value i means out(block i) == hidden_state[i + 1], and the embedding (hidden_state[0]) sits
at -1. So cell (i, j) is exactly the pair a skip=(skip_from=i, skip_to=j) has to bridge,
and high CKA there predicts an ``identity`` translator suffices.

Legacy alias stems ("dinobase_cifar100") are ignored whenever the canonical stem
("facebook_dinov2-base_cifar100") holds the same matrix, so a combination is plotted once.

Usage:
    python src/layers/plot_cka.py
    python src/layers/plot_cka.py --output-dir src/layers/outputs --plot-dir src/layers/plots/cka
"""

import argparse
import ast
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: save figures, never open a window
import matplotlib.pyplot as plt
import numpy as np


def _model_aliases():
    """MODEL_ALIASES from layer_priority.py, read from source rather than imported.

    Importing that module pulls in torch/transformers/datasets, which need not be
    installed wherever the plots are made (a laptop, not the GPU node), so parse the
    literal out instead of duplicating it here and letting the two drift.
    """
    src = (Path(__file__).with_name("layer_priority.py")).read_text()
    for node in ast.parse(src).body:
        if isinstance(node, ast.Assign) and any(
            getattr(t, "id", None) == "MODEL_ALIASES" for t in node.targets
        ):
            return ast.literal_eval(node.value)
    raise RuntimeError("MODEL_ALIASES not found in layer_priority.py")


MODEL_ALIASES = _model_aliases()

# Row (model) / column (dataset) order of the overview grid, matching the skip-distance
# accuracy figure so the two can sit side by side. Anything found on disk but not listed
# here is appended after these, so the grid never silently drops a combination -- which is
# why chestmnist lands in a trailing column (pass --datasets to leave it out).
MODEL_ORDER = [
    "facebook/deit-base-patch16-224",
    "facebook/deit-small-patch16-224",
    "facebook/dinov2-base",
    "google/vit-base-patch16-224",
    "google/vit-large-patch16-224",
    "microsoft/rad-dino",
]
DATASET_ORDER = ["pneumoniamnist", "imagenet-1k", "cifar100", "dermamnist", "chestmnist"]

CMAP = "magma"  # sequential, one hue light->dark: CKA is a magnitude in [0, 1]


def _canonical_prefixes():
    """model-stem prefix -> HF id, for both the canonical form and every alias."""
    prefixes = {}
    for alias, hf_id in MODEL_ALIASES.items():
        prefixes[hf_id.lower().replace("/", "_")] = hf_id
        prefixes[alias] = hf_id
    return prefixes


def discover(output_dir):
    """[(hf_id, dataset, stem, path)] for each CKA matrix, aliases folded into canonicals."""
    prefixes = _canonical_prefixes()
    found = {}
    for path in sorted(output_dir.glob("*_cka.npy")):
        stem = path.name[: -len("_cka.npy")]
        match = max(
            (p for p in prefixes if stem.startswith(p + "_")), key=len, default=None
        )
        if match is None:
            print(f"  ! unrecognised stem, skipping: {stem}")
            continue
        hf_id = prefixes[match]
        dataset = stem[len(match) + 1 :]
        key = (hf_id, dataset)
        canonical = f"{hf_id.lower().replace('/', '_')}_{dataset}"
        # Prefer the canonical file when both it and an alias copy exist.
        if key in found and found[key][0] == canonical:
            continue
        found[key] = (stem, path)

    def sort_key(item):
        (hf_id, dataset), _ = item
        m = MODEL_ORDER.index(hf_id) if hf_id in MODEL_ORDER else len(MODEL_ORDER)
        d = DATASET_ORDER.index(dataset) if dataset in DATASET_ORDER else len(DATASET_ORDER)
        return (m, hf_id, d, dataset)

    return [
        (hf_id, dataset, stem, path)
        for (hf_id, dataset), (stem, path) in sorted(found.items(), key=sort_key)
    ]


# The matrix is indexed by hidden state, where hidden_state[0] is the embedding output and
# hidden_state[i + 1] is the output of block i. Plot in BLOCK-OUTPUT indices instead, so
# axis value i means out(block i) and matches the skip=(skip_from, skip_to) convention:
# shift the image by one cell, which puts the embedding at -1.
EMBED_INDEX = -1


def _extent(n_states):
    """imshow extent putting cell centres at -1 (embedding), 0 .. n_states - 2 (blocks)."""
    lo, hi = EMBED_INDEX - 0.5, n_states - 1.5
    return (lo, hi, lo, hi)


def _tick_positions(n_states):
    """Block-output ticks 0 .. L-1 (the embedding column is drawn but not ticked here)."""
    n_blocks = n_states - 1
    step = 1 if n_blocks <= 13 else 2 if n_blocks <= 25 else 4
    return list(range(0, n_blocks, step))


def plot_one(cka, hf_id, dataset, out_path):
    n = cka.shape[0]
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(cka, cmap=CMAP, origin="lower", vmin=0, vmax=1, extent=_extent(n))
    ticks = [EMBED_INDEX] + _tick_positions(n)
    # "e", not "emb": the embedding tick sits one cell from the 0 tick, and the longer
    # label runs into it on the x axis.
    labels = ["e"] + [str(t) for t in ticks[1:]]
    ax.set_xticks(ticks, labels); ax.set_yticks(ticks, labels)
    ax.set_xlabel("block output j   (e = embedding)")
    ax.set_ylabel("block output i   (e = embedding)")
    ax.set_title(f"{hf_id}  |  {dataset}\nlinear CKA between block outputs", fontsize=10)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("linear CKA  (1 = identical representation)", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_grid(entries, out_path):
    """One model x dataset panel grid, laid out like the skip-distance accuracy figure:
    a two-line `{model}\\n{dataset}` title over every panel, one shared colorbar."""
    models = list(dict.fromkeys(hf_id for hf_id, _, _, _ in entries))
    datasets = list(dict.fromkeys(ds for _, ds, _, _ in entries))
    by_key = {(hf_id, ds): path for hf_id, ds, _, path in entries}

    fig, axes = plt.subplots(
        len(models), len(datasets),
        figsize=(3.0 * len(datasets) + 1.0, 3.0 * len(models)),
        squeeze=False,
    )
    im = None
    for r, hf_id in enumerate(models):
        for c, ds in enumerate(datasets):
            ax = axes[r][c]
            ax.set_title(f"{hf_id}\n{ds}", fontsize=9)
            path = by_key.get((hf_id, ds))
            if path is None:
                ax.set_xticks([]); ax.set_yticks([])
                ax.text(.5, .5, "not run", ha="center", va="center",
                        fontsize=9, color="#888888", transform=ax.transAxes)
                for side in ax.spines.values():
                    side.set_color("#cccccc")
                continue
            cka = np.load(path)
            im = ax.imshow(cka, cmap=CMAP, origin="lower", vmin=0, vmax=1,
                           extent=_extent(cka.shape[0]))
            # Halve the per-panel tick density of the single plots: at 3 in a panel, every
            # index collides on the 24-block models. The embedding cell (-1) is left
            # unticked here too -- "emb" would run into the 0 tick at this size.
            ticks = _tick_positions(cka.shape[0])[::2]
            ax.set_xticks(ticks); ax.set_yticks(ticks)
            ax.tick_params(labelsize=7)
            ax.set_xlabel("block output j", fontsize=8)
            ax.set_ylabel("block output i", fontsize=8)

    fig.tight_layout(rect=(0, 0, 0.93, 1))
    if im is not None:
        cax = fig.add_axes((0.945, 0.15, 0.011, 0.7))
        cbar = fig.colorbar(im, cax=cax)
        cbar.set_label("linear CKA  (1 = identical representation)", fontsize=9)
        cbar.ax.tick_params(labelsize=8)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main(output_dir, plot_dir, datasets=None, grid_only=False):
    output_dir = Path(output_dir)
    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    entries = discover(output_dir)
    if not entries:
        raise SystemExit(f"No *_cka.npy files found in {output_dir.resolve()}")

    if not grid_only:
        for hf_id, dataset, stem, path in entries:
            cka = np.load(path)
            out = plot_dir / f"{stem}_cka.png"
            plot_one(cka, hf_id, dataset, out)
            print(f"  {hf_id:34s} {dataset:16s} {cka.shape[0]:>3d} states -> {out}")

    if datasets:
        keep = [e for e in entries if e[1] in datasets]
        if not keep:
            raise SystemExit(f"No combinations left after --datasets {datasets}")
        entries = keep

    grid = plot_dir / "cka_all_combinations.png"
    plot_grid(entries, grid)
    print(f"\n  overview grid ({len(entries)} panels) -> {grid}")
    print(f"\nDone. Plots written to: {plot_dir.resolve()}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Save linear-CKA heatmaps for every combination")
    p.add_argument("--output-dir", default="src/layers/outputs", help="dir holding *_cka.npy")
    p.add_argument("--plot-dir", default="src/layers/plots/cka", help="dir to write the PNGs into")
    p.add_argument("--datasets", nargs="+", help="restrict the grid to these datasets (default: all)")
    p.add_argument("--grid-only", action="store_true", help="write only the overview grid")
    args = p.parse_args()
    main(args.output_dir, args.plot_dir, args.datasets, args.grid_only)
