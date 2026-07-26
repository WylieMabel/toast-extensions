"""
Layer / sublayer / head redundancy — analysis plots (headless).

Script port of ``notebooks/analysis.ipynb``: reads the score files written by
``layer_priority.py`` for one ``{model}_{dataset}`` stem and SAVES each figure into a
per-run folder (instead of showing them inline). See ``LAYER_PRIORITY.md`` for what each
metric means. Low ``bi_*`` = redundant = safe to drop.

Usage:
    python analysis_plots.py --stem dinobase_chestmnist \
        --output-dir src/layers/outputs --plot-dir src/layers/plots/dinobase_chestmnist
"""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: save figures, never open a window
import matplotlib.pyplot as plt
import numpy as np


def load_block_scores(path):
    # Two sections in one file; switch on the header token (robust to missing blank line).
    block, span, mode = {}, {}, None
    for r in csv.reader(open(path)):
        if not r or not r[0]:
            continue
        if r[0] == "block":
            mode = "block"; continue
        if r[0] == "blocks_dropped":
            mode = "span"; continue
        if mode == "block":
            block[int(r[0])] = (float(r[1]), float(r[2]))
        elif mode == "span":
            span[int(r[0])] = (int(r[1]), int(r[2]), float(r[3]))
    return block, span


def main(stem, output_dir, plot_dir):
    output_dir = Path(output_dir)
    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    def _p(suffix):
        return output_dir / f"{stem}_{suffix}"

    block, span = load_block_scores(_p("block_scores.csv"))
    angdist = np.load(_p("angdist.npy"))
    cka = np.load(_p("cka.npy"))
    sub_path = _p("sublayer_scores.csv")
    sub = (
        {int(r[0]): tuple(map(float, r[1:])) for r in list(csv.reader(open(sub_path)))[1:]}
        if sub_path.exists()
        else {}
    )
    head_path = _p("head_similarity.npy")
    head_sim = np.load(head_path) if head_path.exists() else None

    L = len(block)
    print(
        f"{stem}: {L} blocks | sublayer split: {bool(sub)} | "
        f"heads: {head_sim.shape if head_sim is not None else None}"
    )

    def _save(fig, name):
        out = plot_dir / name
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved -> {out}")

    # ----- Block Influence vs depth -----
    idx = sorted(block)
    bi_block = [block[i][0] for i in idx]
    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.bar(idx, bi_block, color="#4c78a8")
    lo = int(np.argmin(bi_block))
    bars[lo].set_color("#e45756")
    ax.set_xlabel("block index (depth ->)"); ax.set_ylabel("Block Influence  (1 - cos)")
    ax.set_title(f"{stem} — whole-block BI (red = lowest = most droppable)")
    ax.set_xticks(idx); ax.grid(axis="y", alpha=.3)
    fig.tight_layout(); _save(fig, "block_influence.png")

    order = np.argsort(bi_block)
    print("Blocks ranked most -> least droppable (skip pair to drop that single block):")
    for r in order:
        j = idx[r]
        pair = f"[({j - 1}, {j})]" if j >= 1 else "(block 0 cannot be dropped via reconnect)"
        print(f"  block {j:2d}   BI = {bi_block[r]:.4f}   skip = {pair}")

    # ----- Attention vs MLP sub-blocks -----
    if sub:
        li = sorted(sub)
        bi_attn = [sub[i][0] for i in li]; rn_attn = [sub[i][1] for i in li]
        bi_mlp = [sub[i][2] for i in li]; rn_mlp = [sub[i][3] for i in li]

        fig, ax = plt.subplots(1, 2, figsize=(13, 4), sharex=True)
        for a, bi, rn, name, col in [
            (ax[0], bi_attn, rn_attn, "attention", "#54a24b"),
            (ax[1], bi_mlp, rn_mlp, "MLP", "#f58518"),
        ]:
            a.bar(li, bi, color=col, label="BI")
            a2 = a.twinx(); a2.plot(li, rn, "k.--", lw=1, label="relnorm")
            a.set_title(f"{name} sub-block"); a.set_xlabel("layer")
            a.set_ylabel("BI (1 - cos)"); a2.set_ylabel("relnorm  ||d|| / ||h||")
            a.set_xticks(li); a.grid(axis="y", alpha=.3)
        fig.suptitle(f"{stem} — sub-block redundancy"); fig.tight_layout()
        _save(fig, "sublayer_bi.png")

        print("Lowest-BI attention layers (-> attn_skip):", list(np.argsort(bi_attn)[:5]))
        print("Lowest-BI MLP layers       (-> mlp_skip): ", list(np.argsort(bi_mlp)[:5]))
    else:
        print("No sublayer split for this architecture.")

    # ----- Angular-distance heatmap + best spans -----
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(angdist, cmap="viridis", origin="lower")
    ax.set_xlabel("hidden state j"); ax.set_ylabel("hidden state i")
    ax.set_title(f"{stem} — angular distance d(i, j)")
    fig.colorbar(im, ax=ax, fraction=.046)
    for k, (s, e, d) in sorted(span.items()):
        ax.plot(e + 1, s + 1, "r.", ms=8)
    fig.tight_layout(); _save(fig, "angular_distance.png")

    print("Best placement per number of blocks dropped  (skip = [(skip_from, skip_to)]):")
    for k in sorted(span):
        s, e, d = span[k]
        print(f"  drop {k} block(s):  skip = [({s}, {e})]   angdist = {d:.4f}   cka = {cka[s + 1, e + 1]:.2f}")

    # ----- CKA heatmap -----
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cka, cmap="magma", origin="lower", vmin=0, vmax=1)
    ax.set_xlabel("hidden state j"); ax.set_ylabel("hidden state i")
    ax.set_title(f"{stem} — linear CKA")
    fig.colorbar(im, ax=ax, fraction=.046)
    fig.tight_layout(); _save(fig, "cka.png")

    # ----- Attention-head similarity (JSD) -----
    if head_sim is not None:
        L_h, H, _ = head_sim.shape
        cols = 4; rows = (L_h + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
        for l, ax in enumerate(axes.ravel()):
            if l < L_h:
                ax.imshow(head_sim[l], cmap="cividis", vmin=0, vmax=1)
                ax.set_title(f"layer {l}", fontsize=9)
                ax.set_xticks([]); ax.set_yticks([])
            else:
                ax.axis("off")
        fig.suptitle(f"{stem} — head JSD similarity (1 = identical)"); fig.tight_layout()
        _save(fig, "head_similarity.png")
    else:
        print("No head similarity file (model did not expose attentions).")

    print(f"\nDone. Plots written to: {plot_dir.resolve()}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Save layer-priority analysis plots for one stem")
    p.add_argument("--stem", required=True, help="{model}_{dataset} stem, e.g. dinobase_chestmnist")
    p.add_argument("--output-dir", default="src/layers/outputs", help="dir holding the score files")
    p.add_argument("--plot-dir", required=True, help="dir to write the PNGs into")
    args = p.parse_args()
    main(args.stem, args.output_dir, args.plot_dir)
