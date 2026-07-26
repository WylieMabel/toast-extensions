"""
Recommend runs from redundancy scores (script port of ``notebooks/recommend_runs.ipynb``).

Turns the ``layer_priority.py`` score files for one ``{model}_{dataset}`` stem into candidate
rows for an ``experiments_*.csv``, in the exact 9-column schema the pipeline expects
(``dataset,encoder,skip,mlp_skip,attn_skip,head_dict,skip_translator,mlp_mode,attn_mode``).

With ``--append`` the rows are appended (no header) to an existing CSV, so a master loop can
accumulate one file across many model x dataset combos. Without it the file is (over)written
with a header — use this for the first combo.

``skip`` convention (SkipModel): a pair ``(skip_from, skip_to)`` keeps block ``skip_from``,
DROPS blocks ``skip_from+1 .. skip_to``, and bridges ``out(skip_from) -> out(skip_to)``. Both
indices must be in ``0 .. num_layers-1``. To drop a single block ``j``, write ``(j-1, j)``.

Usage:
    python recommend_runs.py --stem raddino_dermamnist --dataset dermamnist \
        --encoder microsoft/rad-dino --output-dir src/layers/outputs \
        --out-csv src/configs/experiments_pipeline.csv [--append]
"""

import argparse
import ast
import csv
from pathlib import Path

import numpy as np
import pandas as pd

# ---- knobs (edit here to change how aggressive the candidate list is) ----
N_SINGLE_BLOCKS = 3     # lowest-BI single blocks to try, dropped as skip=[(j-1, j)]
MAX_SPAN        = 4     # try best placement for dropping 2..MAX_SPAN contiguous blocks
N_MLP           = 3     # lowest-BI MLP layers -> one combined mlp_skip candidate (+ individually)
N_ATTN          = 3     # lowest-BI attention layers -> one combined attn_skip candidate (+ individually)
CKA_IDENTITY    = 0.90  # bridged-endpoint CKA above this -> skip_translator=identity, else linear
HEAD_THRESHOLD  = 0.90  # head-JSD similarity above this -> prune the higher-index head
# --------------------------------------------------------------------------

SCHEMA = ["dataset", "encoder", "skip", "mlp_skip", "attn_skip",
          "head_dict", "skip_translator", "mlp_mode", "attn_mode"]


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
            block[int(r[0])] = float(r[1])
        elif mode == "span":
            span[int(r[0])] = (int(r[1]), int(r[2]), float(r[3]))
    return block, span


def build_candidates(stem, dataset, encoder, output_dir):
    output_dir = Path(output_dir)

    def _p(s):
        return output_dir / f"{stem}_{s}"

    block_bi, span = load_block_scores(_p("block_scores.csv"))
    cka = np.load(_p("cka.npy"))
    sub_path = _p("sublayer_scores.csv")
    sub = (
        {int(r[0]): tuple(map(float, r[1:])) for r in list(csv.reader(open(sub_path)))[1:]}
        if sub_path.exists()
        else {}
    )
    head_path = _p("head_similarity.npy")
    head_sim = np.load(head_path) if head_path.exists() else None
    L = len(block_bi)
    print(f"Loaded {stem}: {L} blocks, sublayer={bool(sub)}, heads={head_sim is not None}")

    def row(skip=None, mlp_skip=None, attn_skip=None, head_dict=None,
            skip_translator="identity", mlp_mode="identity", attn_mode="identity", note=""):
        return {
            "dataset": dataset, "encoder": encoder,
            "skip": repr(skip or []),
            "mlp_skip": repr(mlp_skip or []),
            "attn_skip": repr(attn_skip or []),
            "head_dict": repr(head_dict or {}),
            "skip_translator": skip_translator,
            "mlp_mode": mlp_mode, "attn_mode": attn_mode,
            "_note": note,
        }

    def translator_for_span(skip_from, skip_to):
        # skip bridges hidden state (skip_from+1) -> (skip_to+1)
        return "identity" if cka[skip_from + 1, skip_to + 1] >= CKA_IDENTITY else "linear"

    def head_keep_dict(threshold):
        # greedy within-layer prune (same rule as head_priority.py)
        if head_sim is None:
            return {}
        Lh, H, _ = head_sim.shape
        out = {}
        for l in range(Lh):
            remove = set()
            for i in range(H):
                if i in remove:
                    continue
                for j in range(i + 1, H):
                    if head_sim[l, i, j] > threshold:
                        remove.add(j)
            out[l] = sorted(set(range(H)) - remove)
        return out

    cands = [row(note="baseline")]

    # --- single whole blocks (lowest BI); drop block j via skip=(j-1, j), so skip block 0 ---
    single = [j for j in sorted(block_bi, key=block_bi.get) if j >= 1][:N_SINGLE_BLOCKS]
    for j in single:
        cands.append(row(skip=[(j - 1, j)], skip_translator=translator_for_span(j - 1, j),
                         note=f"drop block {j}  BI={block_bi[j]:.3f}  cka={cka[j, j + 1]:.2f}"))

    # --- best contiguous multi-block spans ---
    for k in range(2, MAX_SPAN + 1):
        if k in span:
            s, e, d = span[k]
            cands.append(row(skip=[(s, e)], skip_translator=translator_for_span(s, e),
                             note=f"drop {k} blocks {s + 1}..{e}  angdist={d:.3f}  cka={cka[s + 1, e + 1]:.2f}"))

    mlp_rank = attn_rank = []
    # --- MLP sub-blocks ---
    if sub:
        mlp_rank = sorted(sub, key=lambda i: sub[i][2])
        top_mlp = mlp_rank[:N_MLP]
        for i in top_mlp:
            cands.append(row(mlp_skip=[i], mlp_mode="identity", note=f"mlp {i}  BI={sub[i][2]:.3f}"))
        cands.append(row(mlp_skip=sorted(top_mlp), mlp_mode="identity", note=f"mlp combined {sorted(top_mlp)}"))

        # --- attention sub-blocks ---
        attn_rank = sorted(sub, key=lambda i: sub[i][0])
        top_attn = attn_rank[:N_ATTN]
        for i in top_attn:
            cands.append(row(attn_skip=[i], attn_mode="identity", note=f"attn {i}  BI={sub[i][0]:.3f}"))
        cands.append(row(attn_skip=sorted(top_attn), attn_mode="identity", note=f"attn combined {sorted(top_attn)}"))

    # --- head pruning ---
    hd = head_keep_dict(HEAD_THRESHOLD)
    if hd:
        pruned = sum(head_sim.shape[1] - len(v) for v in hd.values())
        cands.append(row(head_dict=hd, note=f"heads @thr={HEAD_THRESHOLD}  ({pruned} pruned)"))

    # --- combined: safest single block + safest MLP + safest attn + heads ---
    if sub and single:
        b0 = single[0]
        cands.append(row(skip=[(b0 - 1, b0)], mlp_skip=[mlp_rank[0]], attn_skip=[attn_rank[0]],
                         head_dict=hd, skip_translator=translator_for_span(b0 - 1, b0),
                         mlp_mode="identity", attn_mode="identity", note="combined safest-of-each"))

    return pd.DataFrame(cands), L


def write_csv(df, out_csv, L, append):
    out = df[SCHEMA].copy()
    out_csv = Path(out_csv)

    # round-trip check: every literal parses back AND every skip index is a valid layer-output key
    for col in ["skip", "mlp_skip", "attn_skip", "head_dict"]:
        for v in out[col]:
            ast.literal_eval(v)
    for v in out["skip"]:
        for a, b in ast.literal_eval(v):
            assert 0 <= a < b <= L - 1, f"illegal skip ({a}, {b}) for {L}-block model"

    if append and out_csv.exists():
        out.to_csv(out_csv, mode="a", header=False, index=False, quoting=csv.QUOTE_MINIMAL)
        print(f"Appended {len(out)} rows -> {out_csv.resolve()}")
    else:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(out_csv, index=False, quoting=csv.QUOTE_MINIMAL)
        print(f"Wrote {len(out)} rows -> {out_csv.resolve()}")
    print(f"All literals parse; all skip pairs within 0..{L - 1}.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Turn layer-priority scores into candidate config rows")
    p.add_argument("--stem", required=True, help="{model}_{dataset} stem, e.g. raddino_dermamnist")
    p.add_argument("--dataset", required=True, help="dataset key written into the CSV")
    p.add_argument("--encoder", required=True, help="HF encoder id written into the CSV")
    p.add_argument("--output-dir", default="src/layers/outputs", help="dir holding the score files")
    p.add_argument("--out-csv", required=True, help="config CSV to write/append")
    p.add_argument("--append", action="store_true", help="append rows without a header (default: overwrite)")
    args = p.parse_args()

    df, L = build_candidates(args.stem, args.dataset, args.encoder, args.output_dir)
    write_csv(df, args.out_csv, L, args.append)
