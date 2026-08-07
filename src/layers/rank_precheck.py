"""Reconstruction error vs. translator rank, without running the accuracy pipeline.

Fits the three low-rank translator methods on real activations for one or more skip spans and
reports how well each reconstructs the target hidden state. Minutes on one GPU, versus hours
for the full encode + probe sweep.

Use it to decide which ranks are worth spending accuracy runs on: if error has already
flattened by r=64 there is no point running r=128 and r=256 at five seeds each. It also
validates the factorisations end-to-end on real data before they touch the pipeline -- the
full-rank case of both closed forms must reproduce the plain lstsq error exactly.

    python src/layers/rank_precheck.py --model dinobase --dataset imagenet-1k \
        --spans "[(5,6),(2,4),(0,1)]" --out src/layers/outputs/rank_precheck.csv

What it reports per (span, method, rank):
    rel_error    ||X W_r - Y||^2 / ||Y||^2 on the fit data
    params       2*d*r, the stored translator size
    vs_full      rel_error relative to the full-rank lstsq solution (1.0 = as good as linear)

Caveat: this is reconstruction error on the fitting data, not downstream accuracy. It ranks
ranks well and is a sound way to prune the grid, but the accuracy sweep is still the result --
a translator can reconstruct the hidden state well and still cost accuracy, and vice versa.
"""

import argparse
import ast
import csv
import sys
from pathlib import Path

import torch

sys.path.insert(0, "src")

from layer_priority import _load_dataset, _load_model, _resolve_hf_id, _resolve_layers  # noqa: E402
from toast.modules.lowrank_translator import (  # noqa: E402
    DEFAULT_RANKS,
    LowRankAligner,
    ReducedRankAligner,
    FactoredSGDAligner,
)
from toast.utils.dictionaries import MODEL2CONFIGS  # noqa: E402

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def collect_hidden_states(model, loader, max_samples, num_layers):
    """{layer_idx: [n_samples, n_tokens, d]} for the unmodified encoder.

    Mirrors utils.extract_representations but stays local so this script does not drag in the
    encode pipeline. hidden_states[0] is the embedding output, so it is dropped -- layer i
    here means "output of block i", matching the (skip_from, skip_to) convention.
    """
    collected = {i: [] for i in range(num_layers)}
    seen = 0
    with torch.no_grad():
        # layer_priority's collate yields a (pixel_values, labels) tuple, not a dict.
        for pixel_values, _ in loader:
            pixel_values = pixel_values.to(device)
            out = model(pixel_values, output_hidden_states=True, return_dict=True)
            for i, hs in enumerate(out.hidden_states[1:]):
                collected[i].append(hs.detach().float().cpu())
            seen += pixel_values.shape[0]
            if seen >= max_samples:
                break
    return {i: torch.cat(v)[:max_samples] for i, v in collected.items()}


def evaluate_span(x, y, ranks, methods, lora_steps):
    """Relative reconstruction error for each (method, rank), plus the full-rank reference."""
    x = x.reshape(-1, x.shape[-1]).to(device)
    y = y.reshape(-1, y.shape[-1]).to(device)
    denom = (y ** 2).sum().item()

    w_full = torch.linalg.lstsq(x, y).solution
    full_error = ((x @ w_full - y) ** 2).sum().item() / denom

    builders = {
        "lowrank": lambda r: LowRankAligner(rank=r),
        "rrr": lambda r: ReducedRankAligner(rank=r),
        "lora": lambda r: FactoredSGDAligner(rank=r, num_steps=lora_steps),
    }

    rows = []
    for method in methods:
        for r in ranks:
            aligner = builders[method](r).to(device)
            aligner.fit(x, y)
            pred, _ = aligner.transform(x)
            rel = ((pred - y) ** 2).sum().item() / denom
            rows.append({
                "method": method,
                "rank": r,
                "rel_error": rel,
                "params": aligner.num_parameters,
                "vs_full": rel / full_error if full_error else float("nan"),
            })
            del aligner
            torch.cuda.empty_cache() if device.type == "cuda" else None
    return rows, full_error, x.shape


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, help="alias or HF id, e.g. dinobase")
    p.add_argument("--dataset", required=True)
    p.add_argument("--spans", default="[(5,6),(2,4),(0,1)]",
                   help="list of (skip_from, skip_to) tuples")
    p.add_argument("--num-samples", type=int, default=250,
                   help="images; each contributes n_tokens rows to the fit")
    p.add_argument("--ranks", nargs="*", type=int, default=list(DEFAULT_RANKS))
    p.add_argument("--methods", nargs="*", default=["lowrank", "rrr", "lora"])
    p.add_argument("--lora-steps", type=int, default=300)
    p.add_argument("--out", default=None, help="CSV path; prints to stdout if omitted")
    args = p.parse_args()

    spans = ast.literal_eval(args.spans)
    hf_id = _resolve_hf_id(args.model)
    if hf_id not in MODEL2CONFIGS:
        raise SystemExit(f"No MODEL2CONFIGS entry for '{hf_id}'.")

    model, num_layers, _ = _load_model(hf_id)
    loader, n_available = _load_dataset(args.dataset, args.num_samples, hf_id)
    num_layers = len(_resolve_layers(model, hf_id))

    print(f"Collecting hidden states: {hf_id} x {args.dataset}, "
          f"{n_available} images, {num_layers} blocks", file=sys.stderr)
    states = collect_hidden_states(model, loader, args.num_samples, num_layers)

    all_rows = []
    for skip_from, skip_to in spans:
        if skip_from not in states or skip_to not in states:
            raise SystemExit(f"span ({skip_from}, {skip_to}) outside 0..{num_layers - 1}")
        rows, full_error, shape = evaluate_span(
            states[skip_from], states[skip_to], args.ranks, args.methods, args.lora_steps
        )
        print(f"\nspan ({skip_from}, {skip_to})  fit matrix {tuple(shape)}  "
              f"full-rank lstsq rel_error {full_error:.6f}", file=sys.stderr)
        print(f"  {'method':>8} {'rank':>5} {'rel_error':>11} {'vs_full':>9} {'params':>10}",
              file=sys.stderr)
        for r in rows:
            r["skip_from"], r["skip_to"] = skip_from, skip_to
            r["full_rank_error"] = full_error
            print(f"  {r['method']:>8} {r['rank']:>5} {r['rel_error']:>11.6f} "
                  f"{r['vs_full']:>9.3f} {r['params']:>10,}", file=sys.stderr)
        all_rows.extend(rows)

    fields = ["skip_from", "skip_to", "method", "rank", "rel_error", "vs_full",
              "params", "full_rank_error"]
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(all_rows)
        print(f"\n-> {args.out}", file=sys.stderr)
    else:
        w = csv.DictWriter(sys.stdout, fieldnames=fields)
        w.writeheader()
        w.writerows(all_rows)


if __name__ == "__main__":
    main()
