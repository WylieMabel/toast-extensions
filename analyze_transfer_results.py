#!/usr/bin/env python3
"""Analyze transfer learning results: compare source-only vs transfer performance.

Usage:
    python analyze_transfer_results.py --results results_transfer_learning_part{1,2,3,4}.csv
    python analyze_transfer_results.py --results results_transfer_learning*.csv --output summary.csv
"""

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path


def load_results(csv_paths):
    """Load results from one or more CSV files."""
    results = defaultdict(dict)

    for path in csv_paths:
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                if not row.get("dataset") or not row.get("encoder"):
                    continue

                # Build a key that includes the skip pattern and fit_dataset
                key = (
                    row["dataset"],
                    row["encoder"],
                    row.get("skip", "[]"),
                    row.get("fit_dataset", ""),
                )
                results[key] = {
                    "acc": float(row.get("test_accuracy", 0)),
                    "fit_dataset": row.get("fit_dataset", ""),
                    "skip": row.get("skip", "[]"),
                }

    return results


def transfer_analysis(results):
    """Compare source-only vs transfer performance for each pattern."""

    # Group by (dataset, encoder, skip, target_fit_dataset)
    comparisons = defaultdict(dict)

    for (source_ds, encoder, skip, fit_ds), data in results.items():
        key = (source_ds, encoder, skip)
        comparisons[key][fit_ds] = data["acc"]

    # Compute transfer losses
    transfer_losses = []

    for (source_ds, encoder, skip), accuracies in comparisons.items():
        source_acc = accuracies.get("", None)
        if source_acc is None:
            continue

        for fit_ds, fit_acc in accuracies.items():
            if fit_ds == "":  # Skip source-only
                continue

            loss = fit_acc - source_acc
            transfer_losses.append({
                "source_dataset": source_ds,
                "fit_dataset": fit_ds,
                "encoder": encoder,
                "skip": skip,
                "source_accuracy": source_acc,
                "transfer_accuracy": fit_acc,
                "transfer_loss": loss,
                "transfer_loss_pct": (loss / source_acc * 100) if source_acc > 0 else 0,
            })

    return transfer_losses


def rank_sources_by_transfer(transfer_losses):
    """Rank source datasets by average transfer performance to each target."""

    rankings = defaultdict(lambda: defaultdict(list))

    for item in transfer_losses:
        fit_ds = item["fit_dataset"]
        source_ds = item["source_dataset"]
        rankings[fit_ds][source_ds].append(item["transfer_accuracy"])

    summary = []
    for fit_ds in sorted(rankings.keys()):
        for source_ds in sorted(rankings[fit_ds].keys()):
            accs = rankings[fit_ds][source_ds]
            avg_acc = sum(accs) / len(accs) if accs else 0
            summary.append({
                "fit_dataset": fit_ds,
                "source_dataset": source_ds,
                "avg_transfer_accuracy": avg_acc,
                "num_patterns": len(accs),
                "min_accuracy": min(accs),
                "max_accuracy": max(accs),
            })

    return sorted(summary, key=lambda x: (x["fit_dataset"], -x["avg_transfer_accuracy"]))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", nargs="+", required=True,
                   help="Results CSV file(s) to analyze")
    p.add_argument("--output", default=None,
                   help="Output summary CSV (default: print to stdout)")
    p.add_argument("--transfer-losses", default=None,
                   help="Output detailed transfer loss CSV")
    p.add_argument("--ranking", default=None,
                   help="Output source dataset ranking by transfer performance")
    args = p.parse_args()

    # Handle wildcards
    csv_paths = []
    for pattern in args.results:
        if "*" in pattern:
            csv_paths.extend(sorted(Path(".").glob(pattern)))
        else:
            csv_paths.append(Path(pattern))

    print(f"Loading {len(csv_paths)} file(s)...", file=sys.stderr)
    results = load_results(csv_paths)
    print(f"Loaded {len(results)} result rows", file=sys.stderr)

    # Compute transfer analysis
    transfer_losses = transfer_analysis(results)
    print(f"Computed transfer metrics for {len(transfer_losses)} comparisons", file=sys.stderr)

    # Compute rankings
    rankings = rank_sources_by_transfer(transfer_losses)

    # Output rankings
    if args.ranking:
        with open(args.ranking, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=[
                "fit_dataset", "source_dataset", "avg_transfer_accuracy",
                "num_patterns", "min_accuracy", "max_accuracy"
            ])
            w.writeheader()
            w.writerows(rankings)
        print(f"Wrote ranking to {args.ranking}", file=sys.stderr)
    else:
        # Print summary to stdout
        print("\nSource Dataset Rankings (by avg transfer accuracy):")
        print("=" * 80)
        for fit_ds in ["chestmnist", "pneumoniamnist"]:
            print(f"\nWhen fitting to {fit_ds}:")
            for item in rankings:
                if item["fit_dataset"] == fit_ds:
                    print(f"  {item['source_dataset']:15s}: {item['avg_transfer_accuracy']:.4f} "
                          f"(n={item['num_patterns']}, range=[{item['min_accuracy']:.4f}, {item['max_accuracy']:.4f}])")

    # Output transfer losses
    if args.transfer_losses:
        with open(args.transfer_losses, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=[
                "source_dataset", "fit_dataset", "encoder", "skip",
                "source_accuracy", "transfer_accuracy", "transfer_loss", "transfer_loss_pct"
            ])
            w.writeheader()
            w.writerows(sorted(transfer_losses,
                             key=lambda x: (x["fit_dataset"], x["transfer_loss"], x["source_dataset"])))
        print(f"Wrote transfer losses to {args.transfer_losses}", file=sys.stderr)

    # Output summary statistics
    if args.output:
        with open(args.output, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=[
                "fit_dataset", "source_dataset", "avg_transfer_accuracy",
                "num_patterns", "min_accuracy", "max_accuracy"
            ])
            w.writeheader()
            w.writerows(rankings)
        print(f"Wrote summary to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
