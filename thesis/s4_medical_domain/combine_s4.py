"""Combine the s4 per-job results CSVs into one file per model x dataset, then summarise them.

The sweep was split into several jobs per combo (_len1.._len4, _len12/_len34, _len1_rerun), each
writing its own results/accuracies CSV, and only the first part has the baseline row. This script
concatenates the parts and writes
    results/combined/results_<alias>_<tag>.csv               every (span, seed) row; on a duplicate the later file wins (_len1_rerun over _len1)
    accuracies/combined/accuracies_results_<alias>_<tag>.csv mean/std per span (calculate_accuracies.py), only spans with all 3 seeds
The raw job outputs stay where they are, so re-syncing results/ from the cluster never clobbers
the combined files. Re-run this after every sync.

Six results CSVs were overwritten on the cluster (2026-10-14/15) with copies taken mid-run on
2026-10-13 ~18:52, so their per-seed rows are partly gone. The per-job accuracies CSVs were written
from the full results at the end of each job and are intact (checked against the per-seed
accuracies in the job logs). For any span that does not have all 3 seeds in results, its row is
taken from those per-job accuracies CSVs instead (later file wins, as above).

Each combo is checked against its configs. Spans still missing or short of seeds after that are
left out of the accuracies file and listed. The exit status is 1 if any combo is incomplete.

    python thesis/s4_medical_domain/combine_s4.py
"""

import sys
from pathlib import Path

import pandas as pd

S4 = Path(__file__).resolve().parent
sys.path.insert(0, str(S4.parents[1] / "skipping_heads"))
from calculate_accuracies import accuracies  # noqa: E402

N_SEEDS = 3
# chestxray14multilabel: one job per encoder (launch_chest14ml.sh). Its accuracy_* columns hold
# macro-AUROC, as train_NN.py reports for multilabel datasets.
COMBOS = {
    "dinov2base": ["chestxray14", "chestxraypneumonia", "chestxray14multilabel"],
    "raddino": ["chestxray14", "chestxraypneumonia", "chestxray14multilabel"],
    "raddinomaira2": ["chestxray14", "chestxraypneumonia", "chestxray14multilabel"],
    "vitlarge": ["patchcamelyon", "tcgaut"],
    "phikonv2": ["patchcamelyon", "tcgaut"],
}


def parts(folder, prefix, alias, tag):
    """The combo's own job files: <prefix>_<alias>_<tag>.csv and <prefix>_<alias>_<tag>_len*.csv
    (not _chestxray14multilabel*, smoke or check runs). Sorted so _len1_rerun comes after _len1."""
    exact = folder / f"{prefix}_{alias}_{tag}.csv"
    return ([exact] if exact.exists() else []) + sorted(folder.glob(f"{prefix}_{alias}_{tag}_len*.csv"))


def main():
    (S4 / "results" / "combined").mkdir(exist_ok=True)
    (S4 / "accuracies" / "combined").mkdir(exist_ok=True)
    all_complete = True

    for alias, tags in COMBOS.items():
        for tag in tags:
            name = f"{alias}_{tag}"
            expected = set()
            for c in parts(S4 / "configs", "experiments", alias, tag):
                expected |= set(pd.read_csv(c)["skip"])

            files = parts(S4 / "results", "results", alias, tag)
            if not files:
                print(f"{name:34s} NO RESULTS")
                all_complete = False
                continue
            res = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
            res = res.drop_duplicates(subset=["approx_layer", "seed"], keep="last")
            res.to_csv(S4 / "results" / "combined" / f"results_{name}.csv", index=False)

            seeds = res.groupby("approx_layer")["seed"].nunique()
            partial = sorted(seeds.index[seeds < N_SEEDS])
            missing = sorted(expected - set(seeds.index))
            extra = sorted(set(seeds.index) - expected)

            # Only spans with every seed go into the summary, so no mean is over fewer seeds.
            full = res[res["approx_layer"].isin(seeds.index[seeds == N_SEEDS])]
            acc_path = S4 / "accuracies" / "combined" / f"accuracies_results_{name}.csv"
            acc = accuracies(full, acc_path)

            # Spans whose per-seed rows were lost: take the summary row the job itself wrote.
            job_acc = pd.concat([pd.read_csv(f) for f in parts(S4 / "accuracies", "accuracies_results", alias, tag)],
                                ignore_index=True).drop_duplicates(subset="approx_layer", keep="last")
            fill = job_acc[~job_acc["approx_layer"].isin(acc["approx_layer"])
                           & job_acc["approx_layer"].isin(expected)
                           & job_acc["accuracy_std"].notna()]
            if len(fill):
                acc = pd.concat([acc, fill[acc.columns]], ignore_index=True)
                acc.to_csv(acc_path, index=False)
            partial = [s for s in partial if s not in set(fill["approx_layer"])]
            missing = [s for s in missing if s not in set(fill["approx_layer"])]

            ok = not (partial or missing) and (acc["approx_layer"] == "[]").any()
            all_complete &= ok
            print(f"{name:34s} {len(acc):3d}/{len(expected):3d} spans  {'OK' if ok else 'INCOMPLETE'}"
                  f"   ({len(files)} files: {', '.join(f.stem.removeprefix('results_' + name) or '-' for f in files)})")
            if len(fill):
                print(f"    {len(fill)} taken from the per-job accuracies CSVs (per-seed rows missing from results)")
            if partial:
                print(f"    {len(partial)} with < {N_SEEDS} seeds: {', '.join(partial)}")
            if missing:
                print(f"    {len(missing)} not run: {', '.join(missing)}")
            if extra:
                print(f"    {len(extra)} not in configs (kept): {', '.join(extra)}")

    print("\nall combos complete" if all_complete else "\nsome combos are INCOMPLETE (see above)")
    sys.exit(0 if all_complete else 1)


if __name__ == "__main__":
    main()
