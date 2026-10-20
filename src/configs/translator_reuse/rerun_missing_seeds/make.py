"""One-off generator for the 5 rows that came up short after the raddino_cross /
group2_cross_fit runs finished (checked: nothing still running, so anything short of 3/3
seeds is a failed row, not a straggler). Each file below is a self-contained donor+consumer
pair for one (encoder, skip) -- different encoders/spans across the 5 files, so their
embedding dirs and translator keys never collide and all 5 can be submitted in parallel.

Reruns all 3 seeds per row (not just the missing seed) so the translator gets refit cleanly --
it was deleted at the end of the original sweep. The extra already-good seeds just produce
duplicate rows in these new results files; dedupe (keep latest) when merging.
"""
import csv
from pathlib import Path

COLUMNS = [
    "dataset", "encoder", "skip", "mlp_skip", "attn_skip",
    "head_dict", "skip_translator", "mlp_mode", "attn_mode", "fit_dataset",
]


def row(dataset, encoder, skip, fit_dataset):
    return {
        "dataset": dataset, "encoder": encoder, "skip": skip,
        "mlp_skip": "[]", "attn_skip": "[]", "head_dict": "{}",
        "skip_translator": "linear", "mlp_mode": "identity", "attn_mode": "identity",
        "fit_dataset": fit_dataset,
    }


def write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, quoting=csv.QUOTE_NONNUMERIC)
        w.writeheader()
        w.writerows(rows)


OUT = Path(__file__).parent

# raddino_cross_fit_chestxray14, skip (10,11): donor row was short 1 seed, consumer row
# never ran at all.
write(OUT / "experiments_raddinomaira2_skip_10_11.csv", [
    row("chestxray14", "microsoft/rad-dino-maira-2", "[(10, 11)]", "chestxray14"),
    row("chestxray_pneumonia", "microsoft/rad-dino-maira-2", "[(10, 11)]", "chestxray14"),
])

# group2_cross_fit_tcga_ut, phikon-v2, patchcamelyon consumer short 1 seed on 4 spans.
for a, b in [(1, 2), (11, 12), (7, 8), (9, 10)]:
    write(OUT / f"experiments_phikonv2_skip_{a}_{b}.csv", [
        row("tcga_ut", "owkin/phikon-v2", f"[({a}, {b})]", "tcga_ut"),
        row("patchcamelyon", "owkin/phikon-v2", f"[({a}, {b})]", "tcga_ut"),
    ])

print("wrote 5 files under", OUT)
