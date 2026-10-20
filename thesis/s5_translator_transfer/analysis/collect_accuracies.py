"""Merges the s5 accuracy summaries into one CSV per evaluation x donor combo. Run from the
project root once the s5 jobs are done:
    python thesis/s5_translator_transfer/analysis/collect_accuracies.py

Reads thesis/s5_translator_transfer/accuracies/accuracies_results_<combo>[_<model>_partN].csv
and writes final_folder/final_accuracies/transfer_learning/<combo>.csv, e.g.
group2_crossfit_tcgaut_from_patchcamelyon.csv. Each output holds the donor's own fitting rows
(fit_dataset == dataset) and the transfer rows on the evaluation dataset (fit_dataset == donor).
"""
import re
from pathlib import Path

import pandas as pd

ACCURACIES = Path("thesis/s5_translator_transfer/accuracies")
OUT = Path("final_folder/final_accuracies/transfer_learning")
NAME = re.compile(r"accuracies_results_(group\d_crossfit_[a-z0-9]+_from_[a-z0-9]+)(_.+)?\.csv")


def main():
    shards = {}
    for path in sorted(ACCURACIES.glob("accuracies_results_group*_crossfit_*.csv")):
        m = NAME.fullmatch(path.name)
        if m:
            shards.setdefault(m.group(1), []).append(path)
    if not shards:
        raise SystemExit(f"no s5 accuracy summaries in {ACCURACIES}")

    for combo, paths in shards.items():
        merged = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
        merged["order"] = range(len(merged))
        out = OUT / f"{combo}.csv"
        merged.to_csv(out, index=False)
        print(f"{out}: {len(merged)} rows from {len(paths)} file(s)")


if __name__ == "__main__":
    main()
