"""Print the translator directories a sweep CSV will save, one absolute path per line.

run_pipeline_row_by_row.sh calls this after the sweep finishes to delete exactly what the
sweep created, leaving other sweeps' translators alone. Prints nothing (exit 0) when the
sweep has no transfer rows -- the common case, where nothing was saved to begin with.

Kept out of the heavy modules so the runner does not import torch just to list paths; the
naming rules themselves live in toast.utils.translator_keys and are not duplicated here.

    python src/toast/scripts/sweep_translators.py <sweep_csv> <samples>
"""
import sys
from pathlib import Path

sys.path.insert(0, "src")
from toast import PROJECT_ROOT  # only imports pathlib -- no heavy deps
from toast.utils.translator_keys import read_sweep_rows, sweep_saved_translator_dirs

if len(sys.argv) != 3:
    print(__doc__.strip().splitlines()[-1].strip(), file=sys.stderr)
    raise SystemExit(2)

csv_path, samples = sys.argv[1], int(sys.argv[2])
base = PROJECT_ROOT / "data" / "translators"

for name in sorted(sweep_saved_translator_dirs(read_sweep_rows(csv_path), samples)):
    path = base / name
    if path.is_dir():
        print(path)
