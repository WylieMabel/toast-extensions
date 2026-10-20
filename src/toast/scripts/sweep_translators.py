"""Print translator directories a sweep CSV has finished with, one absolute path per line.

run_pipeline_row_by_row.sh calls this two ways:
  - after each row, with ROW_INDEX = that row's 0-based index: prints only the directories no
    *remaining* row in the sweep still needs, so they can be deleted right away instead of
    sitting on disk until the whole file finishes. This is what bounds a sweep's peak
    translator-directory count to however many spans are currently fit-but-not-yet-fully-
    consumed, rather than "every span the file will ever fit" -- the difference that matters
    once several small sweeps run concurrently and share the same disk.
  - after the whole sweep, with no ROW_INDEX: prints every directory the sweep saved, as a
    final safety net that also catches anything the per-row calls missed (e.g. a row that never
    ran because an earlier one failed).

Prints nothing (exit 0) when the sweep has no transfer rows -- the common case, where nothing
was saved to begin with.

Kept out of the heavy modules so the runner does not import torch just to list paths; the
naming rules themselves live in toast.utils.translator_keys and are not duplicated here.

Reads SEEDS from the environment (space-separated, default "0") to match
run_pipeline_row_by_row.sh's own SEEDS -- each seed gets its own translator directory (see
translator_store_key), so this has to enumerate every seed the sweep loops over, not just one,
or it will under-report what's saved / over-delete what's still needed.

    python src/toast/scripts/sweep_translators.py <sweep_csv> <samples> [row_index]
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, "src")
from toast import PROJECT_ROOT  # only imports pathlib -- no heavy deps
from toast.utils.translator_keys import (
    read_sweep_rows, sweep_saved_translator_dirs, translators_ready_for_cleanup,
)

if len(sys.argv) not in (3, 4):
    print(__doc__.strip().splitlines()[-1].strip(), file=sys.stderr)
    raise SystemExit(2)

csv_path, samples = sys.argv[1], int(sys.argv[2])
seeds = [int(s) for s in os.environ.get("SEEDS", "0").split()]
rows = read_sweep_rows(csv_path)
if len(sys.argv) == 4:
    names = translators_ready_for_cleanup(rows, samples, int(sys.argv[3]), seeds)
else:
    names = sweep_saved_translator_dirs(rows, samples, seeds)

base = PROJECT_ROOT / "data" / "translators"
for name in sorted(names):
    path = base / name
    if path.is_dir():
        print(path)
