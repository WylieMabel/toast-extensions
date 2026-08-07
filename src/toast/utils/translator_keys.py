"""Naming and lifetime rules for the fitted translators in data/translators.

Stdlib only, deliberately: run_pipeline_row_by_row.sh shells out to
scripts/sweep_translators.py between rows, and that must not pay for importing torch. Both
the heavy modules (modules.module, utils.encode_vision_full) import the key functions from
here rather than defining their own, so there is one definition to keep correct -- unlike
cfg_embedding_dir, which is copied into scripts/get_embed_dir.py and carries a "must stay in
lockstep" warning.

WHAT GETS SAVED
    A saved translator is only ever read back by a *transfer* row -- one whose fit_dataset
    differs from its dataset (see run_encoding). Every other row refits from its own
    activations and never consults the store, so saving for those rows writes bytes nothing
    reads.

    A fitting row therefore saves only when the sweep contains a transfer row that is
    identical to it in every respect except which dataset it runs on: same encoder, same
    skip spans, same mlp_skip/attn_skip/head_dict, same translator and modes, with that
    transfer row's fit_dataset naming the fitting row's dataset. config_signature is that
    "everything except the dataset" comparison.

    Matching the whole config rather than just the store key is the conservative choice.
    The store key omits skip, so on its own it would pair a row fitting span (0,1) with a
    transfer row wanting span (9,10) -- saving a translator that the transfer row then fails
    to find anyway. Requiring the full config makes such a pair save nothing, and any row
    that genuinely needs a translator still fails loudly in load_translator rather than
    silently reading a map fitted for a different span.
"""
import ast
import csv
import json


def translator_store_key(fit_dataset, encoder, skip_translator, samples) -> str:
    """Directory name identifying a fitted translator, independent of where it is used.

    Both sides of a transfer derive this from the *fitting* dataset, so the row that fits
    (fit_dataset == dataset) and the row that reuses (fit_dataset != dataset) agree on the
    key without any bookkeeping between them. The span is appended separately, per skip, by
    span_translator_key -- one config can hold several spans and each needs its own map.
    """
    enc_slug = (encoder or "unknown").replace("/", "_")
    return f"{fit_dataset}__{enc_slug}__{skip_translator}__n{samples}"


def span_translator_key(translator_key: str, skip_from: int, skip_to: int) -> str:
    """On-disk key for the translator bridging one span.

    The span has to be part of the key: a config with several skips fits a separate map per
    span, and without the suffix they would all be written to (and read from) one directory.
    """
    return f"{translator_key}_span{skip_from}_{skip_to}"


def read_sweep_rows(csv_path):
    with open(csv_path) as f:
        return list(csv.DictReader(f))


def _parse(value, default):
    """Accept either a raw CSV string or an already-parsed value.

    Rows reach these functions from two places: csv.DictReader (all strings) in
    sweep_translators.py, and _read_config_csv (lists and dicts already evaluated) in
    run_encoding. Both must produce the same signature or a sweep would delete a directory
    it did not save.
    """
    if value is None or value == "":
        return default
    if isinstance(value, str):
        return ast.literal_eval(value)
    return value


def config_signature(row) -> tuple:
    """Everything identifying a config except which dataset it runs on.

    Parsed and re-serialised rather than compared as raw text so that "[(9, 10)]" and
    "[(9,10)]" are the same signature.
    """
    head_dict = _parse(row.get("head_dict"), {}) or {}
    return (
        (row.get("encoder") or "").strip(),
        str(_parse(row.get("skip"), []) or []),
        str(_parse(row.get("mlp_skip"), []) or []),
        str(_parse(row.get("attn_skip"), []) or []),
        json.dumps({str(k): sorted(v) for k, v in head_dict.items()}, sort_keys=True),
        (row.get("skip_translator") or "").strip(),
        (row.get("mlp_mode") or "identity").strip(),
        (row.get("attn_mode") or "identity").strip(),
    )


def _row_fields(row):
    """(dataset, fit_dataset, translator) with the blank-means-same-dataset rule applied."""
    dataset = (row.get("dataset") or "").strip()
    fit_dataset = (row.get("fit_dataset") or "").strip() or dataset
    translator = (row.get("skip_translator") or "").strip()
    return dataset, fit_dataset, translator


def transfer_requirements(rows) -> set:
    """(fit_dataset, config_signature) pairs that some transfer row in this sweep will load."""
    requirements = set()
    for row in rows:
        dataset, fit_dataset, translator = _row_fields(row)
        if not translator or translator == "identity" or fit_dataset == dataset:
            continue
        requirements.add((fit_dataset, config_signature(row)))
    return requirements


def row_saves_translator(row, requirements) -> bool:
    """Whether this row should persist the translator it fits."""
    dataset, fit_dataset, translator = _row_fields(row)
    # Transfer rows load rather than save; identity has no state worth persisting.
    if not translator or translator == "identity" or fit_dataset != dataset:
        return False
    return (dataset, config_signature(row)) in requirements


def sweep_saved_translator_dirs(rows, samples) -> set:
    """Span directory names this sweep will write -- exactly what it may delete afterwards.

    Scoped to the sweep on purpose: the store is shared, so a blanket wipe would take out
    translators another sweep still depends on.
    """
    requirements = transfer_requirements(rows)
    dirs = set()
    for row in rows:
        if not row_saves_translator(row, requirements):
            continue
        _, fit_dataset, translator = _row_fields(row)
        key = translator_store_key(fit_dataset, row.get("encoder"), translator, samples)
        for skip_from, skip_to in _parse(row.get("skip"), []) or []:
            dirs.add(span_translator_key(key, skip_from, skip_to))
    return dirs
