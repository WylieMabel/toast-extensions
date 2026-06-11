"""Print the embedding directory path for the first row of a single-row config CSV.

Self-contained: inlines cfg_embedding_dir to avoid importing heavy deps (pytorch_lightning etc).
"""
import ast
import csv
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, "src")
from toast import PROJECT_ROOT  # only imports pathlib — no heavy deps


def cfg_embedding_dir(cfg: dict, samples: int, base) -> Path:
    head_dict = cfg.get("head_dict") or {}
    canonical = json.dumps({
        "dataset":         cfg.get("dataset"),
        "encoder":         cfg.get("encoder"),
        "skip":            str(cfg.get("skip") or []),
        "mlp_skip":        str(cfg.get("mlp_skip") or []),
        "attn_skip":       str(cfg.get("attn_skip") or []),
        "head_dict":       json.dumps(
            {str(k): sorted(v) for k, v in head_dict.items()}, sort_keys=True
        ),
        "skip_translator": cfg.get("skip_translator"),
        "mlp_mode":        cfg.get("mlp_mode", "identity"),
        "attn_mode":       cfg.get("attn_mode", "identity"),
        "samples":         samples,
    }, sort_keys=True)
    h = hashlib.md5(canonical.encode()).hexdigest()[:12]
    enc_slug = (cfg.get("encoder") or "unknown").split("/")[-1]
    return Path(base) / (cfg.get("dataset") or "unknown") / enc_slug / h


csv_path = sys.argv[1]
samples = int(sys.argv[2])

with open(csv_path) as f:
    row = list(csv.DictReader(f))[0]

cfg = {
    "dataset":         row.get("dataset") or None,
    "encoder":         row.get("encoder") or None,
    "skip":            ast.literal_eval(row["skip"]),
    "mlp_skip":        ast.literal_eval(row["mlp_skip"]),
    "attn_skip":       ast.literal_eval(row["attn_skip"]),
    "head_dict":       ast.literal_eval(row.get("head_dict", "{}")),
    "skip_translator": row.get("skip_translator") or None,
    "mlp_mode":        row.get("mlp_mode") or "identity",
    "attn_mode":       row.get("attn_mode") or "identity",
}

print(cfg_embedding_dir(cfg, samples, PROJECT_ROOT / "data" / "embeddings"))
