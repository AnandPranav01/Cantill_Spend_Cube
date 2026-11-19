"""Config loading. A company = one YAML/JSON file. Paths are resolved relative to `data_dir`
(which itself is relative to the config file unless absolute), and can be overridden from the CLI."""
from __future__ import annotations

import json
import os
from copy import deepcopy
from pathlib import Path

import yaml

DEFAULT_CONFIG = {
    "name": "company",
    "reporting_currency": "USD",
    "data_dir": ".",
    "taxonomy": {"file": None, "level_columns": None, "delimiter": None},
    "labels": None,                       # optional: {file,id_column,level_columns,supplier_key_column,item_key_column,data_dir}
    "sources": [],
    "combine": None,                      # optional: two-system matching
    "exclusion_defaults": {"enabled": True, "extra_regex": []},
    "suppliers": {"fuzzy_threshold": 90, "use_labels": True},
    "items": {"fuzzy_threshold": 93, "use_labels": True},
    "classification": {
        "embeddings": "auto",             # auto | off | on  (sentence-transformers, only if installed)
        "embedding_model": "paraphrase-multilingual-MiniLM-L12-v2",
        "llm": {"enabled": "auto", "model": None, "max_unique": 1500, "conf_threshold": 0.55},
        "use_labels": True,
        "self_training_rounds": 2,
    },
    "output": {"file": "output.csv"},
}

SOURCE_DEFAULTS = {
    "role": "primary",
    "encoding": None,
    "delimiter": None,
    "id": {"column": None, "prefix": ""},
    "fields": {},
    "text_columns": [],
    "context_columns": [],
    "class_columns": [],                  # context columns (subset of context_columns) used as classification evidence
    "amount": {"column": None, "tax": {"mode": "net"}, "fx": None},
    "status": None,
    "exclusions": [],
    "date_formats": None,
    "exclusion_text_columns": None,
}


def _merge(base: dict, over: dict) -> dict:
    out = deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


class Config(dict):
    """dict with a few resolved attributes."""
    path: Path
    data_dir: Path

    def resolve(self, rel: str | None) -> Path | None:
        if rel is None:
            return None
        p = Path(rel)
        return p if p.is_absolute() else self.data_dir / p


def load_config(path: str | os.PathLike, data_dir: str | None = None, overrides: dict | None = None) -> Config:
    path = Path(path)
    with open(path, "r", encoding="utf-8") as fh:
        raw = json.load(fh) if path.suffix.lower() == ".json" else yaml.safe_load(fh)
    cfg = Config(_merge(DEFAULT_CONFIG, raw))
    cfg["sources"] = [_merge(SOURCE_DEFAULTS, s) for s in cfg["sources"]]
    cfg = Config(_merge(cfg, overrides or {}))
    cfg.path = path.resolve()
    dd = Path(data_dir) if data_dir else Path(cfg["data_dir"])
    cfg.data_dir = dd if dd.is_absolute() else (Path.cwd() / dd if data_dir else (cfg.path.parent / dd)).resolve()
    validate(cfg)
    return cfg


def validate(cfg: Config) -> None:
    if not cfg["sources"]:
        raise ValueError("config needs at least one entry under 'sources'")
    names = set()
    for s in cfg["sources"]:
        for req in ("name", "file"):
            if not s.get(req):
                raise ValueError(f"every source needs '{req}'")
        if s["name"] in names:
            raise ValueError(f"duplicate source name {s['name']}")
        names.add(s["name"])
        if not s["id"].get("column"):
            raise ValueError(f"source {s['name']}: id.column is required")
        if not s["amount"].get("column"):
            raise ValueError(f"source {s['name']}: amount.column is required")
    comb = cfg.get("combine")
    if comb:
        for k in ("primary", "secondary"):
            if comb.get(k) not in names:
                raise ValueError(f"combine.{k} must name a source ({sorted(names)})")
