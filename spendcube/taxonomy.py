"""Category tree loading. The tree is a file read at run time; nothing about its categories lives in the code.

Supported layouts (auto-detected, or set under `taxonomy:` in the config):
  * wide   : one column per level (category_l1, category_l2, ...), blanks below the leaf's level
  * path   : one column holding the full path, split on a delimiter (`path_column`, `delimiter`)
Optional `description_column` adds free text to a category's text (used for matching only)."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .ingest import _read_csv
from .textutil import norm_text


class Taxonomy:
    def __init__(self, paths: list[tuple], descriptions: dict | None = None):
        # keep only leaves: a path that is a strict prefix of another path is an internal node
        ps = list(dict.fromkeys(paths))
        pref = set()
        for p in ps:
            for k in range(1, len(p)):
                pref.add(p[:k])
        self.paths = [p for p in ps if p not in pref]
        self.nlevels = max(len(p) for p in self.paths)
        self.desc = descriptions or {}
        self._index = {p: i for i, p in enumerate(self.paths)}
        self._norm = {tuple(norm_text(x) for x in p): i for i, p in enumerate(self.paths)}
        self._leaf_name = {}
        for i, p in enumerate(self.paths):
            self._leaf_name.setdefault(norm_text(p[-1]), []).append(i)

    def __len__(self):
        return len(self.paths)

    def text(self, i: int) -> str:
        p = self.paths[i]
        names = [p[0]] + [x for a, x in zip(p, p[1:]) if norm_text(a) != norm_text(x)]
        t = " > ".join(names)
        d = self.desc.get(p)
        return f"{t}. {d}" if d else t

    def leaf_phrase(self, i: int) -> str:
        """Leaf name repeated with its parents (for the sparse matcher)."""
        p = self.paths[i]
        return " ".join([p[-1]] * 3 + list(p[:-1]) + [self.desc.get(p, "")])

    def find(self, path: tuple):
        """Index of a leaf given a (possibly prefix / differently cased) path, else None."""
        path = tuple(x for x in path if x not in (None, "") and str(x).strip() != "")
        if not path:
            return None
        if path in self._index:
            return self._index[path]
        n = tuple(norm_text(x) for x in path)
        if n in self._norm:
            return self._norm[n]
        # a label stopping above the leaf level, or only the leaf name: accept when unambiguous
        c = [i for i, p in enumerate(self.paths) if tuple(norm_text(x) for x in p[:len(path)]) == n]
        if len(c) == 1:
            return c[0]
        c = self._leaf_name.get(n[-1], [])
        return c[0] if len(c) == 1 else None

    def level_names(self, i: int, nout: int = 4) -> list[str]:
        p = list(self.paths[i])[:nout]
        return p + [""] * (nout - len(p))


def _auto_level_columns(cols: list[str]) -> list[str]:
    pat = re.compile(r"^(?:category[_ ]*|cat[_ ]*)?(?:l|lvl|level)[_ ]*(\d+)$", re.I)
    found = sorted(((int(m.group(1)), c) for c in cols if (m := pat.match(c.strip()))))
    return [c for _, c in found]


def load_taxonomy(cfg) -> Taxonomy:
    tc = cfg["taxonomy"]
    path = cfg.resolve(tc["file"])
    if path is None or not path.exists():
        raise FileNotFoundError(f"taxonomy file not found: {path}")
    df = _read_csv(path, None, tc.get("delimiter"))
    df.columns = [c.strip() for c in df.columns]
    desc_col = tc.get("description_column")
    id_c = tc.get("id_column") or next((c for c in df.columns if re.fullmatch(r"(category|cat|node)?[_ ]*id", c, re.I)), None)
    par_c = tc.get("parent_column") or next((c for c in df.columns if re.search(r"parent", c, re.I)), None)
    name_c = tc.get("name_column") or next((c for c in df.columns if re.fullmatch(r"(category[_ ]*)?(name|title|label)", c, re.I)), None)
    if id_c and par_c and name_c and not tc.get("path_column") and not tc.get("level_columns"):
        # adjacency list (id, parent_id, name): rebuild each node's path by walking up
        nm = dict(zip(df[id_c].astype(str).str.strip(), df[name_c].astype(str).str.strip()))
        pr = {str(i).strip(): (None if (isinstance(p, float) and np.isnan(p)) or str(p).strip() in ("", "0", "nan", "None") else str(p).strip())
              for i, p in zip(df[id_c], df[par_c])}
        rows = []
        for i in nm:
            path, cur, seen = [], i, set()
            while cur is not None and cur in nm and cur not in seen:
                seen.add(cur)
                path.append(nm[cur])
                cur = pr.get(cur)
            rows.append(tuple(reversed(path)))
    elif tc.get("path_column"):
        dl = tc.get("path_delimiter", ">")
        rows = [tuple(x.strip() for x in str(v).split(dl) if x.strip()) for v in df[tc["path_column"]].fillna("")]
    else:
        cols = tc.get("level_columns") or _auto_level_columns(list(df.columns))
        if len(df.columns) == 1 and not cols:   # one column holding "A > B > C" style paths
            col0 = df.columns[0]
            sample = " ".join(df[col0].dropna().astype(str).head(50))
            dl = next((d for d in (">", "|", "/", ";", "::") if d in sample), None)
            if dl:
                cols = None
                rows = [tuple(x.strip() for x in str(v).split(dl) if x.strip()) for v in df[col0].fillna("")]
        if cols is None:
            pass
        elif not cols:   # fall back: every column except an obvious description column, in file order
            cols = [c for c in df.columns if c != desc_col and not re.search(r"desc|code|^id$", c, re.I)]
        if cols is not None:
            rows = [tuple(str(v).strip() for v in r if v is not None and not (isinstance(v, float) and np.isnan(v)) and str(v).strip() != "")
                    for r in df[cols].itertuples(index=False, name=None)]
    desc = {}
    if desc_col and desc_col in df.columns:
        for r, d in zip(rows, df[desc_col]):
            if isinstance(d, str) and d.strip():
                desc[r] = d.strip()
    rows = [r for r in rows if r]
    if not rows:
        raise ValueError("taxonomy file produced no categories; check taxonomy.level_columns")
    return Taxonomy(rows, desc)
