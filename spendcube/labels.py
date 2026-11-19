"""Optional labels (client's known categories / supplier keys / item keys).

Labels are never required. When present they (a) train the classifier, (b) name / merge supplier and item clusters.
`labels.data_dir` may point at the folder holding the labelled source files (e.g. the original data) while the
current run processes new records: the labelled lines are then loaded as extra, train-only rows."""
from __future__ import annotations

import copy
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from .ingest import _read_csv, load_source
from .taxonomy import Taxonomy
from .textutil import norm_text, ocr_fold


def fold_text(s) -> str:
    return ocr_fold(norm_text(s).upper()).lower()


class LabelBook:
    def __init__(self):
        self.other_dir = False
        self.table = None            # record_id -> leaf index (or -1), supplier_key, item_key
        self.extra_frames: list[pd.DataFrame] = []   # train-only frames (other data_dir)
        self.supplier_names = None
        self.supplier_labels = None
        self.item_labels = None
        self.item_by_code: dict = {}
        self.item_by_text: dict = {}

    def attach(self, frames: list[pd.DataFrame]) -> None:
        """Build the name/code -> label maps from every frame that contains labelled records."""
        names, sk = [], []
        votes_c, votes_t = defaultdict(Counter), defaultdict(Counter)
        t = self.table
        for df in frames:
            m = df[["record_id"]].merge(t, on="record_id", how="inner")
            if m.empty:
                continue
            d = df.set_index("record_id").loc[m["record_id"]]
            if "supplier_name" in d:
                nm = d["supplier_name"]
                ok = nm.notna() & m["supplier_key"].notna().values
                names.append(nm[ok]); sk.append(m.loc[ok.values, "supplier_key"])
            ik = m["item_key"].values
            for code, desc, k in zip(d["item_code"] if "item_code" in d else [None] * len(d), d["desc"], ik):
                if k is None or (isinstance(k, float) and np.isnan(k)):
                    continue
                if code is not None and str(code).strip():
                    votes_c[str(code).strip().upper()][k] += 1
                f = fold_text(desc)
                if len(f) >= 4:
                    votes_t[f][k] += 1
        if names:
            self.supplier_names = pd.concat(names, ignore_index=True)
            self.supplier_labels = pd.concat(sk, ignore_index=True)
        if votes_c or votes_t:
            self.item_labels = True
            # a text/code is only trusted when its votes agree
            self.item_by_code = {k: c.most_common(1)[0][0] for k, c in votes_c.items()}
            self.item_by_text = {k: c.most_common(1)[0][0] for k, c in votes_t.items() if len(c) == 1}


def load_labels(cfg, tax: Taxonomy, frames: dict[str, pd.DataFrame], log=print):
    lc = cfg.get("labels")
    if not lc or not lc.get("file"):
        return None
    other = Path(lc["data_dir"]).expanduser().resolve() if lc.get("data_dir") else None
    path = cfg.resolve(lc["file"]) if other is None else other / lc["file"]
    if not path.exists():
        log(f"[labels] file not found ({path}); continuing without labels")
        return None
    df = _read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    idc = lc.get("id_column", "record_id")
    lv = lc.get("level_columns") or [c for c in df.columns if c.lower().startswith("category_l")]
    book = LabelBook()
    idprefix = lc.get("id_prefix", "")
    rid = idprefix + df[idc].astype(str).str.strip()
    leaf = []
    for r in df[lv].itertuples(index=False, name=None):
        i = tax.find(tuple(r))
        leaf.append(-1 if i is None else i)
    t = pd.DataFrame({"record_id": rid, "leaf": leaf})
    for key, col in (("supplier_key", lc.get("supplier_key_column")), ("item_key", lc.get("item_key_column"))):
        t[key] = df[col].where(df[col].notna(), None).values if col and col in df.columns else None
    inc_col = lc.get("include_column")
    if inc_col and inc_col in df.columns:
        t = t[df[inc_col].astype(str).isin(["1", "1.0", "True", "true"]).values]
    book.table = t.drop_duplicates("record_id")
    log(f"[labels] {len(book.table)} labelled records; {(book.table.leaf < 0).sum()} with a category not found in the tree (ignored)")
    flist = list(frames.values())
    book.other_dir = other is not None
    if other is not None:
        c2 = copy.copy(cfg)
        c2.data_dir = other
        have = set(book.table["record_id"])
        for s in cfg["sources"]:
            try:
                f = load_source(c2, s)
            except Exception as e:      # labelled data not available: simply no extra training rows
                log(f"[labels] could not load labelled source {s['name']}: {e}")
                continue
            f = f[f["record_id"].isin(have)].reset_index(drop=True)
            if len(f):
                book.extra_frames.append(f)
        flist = list(book.extra_frames)
    book.attach(flist)
    return book
