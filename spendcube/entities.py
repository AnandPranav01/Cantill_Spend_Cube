"""Supplier and item de-duplication.

Suppliers: a supplier id is trusted when present; otherwise names are normalised (case, punctuation, legal suffixes,
OCR look-alikes, stray spaces) and clustered with a strict fuzzy threshold.
Items: union of (same item code) and (same / near-identical normalised description). Near-identical means a high
fuzzy score AND identical numbers (so 'DIA 20MM' and 'DIA 25MM' are never merged).
Known labels (optional) only *name* the clusters and merge clusters the client has already merged."""
from __future__ import annotations

import hashlib
import re
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process

from .textutil import LEGAL_SUFFIXES, norm_text, ocr_fold, strip_accents


# ------------------------------------------------------------------------------------------ helpers
class UnionFind:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def _has(v) -> bool:
    """True for a real value (not None / NaN / NA / blank / 'nan')."""
    if v is None or v is pd.NA:
        return False
    if isinstance(v, float) and np.isnan(v):
        return False
    t = str(v).strip()
    return t != "" and t.lower() not in ("nan", "none", "<na>", "null")


def _h(prefix: str, s: str) -> str:
    return prefix + hashlib.md5(s.encode("utf-8")).hexdigest()[:8]


def supplier_compact(name) -> str:
    """Matching key for a supplier name."""
    if name is None or (isinstance(name, float) and np.isnan(name)):
        return ""
    s = strip_accents(str(name)).upper()
    s = re.sub(r"[^A-Z0-9 ]", " ", s)
    toks = [t for t in s.split() if t not in LEGAL_SUFFIXES]
    toks = [ocr_fold(t) if re.search(r"[A-Z]", t) else t for t in toks]
    return "".join(toks)


# ------------------------------------------------------------------------------------------ suppliers
def cluster_names(keys: pd.Series, threshold: float = 90) -> pd.Series:
    """keys: compact names (may be ''). Returns cluster rep (a compact name) per row."""
    cnt = Counter(k for k in keys if k)
    reps: list[str] = []
    assign: dict[str, str] = {}
    for k, _ in sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0])):
        best = None
        if reps and len(k) >= 5:
            m = process.extractOne(k, reps, scorer=fuzz.ratio, score_cutoff=threshold)
            if m:
                cand = m[0]
                if abs(len(cand) - len(k)) <= max(3, 0.2 * len(k)):
                    best = cand
        if best is None:
            reps.append(k)
            assign[k] = k
        else:
            assign[k] = best
    return keys.map(lambda k: assign.get(k, "") if k else "")


def assign_supplier_keys(frames: list[pd.DataFrame], cfg_sup: dict, book=None) -> list[pd.Series]:
    """Returns one supplier_key Series per frame (aligned to each frame's index).
    Evidence used: normalised-name similarity, shared supplier id; labels (optional) only name / merge clusters."""
    thr = float(cfg_sup.get("fuzzy_threshold", 90))
    comp_parts, id_parts = [], []
    for df in frames:
        name = df["supplier_name"] if "supplier_name" in df and df["supplier_name"] is not None else pd.Series([None] * len(df), index=df.index)
        comp_parts.append(name.map(supplier_compact))
        sid = df["supplier_id"] if "supplier_id" in df and df["supplier_id"] is not None else pd.Series([None] * len(df), index=df.index)
        id_parts.append(sid)
    allc = pd.concat(comp_parts, ignore_index=True)
    allid = pd.concat(id_parts, ignore_index=True).astype("object")
    rep = cluster_names(allc, thr)
    # union name-clusters that share a supplier id (the same vendor typed under several names)
    nodes = {}
    def node(x):
        if x not in nodes:
            nodes[x] = len(nodes)
        return nodes[x]
    pairs = []
    for r, i in zip(rep, allid):
        if r and _has(i):
            pairs.append((node("N:" + r), node("I:" + str(i).strip())))
    uf = UnionFind(max(len(nodes), 1))
    for a, b in pairs:
        uf.union(a, b)
    comp_root = {}
    for k, v in nodes.items():
        comp_root.setdefault(uf.find(v), []).append(k)
    final = []
    for r, i in zip(rep, allid):
        if r:
            final.append("N:" + r if ("N:" + r) not in nodes else "G%d" % uf.find(nodes["N:" + r]))
        elif _has(i):
            final.append("G%d" % uf.find(nodes["I:" + str(i).strip()]) if ("I:" + str(i).strip()) in nodes else "I:" + str(i).strip())
        else:
            final.append(None)
    final = pd.Series(final, dtype=object)
    # canonical representative text per group (for hashing)
    first_name = {}
    for g, r in zip(final, rep):
        if g is not None and g not in first_name and r:
            first_name[g] = r
    # label-driven naming / merging
    name2label: dict = {}
    if book is not None and book.supplier_names is not None and len(book.supplier_names):
        bn = book.supplier_names.map(supplier_compact)
        tmp = pd.DataFrame({"c": bn.values, "k": book.supplier_labels.values}).dropna()
        tmp = tmp[tmp["c"] != ""]
        known = {}
        for g, r in zip(final, rep):
            if g is not None and r:
                known.setdefault(g, set()).add(r)
        reps_all = sorted(set(rep) - {""})
        votes: dict = defaultdict(Counter)
        for c, k in zip(tmp["c"], tmp["k"]):
            m = c if c in set(reps_all) else None
            if m is None and reps_all:
                mm = process.extractOne(c, reps_all, scorer=fuzz.ratio, score_cutoff=thr)
                m = mm[0] if mm else None
            if m is not None:
                votes[m][k] += 1
        rep2g = {}
        for g, r in zip(final, rep):
            if g is not None and r:
                rep2g[r] = g
        for r, cnt in votes.items():
            g = rep2g.get(r)
            if g is not None:
                name2label.setdefault(g, Counter()).update(cnt)
    out_all = []
    for g in final:
        if g is None:
            out_all.append(None)
        elif g in name2label:
            out_all.append(name2label[g].most_common(1)[0][0])
        else:
            out_all.append(_h("S-", first_name.get(g, g)))
    out_all = pd.Series(out_all, dtype=object)
    res, pos = [], 0
    for df in frames:
        part = out_all.iloc[pos:pos + len(df)].values
        pos += len(df)
        res.append(pd.Series(part, index=df.index, dtype=object).fillna("UNKNOWN"))
    return res


_rep_cache: dict = {}


def _nearest_rep(c: str, reps: pd.Series, thr: float):
    uniq = _rep_cache.get(id(reps))
    if uniq is None or uniq[0] is not reps:
        _rep_cache[id(reps)] = uniq = (reps, sorted(set(r for r in reps if r)))
    cand = uniq[1]
    if c in cand:
        return c
    m = process.extractOne(c, cand, scorer=fuzz.ratio, score_cutoff=thr)
    return m[0] if m else None


# ------------------------------------------------------------------------------------------ items
_NUM = re.compile(r"\d+(?:[.,]\d+)?")


def _num_sig(s: str) -> str:
    return " ".join(sorted(_NUM.findall(s)))


def item_text(s) -> str:
    return norm_text(s)


def assign_item_keys(df: pd.DataFrame, cfg_items: dict, book=None) -> pd.Series:
    """df: concatenation of all frames (needs desc, item_code). Returns item_key per row."""
    thr = float(cfg_items.get("fuzzy_threshold", 93))
    max_group = int(cfg_items.get("max_text_group_codes", 300))
    txt = df["desc"].map(item_text)
    code = df["item_code"].astype("object").map(lambda v: str(v).strip() if _has(v) else None) if "item_code" in df else pd.Series([None] * len(df), index=df.index, dtype=object)
    n = len(df)
    uf = UnionFind(n)
    meaningful = txt.map(lambda t: len(t) >= 4 and len(t.split()) >= 1 and not t.isdigit())

    # 1. identical item code
    pos = {i: k for k, i in enumerate(df.index)}
    byc = defaultdict(list)
    for i, c in zip(df.index, code):
        if _has(c):
            byc[str(c).strip().upper()].append(pos[i])
    for ids in byc.values():
        for j in ids[1:]:
            uf.union(ids[0], j)
    # 2. identical OCR-folded text
    folded = txt.map(lambda t: ocr_fold(t.upper()).lower())
    bt = defaultdict(list)
    for i, (t, f, ok) in zip(df.index, zip(txt, folded, meaningful)):
        if ok:
            bt[f].append(pos[i])
    # ignore text keys shared by a huge number of different codes (generic words)
    for f, ids in bt.items():
        if len({str(code.iloc[j]) for j in ids if _has(code.iloc[j])}) > max_group:
            continue
        for j in ids[1:]:
            uf.union(ids[0], j)
    # 3. fuzzy near-duplicates of text, blocked by identical numbers
    uniq = pd.Series(list(bt.keys()))
    sig = uniq.map(_num_sig)
    for _, grp in uniq.groupby(sig):
        g = grp.tolist()
        if len(g) < 2:
            continue
        if len(g) > 4000:
            g = g[:4000]
        M = process.cdist(g, g, scorer=fuzz.token_sort_ratio, score_cutoff=thr, dtype=np.uint8, workers=-1)
        ii, jj = np.nonzero(np.triu(M, 1))
        for a, b in zip(ii, jj):
            uf.union(bt[g[a]][0], bt[g[b]][0])
    root = pd.Series([uf.find(k) for k in range(n)], index=df.index)
    # key per component: label key (if known) else hash of the most common folded text / code
    keys = pd.Series(index=df.index, dtype=object)
    lab_code = lab_text = None
    if book is not None and book.item_labels is not None:
        lab_code, lab_text = book.item_by_code, book.item_by_text
    for r, idx in root.groupby(root).groups.items():
        idx = list(idx)
        k = None
        if lab_code is not None:
            votes = Counter()
            for i in idx:
                c = code.at[i]
                if _has(c) and str(c).strip().upper() in lab_code:
                    votes[lab_code[str(c).strip().upper()]] += 1
                f = folded.at[i]
                if f in lab_text:
                    votes[lab_text[f]] += 1
            if votes:
                k = votes.most_common(1)[0][0]
        if k is None:
            tcount = Counter(folded.at[i] for i in idx if meaningful.at[i])
            ccount = Counter(str(code.at[i]).strip().upper() for i in idx if _has(code.at[i]))
            if tcount:
                k = _h("I-", tcount.most_common(1)[0][0])
            elif ccount:
                k = _h("C-", ccount.most_common(1)[0][0])
            else:
                k = None
        keys.loc[idx] = k
    return keys
