"""Combine two systems and count each purchase once.

Primary = system of record (ERP goods receipts). Secondary = invoice tool.
 1. Re-exports: the same invoice line exported more than once -> keep the first, rest are DUPLICATE.
 2. Match every remaining secondary line to at most one primary line:
      * candidates from the (normalised) PO reference, and from the amount for lines with no / wrong PO;
      * a pair is accepted when the net amount OR the unit price agrees (weighed goods and partial invoices change
        quantity and amount but not the unit price), with supplier and text agreement guarding no-PO matches.
 3. Matched secondary lines are DUPLICATE (the primary record is counted); unmatched ones are real spend that
    exists only in the secondary system and stay included.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
from rapidfuzz import fuzz

from .textutil import norm_text, ocr_fold


def _po_norm(series: pd.Series, pattern: str) -> pd.Series:
    rx = re.compile(pattern)

    def f(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return None
        m = rx.search(str(v))
        return m.group(0) if m else None
    return series.map(f)


def _po_item_ref(series: pd.Series) -> pd.Series:
    def f(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return np.nan
        m = re.search(r"/\s*(\d+)", str(v))
        return float(m.group(1)) if m else np.nan
    return series.map(f)


def _fold(s: str) -> str:
    return ocr_fold(norm_text(s).upper()).lower()


def match_sources(P: pd.DataFrame, S: pd.DataFrame, ccfg: dict) -> pd.DataFrame:
    """Return S with excl/excl_detail updated (DUPLICATE ...) plus helper columns `match_to`, `match_score`."""
    S = S.copy()
    S["match_to"] = ""
    S["match_score"] = np.nan
    amt_tol = float(ccfg.get("amount_tolerance", 0.02))
    up_tol = float(ccfg.get("unit_price_tolerance", 0.02))
    tight = float(ccfg.get("tight_amount_tolerance", 0.002))
    min_text = float(ccfg.get("min_text_similarity", 0.45))
    allow_multi = bool(ccfg.get("allow_multi_invoice_per_receipt", True))
    pat = ccfg.get("po_pattern", r"\d{6,}")

    P = P.reset_index(drop=True)
    S = S.reset_index(drop=True)
    P["po_n"] = _po_norm(P["po"], pat) if "po" in P and P["po"] is not None else None
    S["po_n"] = _po_norm(S["po"], pat) if "po" in S and S["po"] is not None else None
    P["poi"] = pd.to_numeric(P["po_item"], errors="coerce") if "po_item" in P else np.nan
    S["poi"] = _po_item_ref(S["po"]) if "po" in S else np.nan

    # ---- 1. re-exports among secondary lines that are still eligible
    elig = (S["excl"] == "")
    keymap = {"supplier": "supplier_key", "amount_raw": "amount_raw", "quantity": "quantity",
              "invoice_number": "invoice_number"}
    kcols = [keymap.get(k, k) for k in ccfg.get("reexport_key", ["invoice_number", "supplier", "amount_raw", "quantity"])]
    kdf = pd.DataFrame(index=S.index)
    for c in kcols:
        v = S[c]
        if v.dtype.kind in "fi":
            v = v.round(2)
        kdf[c] = v.astype("object").where(v.notna(), "~").astype(str).str.upper().str.strip()
    S["cluster"] = kdf.agg("|".join, axis=1)
    sub = S[elig].sort_values("_row")
    first_idx = sub.groupby("cluster").head(1).index
    rep_of = sub.groupby("cluster").apply(lambda g: g.index[0], include_groups=False)
    is_dup = elig & ~S.index.isin(first_idx)
    S.loc[is_dup, "excl"] = "DUPLICATE"
    S.loc[is_dup, "excl_detail"] = "re-export of " + S.loc[is_dup, "cluster"].map(rep_of).map(lambda i: S.at[int(i), "record_id"] if pd.notna(i) else "")
    S.loc[is_dup, "match_to"] = S.loc[is_dup, "cluster"].map(rep_of).map(lambda i: S.at[int(i), "record_id"] if pd.notna(i) else "")
    # cluster-level PO (a re-export may carry the reference that the first export lost)
    po_cluster = S[elig & S["po_n"].notna()].groupby("cluster")["po_n"].first()
    poi_cluster = S[elig & S["poi"].notna()].groupby("cluster")["poi"].first()

    # ---- 2. match representatives to primary lines
    reps = S.loc[first_idx].copy()
    reps["po_n"] = reps["po_n"].where(reps["po_n"].notna(), reps["cluster"].map(po_cluster))
    reps["poi"] = reps["poi"].where(reps["poi"].notna(), reps["cluster"].map(poi_cluster))

    pn = P["net"].values.astype(float)
    pq = P["quantity"].values.astype(float) if "quantity" in P else np.full(len(P), np.nan)
    p_up = np.where((pq != 0) & ~np.isnan(pq), pn / np.where(pq == 0, np.nan, pq), np.nan)
    p_sup = P["supplier_key"].values
    p_txt = [_fold(t) for t in P["desc"].values]
    p_poi = P["poi"].values.astype(float)
    po_index = P.groupby("po_n").indices if P["po_n"].notna().any() else {}
    order = np.argsort(pn)
    sorted_n = pn[order]

    cands = []   # (score, s_idx, p_idx)
    for si, r in reps.iterrows():
        sn = float(r["net"]) if pd.notna(r["net"]) else np.nan
        if np.isnan(sn):
            continue
        sq = float(r["quantity"]) if pd.notna(r.get("quantity", np.nan)) else np.nan
        s_up = sn / sq if (not np.isnan(sq) and sq != 0) else np.nan
        s_txt = _fold(r["desc"])
        pset = {}
        po = r["po_n"]
        if po is not None and po in po_index:
            for pi in po_index[po]:
                pset[pi] = True
        # amount-based candidates (no PO, unknown PO, or PO whose lines did not agree)
        lo, hi = sorted((sn * (1 - tight), sn * (1 + tight)))
        a, b = np.searchsorted(sorted_n, lo - 0.005), np.searchsorted(sorted_n, hi + 0.005)
        if b - a <= 60:
            for pi in order[a:b]:
                pset.setdefault(pi, False)
        for pi, via_po in pset.items():
            amt_rel = abs(sn - pn[pi]) / max(abs(pn[pi]), 1.0)
            unit_rel = abs(s_up - p_up[pi]) / max(abs(p_up[pi]), 1e-9) if not (np.isnan(s_up) or np.isnan(p_up[pi])) else np.nan
            text = fuzz.token_sort_ratio(s_txt, p_txt[pi]) / 100.0 if s_txt and p_txt[pi] else 0.0
            sup_same = (r["supplier_key"] is not None and p_sup[pi] is not None and r["supplier_key"] == p_sup[pi])
            item_bonus = 0.1 if (not np.isnan(r["poi"]) and not np.isnan(p_poi[pi]) and r["poi"] == p_poi[pi]) else 0.0
            ok = False
            if via_po:
                same_sign = (sn >= 0) == (pn[pi] >= 0) or abs(pn[pi]) < 1
                ok = same_sign and (amt_rel <= amt_tol or (not np.isnan(unit_rel) and unit_rel <= up_tol and text >= 0.3)
                                    or (amt_rel <= 0.0005))
            if not ok and sup_same and amt_rel <= tight and text >= min_text:
                ok = True
            if not ok and (not via_po) and amt_rel <= 0.0005 and text >= 0.85:
                ok = True       # identical amount and near-identical text, supplier typed differently
            if ok:
                score = (1.0 if via_po else 0.0) + (1 - min(amt_rel, 1)) + 0.5 * text + (0.2 if sup_same else 0) + item_bonus
                cands.append((score, si, pi))
    cands.sort(key=lambda t: -t[0])
    used_p, done_s, multi = {}, {}, []
    for score, si, pi in cands:
        if si in done_s:
            continue
        if pi in used_p:
            multi.append((score, si, pi))
            continue
        used_p[pi] = si
        done_s[si] = (pi, score)
    if allow_multi:        # an already-matched receipt invoiced again for (almost) exactly the same amount
        for score, si, pi in multi:
            if si in done_s:
                continue
            sn = reps.at[si, "net"]
            if abs(sn - pn[pi]) / max(abs(pn[pi]), 1.0) <= 0.0005 and reps.at[si, "po_n"] is not None and reps.at[si, "po_n"] == P.at[pi, "po_n"]:
                done_s[si] = (pi, score)
    # vectorised write-back (a per-row loop over a 28k-row frame is quadratic)
    members_of = S.groupby("cluster").indices
    is_reexp = S["excl_detail"].astype(str).str.startswith("re-export").values
    can_override = S["excl"].isin(["", "DUPLICATE"]).values
    rid_for = np.full(len(S), "", dtype=object)
    score_for = np.full(len(S), np.nan)
    p_rid = P["record_id"].values
    for si, (pi, score) in done_s.items():
        for j in members_of[S.at[si, "cluster"]]:
            if can_override[j] and (is_reexp[j] or j == si):
                rid_for[j] = p_rid[pi]
                score_for[j] = score
    hit = rid_for != ""
    S.loc[hit, "excl"] = "DUPLICATE"
    S.loc[hit, "excl_detail"] = "duplicate of " + pd.Series(rid_for[hit], index=S.index[hit])
    S.loc[hit, "match_to"] = rid_for[hit]
    S.loc[hit, "match_score"] = score_for[hit]
    return S
