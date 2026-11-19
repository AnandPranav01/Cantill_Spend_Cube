"""Load one source file into a canonical frame; clean amounts / dates; apply status and exclusion rules."""
from __future__ import annotations

import re
import numpy as np
import pandas as pd

from .config import Config
from .textutil import norm_text, parse_dates, parse_number_series

# ---------------------------------------------------------------------------------------------
# default (generic, company-agnostic) exclusion vocabulary; extended per company in the config
# ---------------------------------------------------------------------------------------------
DEFAULT_NONPROC = {
    "tax": r"\b(?:sales|use|state|local|input|output|payroll|withholding|income|excise|property|vat|gst|sst)\b[^a-z]{0,4}"
           r"(?:[a-z&/]+ ){0,2}tax(?:es)?\b|\btaxes\b|\b(?:vat|gst)\s*(?:input|output|payable|receivable|control|expense)\b"
           r"|\bvalue added tax\b|\btax (?:payable|expense|authority|payment)\b|\bincome tax\b",
    "intercompany": r"\binter[- ]?company\b|\binterco\b|\bi/c\b|\bic (?:payable|payables|charge|recharge)\b|\bintra[- ]?group\b"
                    r"|\bgroup (?:recharge|charge|allocation)\b|\bmanagement fee\b",
    "royalty_dividend": r"\broyalt(?:y|ies)\b|\bdividends?\b|\blicen[cs]e fee income\b",
    "financing": r"\binterest (?:expense|charge|charges|payment|on loan)\b|\bloan (?:repayment|interest|principal)\b"
                 r"|\bbank (?:charges|fees|charge|fee)\b|\bfinance (?:cost|charge)s?\b|\bforeign exchange (?:gain|loss)\b|\bfx (?:gain|loss)\b",
    "payroll_revenue": r"\bpayroll\b|\bsalar(?:y|ies)\b|\bwages\b|\bsales revenue\b|\bcustomer (?:receipt|refund)s?\b"
                       r"|\bpartner billing\b|\bpass[- ]through\b|\bre-?charge\b",
}
DEFAULT_NONPROC_RE = re.compile("|".join(f"(?:{v})" for v in DEFAULT_NONPROC.values()), re.I)
TAX_AUTH_RE = re.compile(r"\b(?:revenue service|tax authority|inland revenue|internal revenue|revenue authority|tax office|"
                         r"receita federal|finanzamt|hmrc)\b", re.I)

DEFAULT_CANCEL_RE = re.compile(r"cancel|void|revers|delet|annul|abandon|written off|write-off", re.I)
DEFAULT_NOTAPPR_RE = re.compile(r"reject|on hold|\bhold\b|submit|disput|pending|draft|block|parked|declin|unappr|not appr|"
                                r"awaiting|waiting|in review|under review|\bnew\b|\bopen\b|unpaid|entered|created|incomplete", re.I)
DEFAULT_APPR_RE = re.compile(r"approv|paid|posted|complete|cleared|settled|released|closed|accepted|processed|final", re.I)


def _sniff_delimiter(path, enc) -> str:
    import csv
    with open(path, "r", encoding=enc, errors="replace") as fh:
        head = "".join([fh.readline() for _ in range(5)])
    try:
        return csv.Sniffer().sniff(head, delimiters=",;\t|").delimiter
    except csv.Error:
        return ","


def _read_csv(path, encoding=None, delimiter=None) -> pd.DataFrame:
    last = None
    for enc in ([encoding] if encoding else ["utf-8-sig", "utf-8", "cp1252", "latin-1"]):
        try:
            sep = delimiter or _sniff_delimiter(path, enc)
            return pd.read_csv(path, dtype=str, sep=sep, encoding=enc, keep_default_na=False,
                               na_values=["", "NaN", "nan", "NULL", "null"])
        except UnicodeDecodeError as e:   # try next encoding
            last = e
    raise last


def _col(df: pd.DataFrame, name):
    if name is None:
        return pd.Series([None] * len(df), index=df.index, dtype=object)
    if name not in df.columns:
        raise KeyError(f"column '{name}' not found; available: {list(df.columns)[:40]}")
    return df[name]


def _joined(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    cols = [c for c in cols if c in df.columns]
    if not cols:
        return pd.Series([""] * len(df), index=df.index, dtype=object)
    parts = [df[c].fillna("").astype(str).str.strip() for c in cols]
    out = parts[0]
    for p in parts[1:]:
        out = out.str.cat(p, sep=" | ")
    out = out.str.replace(r"(?:\s*\|\s*)+$", "", regex=True).str.replace(r"^(?:\s*\|\s*)+", "", regex=True)
    return out.str.strip()


# ---------------------------------------------------------------------------------------------
# currency conversion
# ---------------------------------------------------------------------------------------------
def _month_index(d: pd.Series) -> pd.Series:
    return d.dt.year * 12 + d.dt.month - 1


def convert_currency(df: pd.DataFrame, amount: pd.Series, fx: dict, reporting: str, cfg: Config,
                     dates: dict[str, pd.Series]) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Return (amount_in_reporting, rate_used, flag). Rate semantics: units of line currency per 1 reporting unit
    unless fx['rate_is'] == 'reporting_per_unit'."""
    n = len(df)
    cur = _col(df, fx.get("currency_column")).astype("object").map(lambda v: "" if v is None or (isinstance(v, float) and np.isnan(v)) else str(v).strip().upper())
    given = parse_number_series(_col(df, fx.get("rate_column"))) if fx.get("rate_column") else pd.Series(np.nan, index=df.index)
    given = given.where(given > 0)
    per_unit = fx.get("rate_is", "units_per_reporting") == "reporting_per_unit"
    reporting = (fx.get("reporting_currency") or reporting).upper()

    table_rate = pd.Series(np.nan, index=df.index)
    tab = fx.get("table")
    if tab and tab.get("file"):
        t = _read_csv(cfg.resolve(tab["file"]))
        t = t.rename(columns={tab.get("currency", "currency"): "cur", tab.get("month", "month"): "month",
                              tab.get("rate", "rate"): "rate"})
        t["cur"] = t["cur"].str.strip().str.upper()
        t["rate"] = parse_number_series(t["rate"])
        t["mi"] = _month_index(pd.to_datetime(t["month"].astype(str).str.slice(0, 7) + "-01", errors="coerce"))
        t = t.dropna(subset=["rate", "mi"]).sort_values(["cur", "mi"])
        gmin, gmax = t["mi"].min(), t["mi"].max()
        # month index per row: first date column that is valid and inside the table range
        mi = pd.Series(np.nan, index=df.index)
        for dc in fx.get("date_columns", []):
            d = dates.get(dc)
            if d is None:
                continue
            m = _month_index(d)
            ok = mi.isna() & m.notna() & (m >= gmin - 1) & (m <= gmax + 1)
            mi[ok] = m[ok]
        # nothing usable -> use any date (clipped to range) so that a rate is still found
        for dc in fx.get("date_columns", []):
            d = dates.get(dc)
            if d is None:
                continue
            m = _month_index(d)
            ok = mi.isna() & m.notna()
            mi[ok] = m[ok].clip(gmin, gmax)
        for c, grp in t.groupby("cur"):
            rows = (cur == c) & mi.notna()
            if not rows.any():
                continue
            idx = grp["mi"].values
            vals = grp["rate"].values
            q = mi[rows].values.astype(int)
            pos = np.searchsorted(idx, q)
            pos = np.clip(pos, 1, len(idx) - 1) if len(idx) > 1 else np.zeros_like(pos)
            if len(idx) > 1:
                left, right = idx[pos - 1], idx[pos]
                pick = np.where(np.abs(q - left) <= np.abs(right - q), pos - 1, pos)
            else:
                pick = pos
            table_rate[rows] = vals[pick]
    tol = float(fx.get("sanity_tolerance", 1.0))
    rate = pd.Series(np.nan, index=df.index)
    flag = pd.Series("", index=df.index, dtype=object)
    is_rep = (cur == reporting) | (cur == "")
    rate[is_rep] = 1.0
    flag[cur == ""] = "no_currency_assumed_reporting"
    nonrep = ~is_rep
    use_given = nonrep & given.notna()
    if table_rate.notna().any() and tol > 0:
        ratio = (given / table_rate) if not per_unit else (table_rate / given)
        ratio = ratio.where(ratio > 0)
        dev = np.maximum(ratio, 1.0 / ratio) - 1.0          # symmetric: 2x too high or 2x too low both count
        absurd = use_given & table_rate.notna() & (dev > tol)
        flag[absurd] = "rate_overridden_by_table"
        use_given &= ~absurd
    rate[use_given] = given[use_given]
    use_tab = nonrep & rate.isna() & table_rate.notna()
    rate[use_tab] = table_rate[use_tab]
    flag[use_tab & (flag == "")] = "rate_from_table"
    flag[nonrep & rate.isna()] = "no_fx_rate"
    out = amount * rate if per_unit else amount / rate
    return out, rate, flag


# ---------------------------------------------------------------------------------------------
# rule evaluation
# ---------------------------------------------------------------------------------------------
def _rule_mask(df: pd.DataFrame, rule: dict) -> pd.Series:
    cols = rule.get("columns") or ([rule["column"]] if rule.get("column") else [])
    cols = [c for c in cols if c in df.columns]
    mask = pd.Series(False, index=df.index)
    for c in cols:
        s = df[c].astype("object").where(df[c].notna(), "").astype(str)
        m = pd.Series(True, index=df.index)
        if rule.get("regex"):
            m &= s.str.contains(rule["regex"], case=False, regex=True, na=False)
        if rule.get("not_regex"):
            m &= ~s.str.contains(rule["not_regex"], case=False, regex=True, na=False)
        if rule.get("in") is not None:
            vals = {str(v).strip().lower() for v in rule["in"]}
            m &= s.str.strip().str.lower().isin(vals)
        if rule.get("blank"):
            m &= s.str.strip() == ""
        if rule.get("numeric_lt") is not None:
            m &= parse_number_series(df[c]) < rule["numeric_lt"]
        mask |= m
    return mask


def apply_exclusions(df: pd.DataFrame, src: dict, cfg: Config, canon: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Return (code, detail) per row. Order: CANCELLED, NOT_APPROVED, NON_PROCUREMENT (explicit rules first,
    then generic keyword rules). DUPLICATE is assigned later (combine step)."""
    code = pd.Series("", index=df.index, dtype=object)
    detail = pd.Series("", index=df.index, dtype=object)
    st = src.get("status")
    if st and st.get("column") in df.columns:
        sraw = df[st["column"]].astype("object").where(df[st["column"]].notna(), "").astype(str).str.strip()
        low = sraw.str.lower()
        if st.get("cancelled") is not None:
            canc = low.isin({str(v).lower() for v in st["cancelled"]})
        else:
            canc = low.str.contains(DEFAULT_CANCEL_RE, na=False)
        if st.get("approved") is not None:
            appr = low.isin({str(v).lower() for v in st["approved"]})
            notappr = ~canc & ~appr & (low != "")
        else:
            appr = low.str.contains(DEFAULT_APPR_RE, na=False)
            notappr = ~canc & ~appr & low.str.contains(DEFAULT_NOTAPPR_RE, na=False)
        blank_is = st.get("blank_is", "approved")
        if blank_is == "not_approved":
            notappr |= (low == "")
        code[canc] = "CANCELLED"
        detail[canc] = "status=" + sraw[canc]
        m = notappr & (code == "")
        code[m] = "NOT_APPROVED"
        detail[m] = "status=" + sraw[m]
    # explicit rules from the config, in order
    for i, rule in enumerate(src.get("exclusions") or []):
        m = _rule_mask(df, rule) & (code == "")
        if m.any():
            code[m] = rule.get("code", "NON_PROCUREMENT")
            detail[m] = rule.get("name", f"rule{i}")
    # generic keyword rules
    dflt = cfg["exclusion_defaults"]
    if dflt.get("enabled", True):
        cols = src.get("exclusion_text_columns")
        if cols is None:
            cols = list(src["text_columns"]) + [c for c in src["context_columns"] if re.search(r"gl|account|type|categ", c, re.I)]
        cols = [c for c in cols if c in df.columns]
        extra = [re.compile(x, re.I) for x in dflt.get("extra_regex", [])]
        if cols:
            hit = pd.Series(False, index=df.index)
            for c in cols:
                s = df[c].astype("object").where(df[c].notna(), "").astype(str)
                hit |= s.str.contains(DEFAULT_NONPROC_RE, na=False)
                for rx in extra:
                    hit |= s.str.contains(rx, na=False)
        else:
            hit = pd.Series(False, index=df.index)
        sup = canon["supplier_name"].fillna("").astype(str)
        hit |= sup.str.contains(TAX_AUTH_RE, na=False)
        m = hit & (code == "")
        code[m] = "NON_PROCUREMENT"
        detail[m] = "default_keyword"
    return code, detail


# ---------------------------------------------------------------------------------------------
# main entry
# ---------------------------------------------------------------------------------------------
def load_source(cfg: Config, src: dict) -> pd.DataFrame:
    path = cfg.resolve(src["file"])
    if not path.exists():
        raise FileNotFoundError(f"source file not found: {path}")
    raw = _read_csv(path, src.get("encoding"), src.get("delimiter"))
    raw.columns = [c.strip() for c in raw.columns]
    f = src["fields"]
    idc = src["id"]["column"]
    if idc not in raw.columns:
        raise KeyError(f"id column '{idc}' not in {path.name}: {list(raw.columns)[:30]}")

    canon = pd.DataFrame(index=raw.index)
    canon["record_id"] = src["id"].get("prefix", "") + raw[idc].astype(str).str.strip()
    canon["src"] = src["name"]
    # --- dates
    dates, flags = {}, {}
    date_cols = {k: v for k, v in f.items() if k.endswith("date") or k == "date"}
    fx = src["amount"].get("fx")
    wanted = set(date_cols.values()) | set((fx or {}).get("date_columns", []))
    for c in wanted:
        if c in raw.columns:
            dates[c], flags[c] = parse_dates(raw[c], src.get("date_formats"))
    main_date = f.get("date")
    if main_date in dates:
        canon["date"], canon["date_flag"] = dates[main_date], flags[main_date]
    else:
        canon["date"], canon["date_flag"] = pd.NaT, "missing"
    # --- plain fields
    for k in ("supplier_name", "supplier_id", "item_code", "po", "po_item", "invoice_number", "invoice_line"):
        canon[k] = _col(raw, f.get(k)).astype("object") if f.get(k) else None
        if f.get(k):
            canon[k] = canon[k].map(lambda v: None if v is None or (isinstance(v, float) and np.isnan(v)) else str(v).strip()).replace({"": None})
    canon["quantity"] = parse_number_series(_col(raw, f.get("quantity"))) if f.get("quantity") else np.nan
    canon["desc"] = _joined(raw, src["text_columns"])
    canon["desc_norm"] = canon["desc"].map(norm_text)
    for c in src["context_columns"]:
        if c in raw.columns:
            canon["ctx__" + c] = raw[c].astype("object").where(raw[c].notna(), "").astype(str).str.strip()
    # --- amount
    a = src["amount"]
    amt = parse_number_series(raw[a["column"]]) if a["column"] in raw.columns else pd.Series(np.nan, index=raw.index)
    canon["amount_raw"] = amt
    tax = a.get("tax") or {"mode": "net"}
    net = amt.copy()
    aflag = pd.Series("", index=raw.index, dtype=object)
    if tax.get("mode") == "gross":
        dv = float(tax.get("default_vat", 0.15))
        if tax.get("vat_column") and tax["vat_column"] in raw.columns:
            v = parse_number_series(raw[tax["vat_column"]])
            v = v.where(v.isna() | (v <= 1.0), v / 100.0)      # 15 -> 0.15 ; 15% -> 0.15 ; 0.15 stays
            aflag[v.isna()] = "vat_default"
            v = v.fillna(dv)
        else:
            v = pd.Series(dv, index=raw.index)
        net = amt / (1.0 + v)
        canon["vat_rate"] = v
    if fx:
        net, rate, fflag = convert_currency(raw, net, fx, cfg["reporting_currency"], cfg, dates)
        canon["fx_rate"] = rate
        aflag = aflag.where(fflag == "", fflag)
    aflag[amt.isna()] = "amount_unparseable"
    canon["net"] = net
    canon["amount_flag"] = aflag
    # --- exclusions (status, explicit rules, generic keywords)
    code, detail = apply_exclusions(raw, src, cfg, canon)
    canon["excl"], canon["excl_detail"] = code, detail
    # keep a few raw columns that are useful for debugging / matching
    canon["_row"] = np.arange(len(canon))
    return canon
