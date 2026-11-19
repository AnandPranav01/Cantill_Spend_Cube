"""Text / number / date helpers shared by all modules. No company-specific logic here."""
from __future__ import annotations

import re
import unicodedata

import numpy as np
import pandas as pd

# characters OCR commonly confuses; used for *matching keys only*, never for output text
_OCR_MAP = str.maketrans({"0": "O", "1": "I", "5": "S", "8": "B", "$": "S", "|": "I"})

LEGAL_SUFFIXES = {
    "PTY", "LTD", "LIMITED", "LLC", "INC", "INCORPORATED", "CORP", "CORPORATION", "CO", "COMPANY", "GMBH", "AG",
    "SA", "SAS", "SARL", "SRL", "SPA", "BV", "NV", "OY", "AB", "AS", "SDN", "BHD", "PLC", "LLP", "LP",
    "PVT", "PRIVATE", "SL", "SLU", "KG", "OHG", "UG", "DOO", "ZOO", "SP", "KFT", "SRO", "SE", "CV", "CIA",
    "THE", "AND", "OF",
}


def strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


def norm_text(s) -> str:
    """Lower-case, accent-free, punctuation -> space, collapsed whitespace."""
    if s is None or (isinstance(s, float) and np.isnan(s)) or s is pd.NA:
        return ""
    s = strip_accents(str(s)).lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def parse_number(x) -> float:
    """Parse '1,234.56', '1 234,56', '(12.5)', '12.5-', '15%' -> float (nan if impossible)."""
    if x is None or x is pd.NA:
        return np.nan
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)
    s = str(x).strip()
    if s == "" or s.lower() in ("nan", "none", "null", "n/a", "na", "-"):
        return np.nan
    neg = False
    if s.startswith("(") and s.endswith(")"):
        neg, s = True, s[1:-1]
    if s.endswith("-"):
        neg, s = True, s[:-1]
    s = s.replace("%", "").replace("\u00a0", "").replace(" ", "")
    s = re.sub(r"[^0-9,.\-+eE]", "", s)
    if s == "":
        return np.nan
    if "," in s and "." in s:
        if s.rfind(",") > s.rfind("."):          # 1.234,56
            s = s.replace(".", "").replace(",", ".")
        else:                                     # 1,234.56
            s = s.replace(",", "")
    elif "," in s:
        parts = s.split(",")
        if len(parts) == 2 and len(parts[1]) in (1, 2):   # 12,5 / 12,50 -> decimal comma
            s = parts[0] + "." + parts[1]
        else:                                              # 1,234 / 1,234,567 -> thousands
            s = s.replace(",", "")
    try:
        v = float(s)
    except ValueError:
        return np.nan
    return -v if neg else v


def parse_number_series(ser: pd.Series) -> pd.Series:
    """Vectorised number parsing with a slow path only for the strings pandas cannot read."""
    num = pd.to_numeric(ser, errors="coerce")
    bad = num.isna() & ser.notna()
    if bad.any():
        num = num.astype(float)
        num[bad] = ser[bad].map(parse_number)
    return num.astype(float)


_DATE_FORMATS = ["%Y-%m-%d", "%d/%m/%Y", "%d-%b-%y", "%d-%b-%Y", "%Y/%m/%d", "%d.%m.%Y", "%m/%d/%Y",
                 "%d-%m-%Y", "%Y%m%d", "%b %d, %Y", "%d %b %Y", "%d/%m/%y", "%Y-%m-%d %H:%M:%S", "%d-%m-%y"]


def parse_dates(ser: pd.Series, formats: list[str] | None = None, min_year: int = 1990, max_year: int = 2035):
    """Parse a date column trying several formats (first match wins, day-first preferred).
    Returns (datetimes, flag) with flag in '', 'missing', 'unparseable', 'out_of_range'."""
    s = ser.astype("object").where(ser.notna(), None)
    s = s.map(lambda v: None if v is None else (str(v).strip() or None))
    out = pd.Series(pd.NaT, index=ser.index, dtype="datetime64[ns]")
    todo = s.notna()
    for fmt in (formats or _DATE_FORMATS):
        if not todo.any():
            break
        parsed = pd.to_datetime(s[todo], format=fmt, errors="coerce")
        ok = parsed.notna()
        if ok.any():
            idx = parsed.index[ok]
            out.loc[idx] = parsed[ok].values
            todo.loc[idx] = False
    flag = pd.Series("", index=ser.index, dtype=object)
    flag[s.isna()] = "missing"
    flag[todo] = "unparseable"
    oor = out.notna() & ((out.dt.year < min_year) | (out.dt.year > max_year))
    flag[oor] = "out_of_range"
    return out, flag


def ocr_fold(s: str) -> str:
    """Fold common OCR substitutions (0/O, 1/I, 5/S, 8/B) for matching keys only."""
    return s.translate(_OCR_MAP)


def compact(s: str) -> str:
    """Upper-case alphanumerics only (spaces removed): robust to 'COR ORA' vs 'CORORA'."""
    return re.sub(r"[^A-Z0-9]", "", strip_accents(str(s)).upper())


def blank(v) -> bool:
    return v is None or v is pd.NA or (isinstance(v, float) and np.isnan(v)) or str(v).strip() == ""
