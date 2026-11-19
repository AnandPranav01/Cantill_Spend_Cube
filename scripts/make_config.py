"""Draft a config for a NEW company by inspecting its file headers (used for the blind-test company C).
It only guesses column roles from names and value shapes; read the printed summary and edit the YAML if needed.

  python scripts/make_config.py --data-dir path/to/C --file invoice_lines.csv --taxonomy tree.csv --name company_C
"""
import argparse
import re
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from spendcube.ingest import _read_csv          # noqa: E402
from spendcube.textutil import parse_number_series  # noqa: E402


def pick(cols, pats, exclude=()):
    for p in pats:
        for c in cols:
            if c in exclude:
                continue
            if re.search(p, c, re.I):
                return c
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--file", required=True)
    ap.add_argument("--taxonomy", required=True)
    ap.add_argument("--name", default="company_C")
    ap.add_argument("--currency", default="USD", help="reporting currency")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    d = Path(a.data_dir)
    df = _read_csv(d / a.file).head(5000)
    df.columns = [c.strip() for c in df.columns]
    cols = list(df.columns)
    n = len(df)
    uniq = {c: df[c].nunique(dropna=True) for c in cols}
    num = {c: parse_number_series(df[c]).notna().mean() for c in cols}

    idc = pick([c for c in cols if uniq[c] == n], [r"line.?id|record.?id|^id$|_id$|row", r"."]) or cols[0]
    amt_cols = [c for c in cols if num[c] > 0.9 and re.search(r"amount|value|net|total|spend|cost|price|usd", c, re.I)
                and not re.search(r"rate|qty|quant|tax|vat", c, re.I)]
    amt = pick(amt_cols, [r"^(line_?)?amount", r"net", r"usd", r"amount", r"value", r"total", r"."])
    tax_col = pick(cols, [r"vat.?rate|tax.?rate"])
    gross = bool(amt and re.search(r"gross|incl", amt, re.I))
    date = pick(cols, [r"posting.?date", r"invoice.?date", r"date"])
    sup_id = pick(cols, [r"(supplier|vendor).*(id|code|no|number)"])
    sup_name = pick([c for c in cols if c != sup_id and not re.search(r"type|class|status|flag|categ|group|rate|amount", c, re.I)],
                    [r"(supplier|vendor).*name", r"supplier|vendor|payee|creditor"])
    item_code = pick(cols, [r"(material|item|sku|product|part).*(code|number|no|id)$", r"(material|item|sku)"])
    texts = [c for c in cols if re.search(r"desc|text|narrat|memo|title|item.?name|product.?name|material.?name|^name$", c, re.I)
             and num[c] < 0.5 and c not in (sup_name, sup_id)]
    texts = sorted(texts, key=lambda c: -df[c].fillna("").astype(str).str.len().mean())[:3] or \
        [c for c in cols if num[c] < 0.5 and uniq[c] > 0.2 * n and c not in (sup_name, idc)][:2]
    po = pick(cols, [r"^po[_ ]?(number|no|ref)?$", r"purchase.?order"])
    inv = pick(cols, [r"invoice.?(number|no|num)"])
    qty = pick(cols, [r"quant|qty"])
    cur = pick(cols, [r"^currency", r"curr"])
    rate = pick(cols, [r"(currency|fx|exchange).*rate", r"rate"], exclude=(tax_col,) if tax_col else ())
    status = pick(cols, [r"status|state"])
    ctx = [c for c in cols if 2 <= uniq[c] <= 300 and num[c] < 0.9 and c not in (idc, sup_name, sup_id, status, cur, *texts)
           and not re.search(r"date|rate|amount|qty|quant|price|value|number|^po|invoice", c, re.I)]
    ctx = sorted(ctx, key=lambda c: (not re.search(r"gl|account|categ|type|group|class|dept|centre|center|segment", c, re.I), uniq[c]))[:8]
    cls = [c for c in ctx if re.search(r"gl|account|categ|type|group|class|dept|segment", c, re.I)][:4]

    fields = {k: v for k, v in dict(date=date, supplier_name=sup_name, supplier_id=sup_id, item_code=item_code, po=po,
                                    invoice_number=inv, quantity=qty).items() if v}
    amount = {"column": amt, "tax": ({"mode": "gross", "vat_column": tax_col, "default_vat": 0.15} if gross else {"mode": "net"})}
    if cur and df[cur].nunique() > 1:
        amount["fx"] = {"currency_column": cur, "rate_column": rate, "rate_is": "units_per_reporting",
                        "date_columns": [date] if date else []}
    src = {"name": "lines", "file": a.file, "id": {"column": idc}, "fields": fields, "text_columns": texts,
           "context_columns": ctx, "class_columns": cls, "amount": amount}
    if status:
        vals = df[status].dropna().astype(str).str.strip().value_counts()
        src["status"] = {"column": status}
        print("status values:", vals.to_dict(), "-> add `cancelled:` / `approved:` lists if the defaults misclassify", file=sys.stderr)
    cfg = {"name": a.name, "reporting_currency": a.currency, "data_dir": str(d.resolve()), "taxonomy": {"file": a.taxonomy},
           "sources": [src], "output": {"file": f"output_{a.name}.csv"}}
    out = Path(a.out or f"configs/{a.name}.yaml")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("# DRAFT written by scripts/make_config.py - review before running\n" + yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
    print(f"wrote {out}\n  id={idc} amount={amt} date={date} supplier={sup_name or sup_id} text={texts} item_code={item_code}"
          f"\n  currency={cur} rate={rate} status={status} class_columns={cls}", file=sys.stderr)


if __name__ == "__main__":
    main()
