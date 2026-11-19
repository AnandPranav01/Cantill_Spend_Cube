"""Score an output file against the kit's labels (A: labels_A.csv, B: seed_labels.csv).
usage: python scripts/evaluate.py A output_A.csv data/company_A   |   python scripts/evaluate.py B output_B.csv data/company_B"""
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import homogeneity_completeness_v_measure


def acc(o, lab, levels):
    for k, (oc, lc) in enumerate(levels, 1):
        ok = (o[oc].fillna("") == lab[lc].fillna("")).values
        w = o["spend"].abs().values
        print(f"  L{k} accuracy: lines {ok.mean():.3f} | spend-weighted {(ok*w).sum()/max(w.sum(),1):.3f}")


def main(kind, out, d):
    o = pd.read_csv(out, dtype={"record_id": str, "l1": str, "l2": str, "l3": str, "l4": str}, keep_default_na=False)
    print(f"{kind}: {len(o)} rows; included {int(o.include.sum())}; spend {o.spend.sum():,.0f}")
    print("  exclusions:", o.exclusion_reason.value_counts().to_dict())
    if kind == "A":
        lab = pd.read_csv(f"{d}/labels_A.csv", dtype=str)
        m = o.merge(lab, on="record_id")
        print(f"  ERP rows labelled: {len(m)}")
        acc(m[m.include == 1], m[m.include == 1], [("l1", "category_l1"), ("l2", "category_l2"), ("l3", "category_l3")])
        h, c, v = homogeneity_completeness_v_measure(m.supplier_key_y, m.supplier_key_x)
        print(f"  supplier grouping vs golden: homogeneity {h:.3f} completeness {c:.3f} V {v:.3f}  (ours {m.supplier_key_x.nunique()} vs {m.supplier_key_y.nunique()})")
        mi = m[m.item_key_y.notna()]
        h, c, v = homogeneity_completeness_v_measure(mi.item_key_y, mi.item_key_x)
        print(f"  item grouping vs golden:     homogeneity {h:.3f} completeness {c:.3f} V {v:.3f}  (ours {mi.item_key_x.nunique()} vs {mi.item_key_y.nunique()})")
    else:
        lab = pd.read_csv(f"{d}/seed_labels.csv", dtype=str)
        m = o.merge(lab, on="line_id" if "line_id" in o else "record_id", how="inner") if "line_id" in o else o.merge(lab, left_on="record_id", right_on="line_id")
        m["include_y"] = m["include_y"].astype(int)
        print(f"  seed rows: {len(m)}; include agreement {(m.include_x == m.include_y).mean():.3f}")
        print(pd.crosstab(m.include_y, m.include_x, rownames=["seed"], colnames=["ours"]))
        inc = m[(m.include_x == 1) & (m.include_y == 1)]
        acc(inc, inc, [("l1", "category_l1"), ("l2", "category_l2"), ("l3", "category_l3")])


if __name__ == "__main__":
    main(*sys.argv[1:4])
