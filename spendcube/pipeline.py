"""End-to-end run: ingest -> suppliers -> combine -> items -> classify -> output CSV."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from .classify import Classifier, llm_label
from .combine import match_sources
from .config import Config
from .entities import assign_item_keys, assign_supplier_keys
from .ingest import load_source
from .labels import load_labels
from .taxonomy import load_taxonomy

OUT_COLS = ["record_id", "include", "exclusion_reason", "supplier_key", "item_key", "spend",
            "l1", "l2", "l3", "l4", "confidence"]


def _ctx_tokens(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    parts = []
    for c in cols:
        k = "ctx__" + c
        if k in df.columns:
            v = df[k].fillna("").astype(str).str.strip().str.replace(r"\s+", "_", regex=True)
            parts.append(np.where(v != "", c + "=" + v, ""))
    if not parts:
        return pd.Series([""] * len(df), index=df.index)
    s = pd.Series(parts[0], index=df.index)
    for p in parts[1:]:
        s = s + " " + pd.Series(p, index=df.index)
    return s.str.strip()


def run(cfg: Config, out_path: str, use_labels: bool = True, log=print) -> pd.DataFrame:
    t0 = time.time()
    tax = load_taxonomy(cfg)
    log(f"[tax] {len(tax)} leaf categories, {tax.nlevels} levels")
    srcs = {s["name"]: s for s in cfg["sources"]}
    frames = {n: load_source(cfg, s) for n, s in srcs.items()}
    for n, f in frames.items():
        log(f"[ingest] {n}: {len(f)} rows, excluded so far: {f['excl'].value_counts().to_dict()}")

    book = load_labels(cfg, tax, frames, log) if use_labels else None
    extra = book.extra_frames if book else []
    names = list(frames)

    # ---- suppliers (jointly over every frame, so one supplier is one key across systems)
    sk = assign_supplier_keys([frames[n] for n in names] + extra, cfg["suppliers"], book)
    targets = [frames[n] for n in names] + extra
    for f, s in zip(targets, sk):
        f["supplier_key"] = s.values
    log(f"[suppliers] {pd.concat([frames[n]['supplier_key'] for n in names]).nunique()} supplier keys")

    # ---- combine two systems, count once
    comb = cfg.get("combine")
    if comb:
        frames[comb["secondary"]] = match_sources(frames[comb["primary"]], frames[comb["secondary"]], comb)
        S = frames[comb["secondary"]]
        log(f"[combine] secondary: {S['excl'].value_counts().to_dict()}")

    # ---- items
    allf = [frames[n] for n in names] + extra
    cat = pd.concat([f[["desc"] + (["item_code"] if "item_code" in f else [])] for f in allf], ignore_index=True)
    ik = assign_item_keys(cat, cfg["items"], book)
    pos = 0
    for f in allf:
        f["item_key"] = ik.iloc[pos:pos + len(f)].values
        pos += len(f)
    for f in allf:
        miss = f["item_key"].isna()
        f.loc[miss, "item_key"] = "NA-" + f.loc[miss, "record_id"]
    log(f"[items] {pd.concat([f['item_key'] for f in allf]).nunique()} item keys")

    # ---- classification
    ccfg = cfg["classification"]
    rows = []
    for n, f in [(n, frames[n]) for n in names] + [("_train", e) for e in extra]:
        cc = srcs[n]["class_columns"] if n in srcs else srcs[names[0]]["class_columns"]
        if n == "_train":
            cc = None
        rows.append((n, f))
    parts = []
    for n, f in rows:
        if n == "_train":
            # extra frames carry whichever source they came from
            src_name = f["src"].iloc[0]
            cc = srcs[src_name]["class_columns"]
        else:
            cc = srcs[n]["class_columns"]
        inc = (f["excl"] == "").values
        parts.append(pd.DataFrame({
            "record_id": f["record_id"].values, "text": f["desc"].values, "ctx": _ctx_tokens(f, cc).values,
            "supplier_key": f["supplier_key"].values, "item_key": f["item_key"].values,
            "spend": f["net"].fillna(0).values, "is_target": (n != "_train") & inc, "train_only": n == "_train"}))
    L = pd.concat(parts, ignore_index=True)
    L["weight_spend"] = L["spend"]
    L["label"] = -1
    if book is not None:
        m = L[["record_id"]].merge(book.table[["record_id", "leaf"]], on="record_id", how="left")
        L["label"] = m["leaf"].fillna(-1).astype(int).values
        L["label"] = np.where(L["label"] >= 0, L["label"], -1)
        if book.other_dir:      # labels belong to the other data folder: never apply them to the current records
            L.loc[~L["train_only"], "label"] = -1
    L["weight"] = 1.0
    # lines used for the model: included targets + labelled lines (even when train-only); skip excluded unlabelled
    use = L["is_target"] | (L["label"] >= 0)
    Lu = L[use].copy()
    clf = Classifier(tax, ccfg, log)
    llm = llm_label(Lu, tax, ccfg, log)          # {} unless an API key is configured
    if llm:
        for i, leaf in llm.items():
            if Lu.at[i, "label"] < 0:
                Lu.at[i, "label"] = leaf
    res = clf.fit_predict(Lu)
    leaf_by_rec = pd.Series(res["leaf"].values, index=Lu["record_id"].values)
    conf_by_rec = pd.Series(res["confidence"].values, index=Lu["record_id"].values)
    leaf_by_rec = leaf_by_rec[~leaf_by_rec.index.duplicated()]
    conf_by_rec = conf_by_rec[~conf_by_rec.index.duplicated()]

    # ---- assemble output (one row per input record, in input order)
    outs = []
    for n in names:
        f = frames[n]
        inc = (f["excl"] == "").values
        o = pd.DataFrame({"record_id": f["record_id"].values})
        o["include"] = inc.astype(int)
        o["exclusion_reason"] = np.where(inc, "", np.where(f["excl"] != "", f["excl"], "NON_PROCUREMENT"))
        o["supplier_key"] = f["supplier_key"].values
        o["item_key"] = f["item_key"].values
        o["spend"] = np.where(inc, f["net"].fillna(0).round(2), 0.0)
        li = leaf_by_rec.reindex(o["record_id"]).values
        lv = np.array([tax.level_names(int(i)) if i == i else [""] * 4 for i in li], dtype=object).reshape(len(o), 4)
        for k in range(4):
            o[f"l{k+1}"] = np.where(inc, lv[:, k], "")
        c = conf_by_rec.reindex(o["record_id"]).values.astype(float)
        mscore = f["match_score"].values if "match_score" in f else np.full(len(f), np.nan)
        excl_conf = np.where(f["excl_detail"].astype(str).str.startswith("default_keyword"), 0.7,
                             np.where(np.isnan(mscore), 0.9, np.clip(0.6 + 0.1 * mscore, 0.6, 0.98)))
        o["confidence"] = np.where(inc, np.nan_to_num(c, nan=0.0), excl_conf).round(3)
        outs.append(o[OUT_COLS])
    out = pd.concat(outs, ignore_index=True)
    out.to_csv(out_path, index=False)
    log(f"[done] {len(out)} rows -> {out_path}; included {int(out['include'].sum())}, spend {out['spend'].sum():,.0f} "
        f"{cfg['reporting_currency']}; {time.time()-t0:.0f}s")
    return out
