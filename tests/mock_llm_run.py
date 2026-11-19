"""Plumbing test for the optional LLM path: a fake LLM answers from A's labels (with 12% noise).
Shows how accuracy changes when ~N top-spend unique texts get LLM labels. Not part of the tool."""
import os, re, sys, json, random
import pandas as pd
import spendcube.classify as C
from spendcube.config import load_config
from spendcube.pipeline import run
D = "data/company_A"
lab = pd.read_csv(f"{D}/labels_A.csv", dtype=str)
erp = pd.read_csv(f"{D}/erp_goods_receipts.csv", dtype=str)
m = erp.merge(lab, left_on=erp.line_id.radd("ERP:"), right_on="record_id")
txt = (m.material_short_text.fillna("") + " | " + m.service_short_text.fillna("")).str.strip(" |")
truth = dict(zip(txt, zip(m.category_l1, m.category_l2, m.category_l3)))
cfg = load_config("configs/company_A.yaml", overrides={"classification": {"embeddings": "off", "llm": {"enabled": "auto", "max_unique": int(sys.argv[1]) if len(sys.argv) > 1 else 1500}}})
from spendcube.taxonomy import load_taxonomy
tax = load_taxonomy(cfg)
names = {" > ".join(p): j + 1 for j, p in enumerate(tax.paths)}
random.seed(1)
def fake(prompt, key, model, max_tokens=600):
    lines = prompt.split("Lines:\n")[1].split("\n")
    out = {}
    for ln in lines:
        mm = re.match(r"(\d+)\. (.*?)(  \[.*\])?$", ln)
        if not mm: continue
        t = truth.get(mm.group(2).strip())
        v = names.get(" > ".join(t)) if t else 0
        if v and random.random() < 0.12: v = random.randint(1, len(tax))
        out[mm.group(1)] = v or 0
    return json.dumps(out)
C._call_llm = fake
os.environ["ANTHROPIC_API_KEY"] = "test"
run(cfg, "/tmp/A_mock.csv", use_labels=False)
