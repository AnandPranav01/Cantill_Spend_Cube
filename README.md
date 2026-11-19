# spendcube: config-driven spend data cleaning and classification

Takes a company's raw purchase lines and its category tree and produces one clean CSV (one row per input record):
`record_id, include, exclusion_reason, supplier_key, item_key, spend, l1..l4, confidence`.

A new company needs a **new config file, not new code**. No category names exist in the code: the tree is read from a file.

## Setup
```bash
pip install -r requirements.txt
# optional but recommended for an unseen tree (needs internet once, then works offline):
pip install sentence-transformers
```
Python 3.10+. Data files are never committed (see `.gitignore`); put them in `data/company_X/` or pass `--data-dir`.

## The one command
```bash
python -m spendcube --config configs/company_A.yaml --data-dir data/company_A --out outputs/yourname__A.csv
```
Options: `--no-labels` (ignore labels: zero-label mode), `--labels-dir DIR` (train on labelled data in DIR while processing
new records), `--embeddings auto|on|off`, `--llm auto|off`.

| Run | Command |
|---|---|
| Company A / B (build) | `python -m spendcube --config configs/company_A.yaml --data-dir data/company_A --out outputs/yourname__A.csv` |
| A2 / B2 (new records, same layout) | `python -m spendcube --config configs/company_A.yaml --data-dir PATH/A2 --labels-dir data/company_A --out submission/yourname__A2.csv` |
| New company C (no labels) | `python scripts/make_config.py --data-dir PATH/C --file <lines.csv> --taxonomy <tree.csv> --name company_C` then review `configs/company_C.yaml` and `python -m spendcube --config configs/company_C.yaml --out submission/yourname__C.csv` |

**Blind test, one command** (runs A2, B2 and C, writes the three files + C config into `submission/`):
```bash
python scripts/blind_test.py --name yourname --a2 PATH/A2 --b2 PATH/B2 --c PATH/C
```
It drafts C's config with `scripts/make_config.py` (open `submission/company_C.yaml` and check the column mapping; if
something is off, edit the YAML, never the code, and re-run the `python -m spendcube` line it printed). Category trees can be
wide (`level 1..4` columns), a single path column (`A > B > C`) or an adjacency list (`id, parent_id, name`).

If the labels folder is missing, the tool prints a notice and runs with no labels.

## Pipeline
1. **Ingest by config** (`config.py`, `ingest.py`): columns, formats, date parsing (14 formats, flags for unparseable/out-of-range), numbers (`1.234,56`, `(12)`, `15%`), VAT removal for gross amounts (blank rate -> 15%), FX to the reporting currency (given rate, sanity-checked against the monthly table; table used when the rate is missing or absurd).
2. **Exclusions** with reason codes: status rules (CANCELLED / NOT_APPROVED), explicit config rules, and generic keyword rules (tax, intercompany, royalty, financing, payroll/revenue) -> NON_PROCUREMENT.
3. **Combine two systems** (`combine.py`): normalised PO + net amount or unit price + supplier/text agreement; matched tool lines become DUPLICATE, re-exported invoices are counted once, tool-only lines stay as spend.
4. **Suppliers / items** (`entities.py`): supplier id when present, else normalised-name fuzzy clustering with a strict threshold; items by code, identical text, or near-identical text with identical numbers.
5. **Classification** (`classify.py`): see APPROACH.md.
6. **Output** (`pipeline.py`).

## Layout
`spendcube/` code · `configs/` one YAML per company · `scripts/evaluate.py` scores an output against the labels ·
`scripts/make_config.py` drafts a config for a new company · `outputs/` A and B outputs · `APPROACH.md`, `DECISIONS.md`.
