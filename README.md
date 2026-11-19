# 🧊 Cantill Spend Cube

**Turn messy purchase data into a clean answer to: "How much do we spend, on what, and with whom?"**

Built for the Cantill.ai Data & ML Intern challenge (Round 1).

You give the tool a company's purchase lines and its category tree. It gives you one clean CSV where every purchase is counted once, with a clean supplier, a clean item, the net spend, and a category.

> ✨ **A new company needs a new config file, not new code.** The category tree is read from a file. No category names are in the code.

---

## 🔍 What it does

| # | Step | In simple words |

| 1 | **Read the config** | One small file says which column is the amount, date, supplier and so on |
| 2 | **Clean** | Remove VAT, convert currencies to USD, read many date formats |
| 3 | **Count once** | If the same invoice is in two systems, keep one and mark the other `DUPLICATE` |
| 4 | **Exclude non-spend** | Give a reason: `CANCELLED`, `NOT_APPROVED`, `NON_PROCUREMENT` or `DUPLICATE` |
| 5 | **Merge suppliers** | "ACME PTY LTD" and "Acme (Pty)" become one supplier |
| 6 | **Merge items** | The same item typed in different ways becomes one item |
| 7 | **Classify** | Put each line into the company's tree, down to the lowest level, with a confidence score |
| 8 | **Write the CSV** | One row for every input record, excluded rows too |

```
config → clean → count once → exclude → suppliers → items → classify → CSV
```

### The output file
`record_id, include, exclusion_reason, supplier_key, item_key, spend, l1, l2, l3, l4, confidence`

---

## 🚀 Quick start

**1. Install** (Python 3.10 or newer)
```bash
cd Cantill
pip install -r requirements.txt
pip install sentence-transformers      # optional but recommended, downloads a free model once
```

**2. Add the data.** Put each company's files in its own folder. The `data/` folder is ignored by git, so it is never uploaded.
```
Cantill/
  data/
    company_A/
    company_B/
```

**3. Run it**
```bash
python -m spendcube --config configs/company_A.yaml --data-dir data/company_A --out outputs/Anandpranav__A.csv
python -m spendcube --config configs/company_B.yaml --data-dir data/company_B --out outputs/Anandpranav__B.csv
```

> 💡 **On Windows, if you see "An Application Control policy has blocked this file":** your computer is blocking some libraries. Run the tool in Google Colab instead.

### Useful options
| Option | What it does |
|---|---|
| `--no-labels` | Ignore all labels, like for a brand new company |
| `--labels-dir DIR` | Use labelled data from another folder as extra help |
| `--embeddings auto / on / off` | The free meaning model. `off` is much faster |
| `--llm auto / off` | Optional LLM help. Only runs if an API key is set |

---

## 🎯 Blind test (new data from A, B and a new company C)

One command runs all three and saves the files in `submission/`:
```bash
python scripts/blind_test.py --name Anandpranav --a2 PATH/A2 --b2 PATH/B2 --c PATH/C
```

For company C it first **drafts a config** by reading the column names. Open `submission/company_C.yaml` and check the columns. If something is wrong, fix the YAML, never the code.

---

## 🧠 How it classifies an unseen category tree

1. Turn each category path into text.
2. Compare each line's text with every category (word matching, plus a free multilingual model that understands meaning).
3. Take the lines it is most sure about and train a small model on them, so it learns the company's own short forms.
4. If the text is useless ("Product", "Parts"), use the supplier and account text, and give a low confidence.
5. Always return a real path from the tree.

**No labels and no paid API are needed.** If you set `ANTHROPIC_API_KEY`, an LLM can help with the biggest-spend lines. It is optional.

---

## 📊 Results (honest)

| Test | Level 1 | Level 2 | Level 3 |
|---|---|---|---|
| Company A, labels hidden, tree only | 47% | 28% | 18% |
| Company A, new records, with other labelled data | 90.5% | 82.8% | 81.1% |
| Practice for company C (B made to look unseen, no labels) | 80.5% | 78.6% | 30.5% |

These are percentages of lines. The first and third rows use word matching only, without the embedding model. **Classifying a new tree with no labels is the hardest part.** I did not measure how much the embedding model helps.

⏱️ **Run time:** Company A (65,549 records, with embeddings) took about 36 minutes. Without embeddings about 3 to 6 minutes. The limit is 60 minutes for 25,000 lines.

---

## 📁 What is in this repo

```
Cantill/
  spendcube/      the tool (the code)
  configs/        one config file per company
  scripts/        blind_test.py, make_config.py, evaluate.py
  outputs/        results for Company A and B
  submission/     blind-test results and the config for Company C
  APPROACH.md     how it works and how well
  DECISIONS.md    the problems I found in the data and what I decided
```

📄 Read **DECISIONS.md** to see the problems I found and my choices. Read **APPROACH.md** for the design.

🔒 **The task data is confidential and is not in this repo.**

🛠️ **Tools used:** Python, pandas, scikit-learn, rapidfuzz. Optional `sentence-transformers`. Claude (chat) helped me write and fix the code.
