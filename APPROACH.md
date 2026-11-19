# APPROACH

## In one sentence
One tool for every company. A small config file tells it which columns to use, so a new company needs a new config, **not new code**.

**What I used:** Python (pandas, numpy, scipy, scikit-learn, rapidfuzz, pyyaml). An optional free model for text meaning: `sentence-transformers` (`paraphrase-multilingual-MiniLM-L12-v2`). An optional LLM through an API key. I used Claude (chat) to help write and fix the code. **No paid API is needed** for any result in this file.

## The steps
```
read config → clean amounts and dates → exclude non-spend → count each purchase once
→ merge suppliers → merge items → classify into the tree → write the CSV
```

1. **Read.** The config names the files and columns for amount, date, supplier, description and status.
2. **Clean.** Remove VAT from gross amounts (15% if blank). Convert to the reporting currency (B: the line's rate, or the monthly table if missing or wrong). Read many date formats.
3. **Exclude.** Each excluded row gets a reason: `CANCELLED`, `NOT_APPROVED`, `NON_PROCUREMENT`, `DUPLICATE`.
4. **Count once (Company A).** Match the invoice-tool lines to ERP lines using the PO number, the amount or unit price, the supplier and the text. A matched tool line is `DUPLICATE`. An invoice exported twice is counted once. Tool-only lines stay as spend.
5. **Merge suppliers.** Use the vendor ID when there is one. Otherwise make names simple (lower case, no punctuation, no "PTY LTD") and group almost-identical names. I never merge on a loose match.
6. **Merge items.** Same item code, same cleaned text, or almost the same text with the same numbers. So "DIA 20MM" and "DIA 25MM" stay apart.
7. **Classify.** See below.

## How I classify into a tree the tool has never seen
The tree is a file. No category names are in the code. This works with **zero labels**.

1. Turn each category path (for example "Fuel > Diesel") into text.
2. Compare each line's text with every category. I use word matching (TF-IDF) and, if installed, a small free multilingual model that understands meaning across languages.
3. **Teach itself:** take the lines it is most sure about, and train a simple model on them. It learns the company's own short forms. If real labels exist, it uses them instead.
4. **Use the context:** lines of the same item share the answer. Lines with useless text lean on their supplier and account text, with a low confidence.
5. Choose level 1, then level 2 inside it, and so on, so the answer is always a real path in the tree.
6. **Optional LLM:** if an API key is set, an LLM labels the highest-spend texts and those labels help train step 3. Without a key, everything still runs.

## How well it works (honest numbers)
| Test | Level 1 | Level 2 | Level 3 |
|---|---|---|---|
| Company A, labels hidden, tree only (word matching only) | 47% of lines (63% of spend) | 28% (53%) | 18% (43%) |
| Company A, new records, trained on other labelled A data | 90.5% | 82.8% | 81.1% |
| Practice for Company C: B made to look unseen, no labels, word matching only | 80.5% | 78.6% | 30.5% |

- Classifying an unseen tree with no labels is the weakest part. I did not measure how much the embedding model helps, so I make no claim about it.
- Other checks: B exclusions match all 3,000 seed labels (but one rule was set from them, so that is optimistic). Supplier grouping on A made no wrong merges. Item grouping on A scored 0.96.

## Run time
- Company A (65,549 records, with labels and embeddings, on Colab): about 36 minutes.
- Company A without embeddings: about 3 to 6 minutes.
- Company B (41,028 lines): a few minutes.
- The limit is 60 minutes for 25,000 lines, and every run is inside it.

## LLM cost
No API was used for any result here, so the cost is zero. If a key is set, the tool sends about 1 request per 20 unique texts and at most about 125 requests per run. I have not measured the price.
