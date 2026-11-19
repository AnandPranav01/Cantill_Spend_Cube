# APPROACH

**Tools used:** Python (pandas, numpy, scipy, scikit-learn, rapidfuzz, pyyaml). Optional: `sentence-transformers` (local multilingual model, `paraphrase-multilingual-MiniLM-L12-v2`) and an optional LLM (Anthropic API, off unless `ANTHROPIC_API_KEY` is set). Claude (chat) was used to help write and debug the code. No API is required: all reported numbers below are from the free, offline path.

## Design
One generic pipeline; a company is a YAML config (files, columns, formats, status values, extra rules). No category names or company logic in code.
`ingest -> exclusions -> suppliers -> combine two systems -> items -> classify -> output` (see README for details).

## Matching and de-duplication
- **Two systems (A):** PO references normalised to the 6+ digit number; a tool line matches an ERP line on net amount (after VAT removal) **or** unit price (weighed goods and partial invoices change amount but not unit price), with supplier/text agreement guarding no-PO matches. Matched tool lines -> DUPLICATE; repeated exports of one invoice are counted once; tool-only lines stay as spend. In the data, only 178 of 4,257 unmatched tool lines even share a PO with the ERP, and their amounts differ by ~50% at the median, so few real duplicates are missed.
- **Suppliers:** vendor id when present; otherwise normalised names (case, punctuation, legal suffixes, OCR look-alikes) clustered with a strict fuzzy threshold and a length guard. Names are never merged on a loose similarity.
- **Items:** same item code, identical normalised text, or near-identical text with identical numbers (so "DIA 20MM" and "DIA 25MM" stay apart).

## Classification into an unseen tree (no labels needed)
1. Read the tree from the file; each leaf path is a text.
2. **Zero-shot score** of each unique line text against every leaf: char/word TF-IDF matcher (always on) blended with local sentence embeddings when installed.
3. **Self-training:** the most confident lines become pseudo-labels for a linear model (LinearSVC) over text, context columns (account text, material type, ...), supplier and item key. It learns the company's own abbreviations and account patterns; two rounds.
4. **Consistency:** lines of the same item share evidence; weak lines (blank/useless text) lean on their supplier's profile.
5. **Top-down decoding:** pick l1, then l2 within it, etc., so the output is always a valid path of the tree. Confidence is derived from the model probability (capped at 0.5 for lines with no usable text).
6. **Optional LLM:** if a key is present, the highest-spend unique texts are labelled by an LLM in batches of 20 (the whole tree is listed in the prompt); these labels then train step 3. If it fails, the tool continues without it.
7. **Known labels** (if any, e.g. the original A/B data when processing A2/B2 via `--labels-dir`) replace pseudo-labels as training signal.

## Measured accuracy (Company A, ERP lines, vs labels_A.csv)
| Setting | L1 | L2 | L3 |
|---|---|---|---|
| **Labels hidden, tree only (TF-IDF matcher, no embeddings, no LLM)** | 47% of lines / 63% of spend | 28% / 53% | 18% / 43% |
| New records, trained on other labelled A data (80/20 split by PO) | 90.5% / 90.6% | 82.8% / 85.3% | 81.1% / 83.6% |

The zero-label number is the honest estimate for an unseen tree with the free path and is weak: a domain-specific tree is hard to match from names alone. Embeddings and the LLM option are designed to improve it, but **I could not test the embedding model in my build environment** (no access to the model host), so I make no accuracy claim for it. A simulated LLM (answering from the labels with 12% noise on the top 1,500 texts) lifted L1/L3 to 84%/74%; this checked the plumbing and is not a real LLM result.
Other checks: B exclusions agree with all 3,000 seed labels (tuned on them, so optimistic); supplier grouping on A has homogeneity 1.00 (no wrong merges) but completeness 0.52 with no labels (0.90 on the holdout with labels); item grouping V-measure 0.96.

## Rehearsal for an unseen company
To imitate Company C, B's lines were given different column names, shuffled, and paired with B's tree under different header
names; the config was drafted automatically by `scripts/make_config.py` and run with no labels. Result vs B's 3,000 seed labels
(TF-IDF matcher only, no embeddings, no LLM): L1 80.5%, L2 78.6%, L3 30.5% of lines, include/exclude agreement 94.6%.
This is a weak, honest number for the free path; the embedding model and the LLM option are what should lift it, but neither
has been measured here. The LLM client probes a small list of model names (or `CLASSIFY_LLM_MODEL`) before use and silently
falls back to the free path if none works.

## Runtime
A (65,549 records): ~25 s without labels; ~3-6 min with 37k labels (the training step dominates). B (41,028 lines): ~9 s without labels, ~40 s with seed labels. Embeddings add roughly 1-3 minutes for 25k lines on CPU. All well inside 60 minutes per 25,000 lines.

## LLM cost
Not used in any reported result. If enabled: about 1 call per 20 unique texts, capped at 2,500 texts (~125 calls) per run, each prompt listing the tree; with a small model this is expected to cost cents to well under a dollar per run (estimate, not measured). Set `ANTHROPIC_API_KEY` and optionally `CLASSIFY_LLM_MODEL`; `--llm off` disables it.
