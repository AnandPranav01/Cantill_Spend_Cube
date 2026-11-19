# DECISIONS

What I found in the data and what I decided. Where the brief was silent I made an assumption and recorded it here.

## Company A
- **Two systems.** ERP (37,029 lines, net, ZAR) is the system of record; the invoice tool (28,520 lines) is gross. Tool lines that match an ERP line are DUPLICATE; the ERP record is counted.
- **VAT.** `vat_rate` appears as `15`, `0.15`, `15%`, `0` and blank. Values above 1 are read as percentages; blank -> 15% (brief); an explicit `0` is kept as 0.
- **Dates.** The tool has `dd/mm/yyyy`, `yyyy-mm-dd`, `dd-Mon-yy` and blanks; day-first is preferred. The tool's dates are not used for matching.
- **Status.** Approved/Paid count; Cancelled -> CANCELLED; On Hold/Rejected -> NOT_APPROVED. Any other status value would also be NOT_APPROVED (the config lists the approved values explicitly).
- **Matching rule found in the data.** For the same PO, tool and ERP lines agree on *unit price* even when quantity/amount differ slightly (weighed goods, partial invoices), so a pair is accepted on net amount (2%) or unit price. PO references are normalised to the 6+ digit number (`PO 4510...`, `4510.../10.0`); the `/10.0` suffix is used as a tie-break for the PO item. Without a PO, a match needs an identical amount plus supplier and text agreement.
- **Re-exports.** The same invoice exported twice (same invoice number, supplier, amount, quantity) is counted once; the later exports are DUPLICATE.
- **Tool-only lines** (no ERP match) are real spend and stay included.
- **Bug found while building:** a blank vendor id is NaN, and treating it as a value glued unrelated suppliers into one cluster. Fixed by an explicit "has a real value" check.
- **Golden supplier keys.** In the labels, ~20,000 lines with many different names and no vendor id share one golden supplier. I do not copy this: my clusters are conservative (a key is only shared for the same id or an almost identical name), so supplier completeness is lower than homogeneity. With labels available, known names are mapped to the client's key.

## Company B
- **No status column and no supplier names.** Exclusions come from text/account columns. Vendor id is the supplier key.
- **NON_PROCUREMENT** found in the seed labels: intercompany accounts (`IC06`, `IC Payables`, `I/C Payable`, ...) and sales-tax lines in the description. The IC pattern is in the B config (not in code); the generic tax/royalty/financing keywords are in code.
- **FX.** `currency_rate` is units of the line currency per 1 USD. USD lines are not converted. 62 lines carry an absurd given rate (e.g. EUR at 18.56), detected by comparing with the monthly table (more than 2x apart either way) and replaced by the table rate. Lines with no rate use the table rate for the posting month (nearest month if missing).
- **Dates.** All `posting_date` values parse. The FX month uses `posting_date`, then `invoice_date`. Invoice dates from earlier years exist (e.g. 2018) and are not used for FX.
- **Reversals** (equal and opposite amounts on the same line) are both kept and net to zero; I treat them as real postings, not duplicates.
- **3 lines have no amount** (and no rate): kept as included with spend 0 rather than invented.
- **Seed labels** are "useful, not perfect": they are used as training signal, not as truth.

## Classification and general
- Leaf level = the deepest level of each path; shorter paths leave the lower `l` columns blank. Trees deeper than 4 levels are cut to 4 in the output (internally all levels are used).
- Excluded rows get a category-free row, `spend` 0 and a confidence for the *exclusion* decision.
- A blank or useless description falls back on context columns (account text, material type, supplier, item) with a low confidence cap (0.5).
- Confidence is a calibrated-by-hand score from the model's probability, not a measured accuracy.
- The optional LLM and embedding model are never required; without them the tool runs fully offline.

## Unseen trees and new companies
- The tree layout is auto-detected: level columns (`category_l1`, `Level 1`, ...), one path column with `>`, `|`, `/` or `;`, or an
  adjacency list (`id`, `parent_id`, `name`). Leaves are the deepest nodes; shorter paths leave lower `l` columns blank.
- For C the config is drafted from column names and value shapes and must be reviewed by hand: it is the only per-company input.
- Rows with unusable text lean on supplier/item context and get a low confidence instead of a forced guess.
