# DECISIONS

This file lists the problems I found in the data and what I decided to do about each one.
The brief did not answer every question, so where it was unclear I made the most sensible choice and wrote it here.

**How I decide when something is unclear:** I look at the real data first, then pick the choice that follows the brief's rules. If I am still unsure, I keep the row and give it a low confidence. I do not make up a value.

---

## Company A (two systems: ERP and invoice tool)

**Problem 1: The same purchase is in both systems.**
The ERP file has 37,029 lines. The invoice tool has 28,520 lines. Many tool lines are the same purchase as an ERP line.
*What I did:* I match each tool line to an ERP line. The ERP line is counted. The tool line is marked `DUPLICATE`. Result: 24,149 tool lines were marked `DUPLICATE`.
I could not check this number against labels, because the labels only cover ERP lines.

**Problem 2: PO numbers are written in different ways.**
The same PO can look like `PO 4510123456`, `4510123456/10.0`, or be missing.
*What I did:* I keep only the long number (6 or more digits). The part after `/` is used only to break a tie.

**Problem 3: Amounts and quantities do not always match exactly.**
For weighed goods and partial invoices, the amount and quantity change a little, but the price per unit stays the same.
*What I did:* A tool line and an ERP line match if the amount is within 2% **or** the unit price is the same. If there is no PO, I need an identical amount plus the same supplier and similar text.

**Problem 4: The same invoice appears twice in the tool.**
*What I did:* If invoice number, supplier, amount and quantity are all the same, I count it once. The later copies are `DUPLICATE`.

**Problem 5: Tool lines with no ERP match.**
*What I did:* I keep them. They are real spend that only exists in the tool.

**Problem 6: The tool amounts include VAT, and the VAT rate is messy.**
The rate appears as `15`, `0.15`, `15%`, `0`, or blank.
*What I did:* Numbers above 1 are read as a percent. Blank becomes 15% (the brief's rule). A real `0` stays 0. Net amount = gross amount divided by (1 + rate).

**Problem 7: Dates come in several formats.**
The tool uses `dd/mm/yyyy`, `yyyy-mm-dd`, `dd-Mon-yy`, and some are blank.
*What I did:* The tool tries 14 formats, day first. Dates that cannot be read are flagged. I do not use the tool dates to match records.

**Problem 8: Status.**
*What I did:* Approved and Paid count. Cancelled becomes `CANCELLED`. On Hold, Rejected and any other value become `NOT_APPROVED`.

**Problem 9: A bug I found and fixed.**
A blank vendor ID was read as the text "nan", which glued 32 different suppliers into one. I fixed it so a blank is treated as "no value".

**Problem 10: The client's own supplier grouping is very broad.**
In the labels, about 20,000 lines with many different names share one supplier key.
*What I did:* I did not copy this. My tool only merges suppliers when the vendor ID is the same or the names are almost identical. This means I never merge two different suppliers by mistake, but I may leave some real duplicates unmerged.

---

## Company B (many ERPs and 25 currencies)

**Problem 1: No supplier names, only vendor IDs.**
*What I did:* The vendor ID is the supplier key.

**Problem 2: No status column.**
*What I did:* I decide exclusions from the description and the account text instead.

**Problem 3: Some lines are not real supplier spend.**
In the seed labels these are intercompany accounts (like `IC06`, `IC Payables`, `I/C Payable`) and sales-tax lines.
*What I did:* The intercompany pattern is written in B's config file, not in the code. General words like tax, royalty and interest are in the code as simple rules. Result: 2,494 lines are `NON_PROCUREMENT`.

**Problem 4: Amounts are in 25 currencies.**
`currency_rate` is the number of the line's currency per 1 USD. USD lines are not changed.
*What I did:* USD amount = amount divided by rate. If the rate is missing, I use the monthly rate from the FX table for the posting month.

**Problem 5: Some given rates are wrong.**
62 lines have an impossible rate (for example, EUR at 18.56).
*What I did:* If the given rate is more than 2 times different from the monthly table, I use the table rate instead.

**Problem 6: Dates.**
All posting dates could be read. I use the posting date for the FX month, and the invoice date only if the posting date is missing. Some invoice dates are old (for example 2018), so I do not use them for FX.

**Problem 7: Reversals (equal and opposite amounts).**
*What I did:* I keep both. They add up to zero. I treat them as real postings, not duplicates.

**Problem 8: 3 lines have no amount.**
*What I did:* I keep them with spend 0. I do not invent a number.

**Problem 9: The seed labels are "useful, not perfect".**
*What I did:* I use them as training help, not as the truth.

---

## Descriptions and categories (all companies)

**Problem: Many descriptions are short, in many languages, or useless ("Product", "Parts").**
*What I did:* The tool also looks at the account text, material type, supplier and item. If the text is useless, the confidence is capped at 0.5. I would rather give a low score than a confident wrong answer.

**Problem: Every company has its own category tree.**
*What I did:* The tool reads the tree from a file. It accepts three layouts: one column per level, one column with a path like `A > B > C`, or an id / parent id list. The output always uses the exact names from the tree and leaves lower levels blank when the path is shorter. Trees deeper than 4 levels are cut to 4 in the output.

**Problem: Excluded rows.**
*What I did:* They get a reason code, spend 0 and no category. Their confidence shows how sure I am about the exclusion.

**Problem: Company C has no labels and I cannot see it in advance.**
*What I did:* A small script drafts C's config from its column names and checks the values. I must read and fix this config by hand. That config is the only thing that changes for a new company, never the code.

---

## Things I am not sure about

- The intercompany rule for B was set by looking at the seed labels, so the 100% match on those labels is too optimistic.
- I could not check A's duplicate matching against any labels.
- Confidence is a score I made from the model's probability. It is not a measured accuracy.
- I measured the free method (text matching) on its own. I did not measure how much the embedding model helps.
