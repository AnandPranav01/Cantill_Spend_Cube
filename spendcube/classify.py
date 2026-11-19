"""Classify lines into an unseen category tree.

Zero-label pipeline (works on any tree):
  1. zero-shot score of each unique line text against every leaf path: local sentence embeddings (if available)
     blended with a char-n-gram TF-IDF matcher (always available);
  2. self-training: the most confident lines become pseudo-labels for a linear model over text + context columns +
     supplier + item key, which learns the company's own vocabulary and account/supplier patterns;
  3. consistency: lines of the same item (and, weakly, the same supplier) share evidence;
  4. top-down decoding so l1..l4 are always a valid path of the tree; confidence in 0..1;
  5. optional LLM re-check of low-confidence, high-spend texts (only if an API key is configured).
Known labels (optional) simply replace pseudo-labels as training signal."""
from __future__ import annotations

import json
import os
import re
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.svm import LinearSVC

from .taxonomy import Taxonomy
from .textutil import norm_text, strip_accents

_ALPHA = re.compile(r"[a-z]{2,}")


def alpha_text(s: str) -> str:
    """Letters-only words (drops part numbers / codes), used for matching."""
    return " ".join(_ALPHA.findall(norm_text(s)))


def softmax(x: np.ndarray, t: float) -> np.ndarray:
    z = x / t
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _norm_rows(P):
    s = P.sum(axis=1, keepdims=True)
    s[s == 0] = 1
    return P / s


# --------------------------------------------------------------------------------------- embeddings
class Embedder:
    def __init__(self, mode: str, model: str, log=print):
        self.model = None
        self.log = log
        if mode == "off":
            return
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model)
            log(f"[classify] embeddings: {model}")
        except Exception as e:   # not installed / no internet for the first download
            if mode == "on":
                raise
            log(f"[classify] embeddings unavailable ({type(e).__name__}); using TF-IDF matcher only")

    @property
    def ok(self):
        return self.model is not None

    def encode(self, texts: list[str]) -> np.ndarray:
        v = self.model.encode([t[:300] for t in texts], batch_size=128, normalize_embeddings=True,
                              show_progress_bar=False)
        return np.asarray(v, dtype=np.float32)


# --------------------------------------------------------------------------------------- hierarchy decode
def _node_ids(tax: Taxonomy):
    """For every level k: node id per leaf (leaves shorter than k+1 act as their own node)."""
    out = []
    for k in range(min(tax.nlevels, 4)):
        keys = [p[:k + 1] if len(p) > k else p for p in tax.paths]
        u = {key: j for j, key in enumerate(dict.fromkeys(keys))}
        out.append(np.array([u[key] for key in keys]))
    return out


def decode(P: np.ndarray, tax: Taxonomy):
    """Top-down: choose the heaviest L1, then the heaviest L2 inside it, ... Returns leaf index per row."""
    nodes = _node_ids(tax)
    mask = np.ones_like(P, dtype=bool)
    Pm = P
    for ids in nodes:
        k = ids.max() + 1
        A = sp.csr_matrix((np.ones(len(ids)), (np.arange(len(ids)), ids)), shape=(len(ids), k))
        mass = (Pm * mask) @ A
        pick = np.asarray(mass.argmax(axis=1)).ravel()
        mask &= (ids[None, :] == pick[:, None])
    return (Pm * mask + mask * 1e-12).argmax(axis=1)


# --------------------------------------------------------------------------------------- main
class Classifier:
    def __init__(self, tax: Taxonomy, ccfg: dict, log=print):
        self.tax, self.cfg, self.log = tax, ccfg, log
        self.emb = Embedder(ccfg.get("embeddings", "auto"), ccfg.get("embedding_model"), log)

    # ---- step 1: zero-shot on unique texts
    def zero_shot(self, texts: list[str]) -> np.ndarray:
        tax = self.tax
        L = len(tax)
        n = len(texts)
        atxt = [alpha_text(t) for t in texts]
        cat_phr = [alpha_text(tax.leaf_phrase(i)) for i in range(L)]
        tf = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, min_df=1)
        tf.fit(atxt + cat_phr)
        S = (tf.transform(atxt) @ tf.transform(cat_phr).T).toarray().astype(np.float32)
        # word-level overlap adds precision for whole-word matches
        tw = TfidfVectorizer(analyzer="word", ngram_range=(1, 1), sublinear_tf=True, min_df=1)
        tw.fit(atxt + cat_phr)
        Sw = (tw.transform(atxt) @ tw.transform(cat_phr).T).toarray().astype(np.float32)
        P_tf = softmax(0.6 * S + 0.4 * Sw, float(self.cfg.get("tfidf_temperature", 0.07)))
        has_text = np.array([len(a) >= 3 for a in atxt])
        if self.emb.ok:
            E = self.emb.encode([t if len(a) >= 3 else "" for t, a in zip(texts, atxt)])
            C = self.emb.encode([tax.text(i) for i in range(L)])
            P_e = softmax(E @ C.T, float(self.cfg.get("emb_temperature", 0.04)))
            w = float(self.cfg.get("embedding_weight", 0.65))
            P = w * P_e + (1 - w) * P_tf
            self.E_text = E
        else:
            P = P_tf
            self.E_text = None
        P[~has_text] = 1.0 / L
        self.has_text = has_text
        return P.astype(np.float32)

    def fit_predict(self, lines: pd.DataFrame) -> pd.DataFrame:
        """lines columns: text, ctx (str of 'col=value' tokens), supplier_key, item_key, weight, is_target (bool),
        label (leaf idx or -1)."""
        tax, cfg, log = self.tax, self.cfg, self.log
        L = len(tax)
        t0 = time.time()
        n = len(lines)
        key = lines["text"].map(lambda s: norm_text(s)[:300])
        uniq, inv = np.unique(key.values, return_inverse=True)
        P_u = self.zero_shot(list(uniq))
        P0 = P_u[inv]
        informative = self.has_text[inv]
        log(f"[classify] {n} lines, {len(uniq)} unique texts, {L} leaves; zero-shot {time.time()-t0:.0f}s")

        # ---- features for the learned model
        atxt = lines["text"].map(alpha_text)
        vw = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), sublinear_tf=True, min_df=2, max_features=150000)
        vc = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, min_df=3, max_features=150000)
        cat_tokens = (lines["ctx"].fillna("") + " " + "sup=" + lines["supplier_key"].fillna("").astype(str)
                      + " item=" + lines["item_key"].fillna("").astype(str))
        vt = CountVectorizer(token_pattern=r"\S+", binary=True, min_df=2, lowercase=False)
        blocks = [vw.fit_transform(atxt), vc.fit_transform(atxt), 0.7 * vt.fit_transform(cat_tokens).astype(np.float32)]
        if self.emb.ok:
            blocks.append(sp.csr_matrix(self.E_text[inv] * 0.8))
        X = sp.hstack(blocks).tocsr()

        label = lines["label"].values.astype(int)
        weight = lines["weight"].values.astype(float)
        is_t = lines["is_target"].values.astype(bool)
        labelled = label >= 0
        lam = float(cfg.get("model_weight_labels", 0.85)) if labelled.sum() >= 200 else float(cfg.get("model_weight", 0.55))
        P = P0.copy()
        rounds = int(cfg.get("self_training_rounds", 2))
        for r in range(max(rounds, 1)):
            conf = P.max(axis=1)
            pred = P.argmax(axis=1)
            y = np.full(n, -1)
            w = np.zeros(n)
            y[labelled] = label[labelled]
            w[labelled] = 1.0
            cand = np.where(~labelled & informative & (conf >= np.quantile(conf[informative], 0.45 if r else 0.55)))[0] if informative.any() else np.array([], int)
            if len(cand):
                # cap per predicted class so large categories do not swamp small ones
                cnt = np.bincount(pred[cand], minlength=L)
                cap = max(40, int(3 * np.median(cnt[cnt > 0]))) if (cnt > 0).any() else 40
                keep = []
                for c in np.unique(pred[cand]):
                    ids = cand[pred[cand] == c]
                    if len(ids) > cap:
                        ids = ids[np.argsort(-conf[ids])[:cap]]
                    keep.extend(ids.tolist())
                keep = np.array(keep, int)
                y[keep] = pred[keep]
                w[keep] = 0.5
            tr = y >= 0
            if tr.sum() < 20 or len(np.unique(y[tr])) < 2:
                log("[classify] too little training signal for the learned model; zero-shot only")
                break
            m = LinearSVC(C=float(cfg.get("svm_c", 0.5)), class_weight="balanced")
            m.fit(X[tr], y[tr], sample_weight=w[tr])
            D = m.decision_function(X)
            if D.ndim == 1:
                D = np.stack([-D, D], axis=1)
            Pm = np.zeros((n, L), dtype=np.float32)
            Pm[:, m.classes_] = softmax(D, 0.25)
            Pm[:, np.setdiff1d(np.arange(L), m.classes_)] = 0
            Pm = _norm_rows(Pm + 1e-9)
            lam_row = np.where(informative, lam, 1.0)[:, None]
            P = (1 - lam_row) * P0 + lam_row * Pm
            P[labelled] = np.eye(L, dtype=np.float32)[label[labelled]] * 0.9 + 0.1 * P[labelled]
            log(f"[classify] round {r+1}: trained on {int(tr.sum())} lines ({int(labelled.sum())} labelled) {time.time()-t0:.0f}s")
        P = self._share(P, lines, informative)
        return self._finish(P, lines, informative)

    # ---- step 3: item / supplier consistency
    def _group_mean(self, P, codes):
        ok = codes >= 0
        k = codes.max() + 1 if ok.any() else 0
        if k == 0:
            return None, None
        G = sp.csr_matrix((np.ones(ok.sum()), (codes[ok], np.where(ok)[0])), shape=(k, len(codes)))
        cnt = np.asarray(G.sum(axis=1)).ravel()
        M = (G @ P) / np.maximum(cnt, 1)[:, None]
        return M, cnt

    def _share(self, P, lines, informative):
        cfg = self.cfg
        out = P.copy()
        for col, wgt, minc, weak_only in (("item_key", float(cfg.get("item_share", 0.4)), 2, False),
                                         ("supplier_key", float(cfg.get("supplier_share", 0.25)), 4, True)):
            s = lines[col].fillna("")
            codes = pd.Series(pd.factorize(s.where(~s.str.startswith(("UNKNOWN", "NA-")) & (s != ""), None))[0]).values
            M, cnt = self._group_mean(out, codes)
            if M is None:
                continue
            has = (codes >= 0)
            has &= cnt[np.clip(codes, 0, None)] >= minc
            conf = out.max(axis=1)
            sel = has & ((~informative | (conf < 0.5)) if weak_only else np.ones(len(out), bool))
            idx = np.where(sel)[0]
            w = wgt * (1.0 if not weak_only else 1.0)
            out[idx] = (1 - w) * out[idx] + w * M[codes[idx]]
        return _norm_rows(out)

    def _finish(self, P, lines, informative):
        tax = self.tax
        leaf = decode(P, tax)
        pc = P[np.arange(len(P)), leaf]
        conf = np.clip(np.sqrt(pc), 0.01, 0.99)
        conf = np.where(informative, conf, np.minimum(conf, 0.5))
        res = pd.DataFrame({"leaf": leaf, "confidence": conf.round(3)}, index=lines.index)
        res.attrs["P"] = P
        return res


# --------------------------------------------------------------------------------------- optional LLM labelling
def _call_llm(prompt: str, key: str, model: str, max_tokens: int = 600) -> str:
    body = json.dumps({"model": model, "max_tokens": max_tokens,
                       "system": "You classify purchase-order lines into a fixed category list. Reply with JSON only.",
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request("https://api.anthropic.com/v1/messages", data=body, headers={
        "x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())["content"][0]["text"]


def llm_label(lines: pd.DataFrame, tax: Taxonomy, ccfg: dict, log=print):
    """Optional: label the highest-spend unique texts with an LLM (in batches, all leaves listed in the prompt).
    The labels then train the linear model that classifies everything else. Returns {line_index: leaf} or {}.
    Needs an API key in the env var named by classification.llm.api_key_env (default ANTHROPIC_API_KEY); without
    a key (or with llm.enabled: off) this does nothing and the free path is used."""
    lc = ccfg.get("llm") or {}
    key = os.environ.get(lc.get("api_key_env", "ANTHROPIC_API_KEY"), "")
    if lc.get("enabled", "auto") in (False, "off", "false", "False") or not key:
        return {}
    if len(tax) > 400:
        log("[llm] tree too large to list in a prompt; skipping LLM labelling")
        return {}
    explicit = lc.get("model") or os.environ.get("CLASSIFY_LLM_MODEL")
    model = None
    for cand_model in ([explicit] if explicit else ["claude-haiku-4-5", "claude-haiku-5-5", "claude-sonnet-4-5"]):
        try:            # tiny probe so a wrong / retired model name does not silently disable the LLM
            _call_llm("Reply with the JSON {}", key, cand_model, max_tokens=10)
            model = cand_model
            break
        except Exception as e:
            log(f"[llm] model {cand_model} not usable: {repr(e)[:120]}")
    if model is None:
        log("[llm] no usable model; continuing without the LLM")
        return {}
    max_unique = int(lc.get("max_unique", 2500))
    batch = int(lc.get("batch_size", 20))
    cand = lines[lines["is_target"] & (lines["label"] < 0)]
    if cand.empty:
        return {}
    d = pd.DataFrame({"text": cand["text"].fillna("").astype(str), "ctx": cand["ctx"].fillna(""),
                      "spend": cand["spend"].abs()})
    d["k"] = d["text"].map(norm_text) + "|" + d["ctx"]
    d = d[(d["text"].map(alpha_text).str.len() >= 3) | (d["ctx"] != "")]
    g = d.assign(i=d.index).groupby("k").agg(text=("text", "first"), ctx=("ctx", "first"),
                                             spend=("spend", "sum"), n=("i", "size"), idx=("i", list))
    g = g.sort_values(["spend", "n"], ascending=False).head(max_unique).reset_index()
    cats = "\n".join(f"{j+1}. {' > '.join(p)}" for j, p in enumerate(tax.paths))
    batches = [g.iloc[i:i + batch] for i in range(0, len(g), batch)]
    errors = []

    def work(b):
        items = "\n".join(f"{n+1}. {r.text[:200]}" + (f"  [{r.ctx[:120]}]" if r.ctx else "") for n, r in enumerate(b.itertuples()))
        prompt = (f"Categories:\n{cats}\n\nFor each numbered line, give the number of the single best category "
                  f"(0 if you cannot tell). Reply as JSON: {{\"1\": 12, \"2\": 0, ...}}\n\nLines:\n{items}")
        for attempt in range(3):
            try:
                txt = _call_llm(prompt, key, model)
                m = re.search(r"\{.*\}", txt, re.S)
                ans = json.loads(m.group(0))
                out = {}
                for n, r in enumerate(b.itertuples()):
                    v = int(ans.get(str(n + 1), 0))
                    if 1 <= v <= len(tax):
                        out[r.k] = v - 1
                return out
            except Exception as e:
                errors.append(repr(e)[:150])
                time.sleep(1.5 * (attempt + 1))
        return {}

    got = {}
    with ThreadPoolExecutor(int(lc.get("workers", 8))) as ex:
        for o in ex.map(work, batches):
            got.update(o)
    if errors and not got:
        log(f"[llm] no labels obtained (first error: {errors[0]}); continuing without the LLM")
        return {}
    kmap = dict(zip(g["k"], g["idx"]))
    res = {}
    for k, leaf in got.items():
        for i in kmap[k]:
            res[i] = leaf
    log(f"[llm] {model}: labelled {len(got)} of {len(g)} unique texts ({len(res)} lines), {len(batches)} calls, {len(errors)} errors")
    return res
