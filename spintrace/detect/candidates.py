"""Candidate source retrieval: the suspect's rare terms as a query, its quotes as phrase queries.

Rewrites change wording but keep names, numbers and quotes, which are high-idf terms and exact
phrases. Index elimination keeps only the suspect's highest-weight rare terms, so the query
touches a small part of the index, and a parametric filter keeps only earlier articles.
"""
import numpy as np

from .. import text as tx

QUERY_TERMS = 30
MIN_DF = 2
TOP_K = 20
QUOTE_WINDOW = 6
QUOTE_WEIGHT = 0.5
FACT_BOOST = 2.0


def fact_terms(idx, text):
    """Index terms that come from the suspect's names and numbers, the parts a rewrite keeps."""
    names, nums = tx.facts(text)
    out = set()
    for f in names | nums:
        for t in idx.tokens(f):
            tid = idx.vocab.get(t)
            if tid is not None and t not in tx.STOPWORDS:
                out.add(tid)
    return out


def rare_term_query(idx, n, m=QUERY_TERMS, facts=None):
    """Top m body terms of doc n by tf-idf (fact terms boosted), skipping stopwords and terms only this doc has.

    Returns {term id: query weight multiplier * tf} so the boost carries into the ltc query vector.
    """
    terms = idx.doc_terms(n)
    facts = facts or set()
    scored = []
    for t, tf in terms.items():
        if idx.df[t] < MIN_DF or idx.terms[t] in tx.STOPWORDS:
            continue
        boost = FACT_BOOST if t in facts else 1.0
        scored.append(((1 + np.log10(tf)) * idx.idf[t] * boost, t, tf * boost))
    scored.sort(reverse=True)
    return {t: w for _, t, w in scored[:m]}


def earlier_mask(idx, n, require_dates=True):
    """Parametric filter: other sites, published before the suspect. Undated documents stay in, flagged later."""
    pub = idx.meta["published"]
    mask = idx.meta["site"] != idx.meta["site"][n]
    mask[n] = False
    t = pub[n]
    if require_dates and not np.isnan(t):
        mask &= ~(pub >= t)
    return mask


def quote_hits(idx, text, allowed):
    """For each quote, docs containing any QUOTE_WINDOW-token window of it as an exact phrase."""
    hits = {}
    qs = tx.quotes(text)
    for q in qs:
        toks = idx.tokens(q)
        found = set()
        step = max(1, QUOTE_WINDOW // 2)
        for i in range(0, max(1, len(toks) - QUOTE_WINDOW + 1), step):
            found |= idx.phrase(toks[i:i + QUOTE_WINDOW])
        for d in found:
            if allowed[d]:
                hits[d] = hits.get(d, 0) + 1
    return {d: c / len(qs) for d, c in hits.items()}, len(qs)


def retrieve(idx, n, text, k=TOP_K, m=QUERY_TERMS, use_quotes=True, require_dates=True, exclude=None,
             use_facts=True):
    """Return (candidates, stats). Each candidate: dict(doc, rare_cos, quote_frac, score)."""
    allowed = earlier_mask(idx, n, require_dates)
    if exclude is not None:
        allowed &= ~exclude
    q = rare_term_query(idx, n, m, fact_terms(idx, text) if use_facts else None)
    qvec = idx.query_vector(q)
    cos = dict(idx.cosine(qvec, allowed, k=k))
    qh, n_quotes = quote_hits(idx, text, allowed) if use_quotes else ({}, 0)
    docs = set(cos) | set(qh)
    cands = [{"doc": d, "rare_cos": cos.get(d, 0.0), "quote_frac": qh.get(d, 0.0)} for d in docs]
    for c in cands:
        c["score"] = c["rare_cos"] + QUOTE_WEIGHT * c["quote_frac"]
    cands.sort(key=lambda c: -c["score"])
    touched = int(sum(len(idx.post["body"][t].docs) for t in qvec))
    stats = {"query_terms": [(idx.terms[t], round(float(idx.idf[t]), 3)) for t in qvec], "postings_touched": touched,
             "n_quotes": n_quotes, "allowed": int(allowed.sum())}
    return cands[:k], stats
