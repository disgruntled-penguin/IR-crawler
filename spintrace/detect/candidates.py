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


def text_terms(idx, text):
    """{term id: tf} for a text that is not in the index, over the index's vocabulary."""
    out = {}
    for t in idx.tokens(text):
        tid = idx.vocab.get(t)
        if tid is not None:
            out[tid] = out.get(tid, 0) + 1
    return out


def rare_term_query(idx, n, m=QUERY_TERMS, facts=None, terms=None):
    """Top m body terms of doc n (or of the given {term id: tf}) by tf-idf, fact terms boosted, skipping
    stopwords and terms only one doc has.

    Returns {term id: query weight multiplier * tf} so the boost carries into the ltc query vector.
    """
    terms = terms if terms is not None else idx.doc_terms(n)
    facts = facts or set()
    scored = []
    for t, tf in terms.items():
        if idx.df[t] < MIN_DF or idx.terms[t] in tx.STOPWORDS:
            continue
        boost = FACT_BOOST if t in facts else 1.0
        scored.append(((1 + np.log10(tf)) * idx.idf[t] * boost, t, tf * boost))
    scored.sort(reverse=True)
    return {t: w for _, t, w in scored[:m]}


def budget_query(idx, n, m, mode, seed=0):
    """Query of m body terms chosen by mode: 'rare' (highest idf), 'common' (lowest idf) or 'random'."""
    terms = [(t, tf) for t, tf in idx.doc_terms(n).items()
             if idx.df[t] >= MIN_DF and idx.terms[t] not in tx.STOPWORDS]
    if mode == "random":
        rng = np.random.RandomState(seed + n)
        pick = rng.permutation(len(terms))[:m]
        return {terms[i][0]: terms[i][1] for i in pick}
    terms.sort(key=lambda x: idx.idf[x[0]], reverse=(mode == "rare"))
    return dict(terms[:m])


def earlier_mask(idx, n, require_dates=True):
    """Parametric filter: other sites, published before the suspect. Undated documents stay in, flagged later."""
    mask = mask_before(idx, idx.meta["published"][n], idx.meta["site"][n], require_dates)
    mask[n] = False
    return mask


def mask_before(idx, published, site=None, require_dates=True):
    """The same filter for a suspect outside the index: other sites, published before `published`."""
    pub = idx.meta["published"]
    mask = np.ones(idx.N, dtype=bool) if site is None else idx.meta["site"] != site
    if require_dates and published is not None and not np.isnan(published):
        mask &= ~(pub >= published)
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
    return _retrieve(idx, n, None, text, allowed, k, m, use_quotes, exclude, use_facts)


def retrieve_text(idx, text, published=None, site=None, k=TOP_K, m=QUERY_TERMS, use_quotes=True,
                  require_dates=True, exclude=None, use_facts=True):
    """retrieve() for a suspect that is not in the index (a translation back to English, a mashup)."""
    allowed = mask_before(idx, published, site, require_dates)
    return _retrieve(idx, None, text_terms(idx, text), text, allowed, k, m, use_quotes, exclude, use_facts)


def _retrieve(idx, n, terms, text, allowed, k, m, use_quotes, exclude, use_facts):
    if exclude is not None:
        allowed &= ~exclude
    q = rare_term_query(idx, n, m, fact_terms(idx, text) if use_facts else None, terms)
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
