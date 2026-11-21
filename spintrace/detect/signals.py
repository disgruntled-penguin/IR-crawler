"""Independent verification signals for a (suspect, candidate) pair. Each returns a float in [0, 1].

Same-event reporting shares names and numbers too, so the signals that separate a rewrite from
independent coverage are sentence-level: how much of the suspect aligns to the candidate, and
whether the aligned sentences keep the candidate's order.
"""
from bisect import bisect_left

import numpy as np

from .. import dedup, text as tx

ALIGN_THRESHOLD = 0.62


class Pair:
    """Lazily computed shared state for one pair, so each signal stays a small function."""

    def __init__(self, s, c, feats, emb, retrieval=None, idx=None):
        self.s, self.c = s, c
        self.feats = feats
        self.emb = emb
        self.retrieval = retrieval or {}
        self.idx = idx
        self._align = None
        self._facts = {}

    def alignment(self):
        """Best-matching candidate sentence for every suspect sentence (cosine of sentence embeddings)."""
        if self._align is None:
            _, es = self.emb.doc(self.s["doc_id"], self.s["text"])
            _, ec = self.emb.doc(self.c["doc_id"], self.c["text"])
            sims = es.astype(np.float32) @ ec.astype(np.float32).T
            self._align = (sims.max(axis=1), sims.argmax(axis=1), sims)
        return self._align


def _lis(seq):
    tails = []
    for x in seq:
        i = bisect_left(tails, x)
        if i == len(tails):
            tails.append(x)
        else:
            tails[i] = x
    return len(tails)


def rare_cos(p):
    return float(p.retrieval.get("rare_cos", 0.0))


def shingle_containment(p):
    return dedup.containment(p.feats[p.s["doc_id"]]["shingles"], p.feats[p.c["doc_id"]]["shingles"])


def align_coverage(p):
    best, _, _ = p.alignment()
    return float(np.mean(best >= ALIGN_THRESHOLD))


def align_mean(p):
    best, _, _ = p.alignment()
    return float(np.clip(best.mean(), 0, 1))


def align_order(p):
    """Share of aligned suspect sentences whose matches appear in the candidate's order (LIS / matched)."""
    best, arg, _ = p.alignment()
    matched = arg[best >= ALIGN_THRESHOLD].tolist()
    if len(matched) < 2:
        return 0.0
    return _lis(matched) / len(matched)


def back_coverage(p):
    """Share of the candidate's sentences matched by the suspect; a rewrite covers most of its source."""
    _, _, sims = p.alignment()
    return float(np.mean(sims.max(axis=0) >= ALIGN_THRESHOLD))


def ordered_coverage(p):
    """Longest in-order chain of aligned sentences as a share of the whole suspect: copying keeps the source's
    structure end to end, independent reports of one event share facts but not their sequence."""
    best, arg, _ = p.alignment()
    matched = arg[best >= ALIGN_THRESHOLD].tolist()
    return _lis(matched) / len(best) if len(best) else 0.0


def aligned_word_overlap(p):
    """Mean content-word Jaccard of aligned sentence pairs: a rewrite inherits its source's phrasing even when it
    paraphrases; an article written independently from the same facts does not."""
    best, arg, _ = p.alignment()
    ss, _ = p.emb.doc(p.s["doc_id"], p.s["text"])
    cs, _ = p.emb.doc(p.c["doc_id"], p.c["text"])
    vals = []
    for i in range(len(best)):
        if best[i] >= ALIGN_THRESHOLD:
            a = set(tx.content_terms(tx.tokens(ss[i])))
            b = set(tx.content_terms(tx.tokens(cs[arg[i]])))
            vals.append(len(a & b) / len(a | b) if a | b else 0.0)
    return float(np.mean(vals)) if vals else 0.0


def quote_overlap(p):
    qs = tx.quotes(p.s["text"])
    if not qs:
        return float(p.retrieval.get("quote_frac", 0.0))
    ctoks = tx.tokens(p.c["text"])
    grams = {tuple(ctoks[i:i + 5]) for i in range(len(ctoks) - 4)}
    hit = 0
    for q in qs:
        t = tx.tokens(q)
        if any(tuple(t[i:i + 5]) in grams for i in range(max(1, len(t) - 4))):
            hit += 1
    return hit / len(qs)


def fact_overlap(p):
    """Share of the suspect's names and numbers that the candidate also contains."""
    sn, snum = tx.facts(p.s["text"])
    cn, cnum = tx.facts(p.c["text"])
    s = sn | snum
    return len(s & (cn | cnum)) / len(s) if s else 0.0


def _idf(p, term):
    if p.idx is None:
        return 1.0
    t = p.idx.vocab.get(term)
    return float(p.idx.idf[t]) if t is not None else float(np.log10(max(p.idx.N, 2)))


def fact_units(p, text):
    """Names, numbers and quotes of a text as (kind, token tuple, idf weight); the weight is the unit's rarest token."""
    if text not in p._facts:
        names, nums = tx.facts(text)
        units = []
        for f in sorted(names | nums):
            toks = tuple(t for t in tx.tokens(f) if t not in tx.STOPWORDS)
            if toks:
                units.append(("fact", toks, max(_idf(p, t) for t in toks)))
        for q in tx.quotes(text):
            toks = tuple(tx.tokens(q))
            units.append(("quote", toks, max(_idf(p, t) for t in toks)))
        p._facts[text] = units
    return p._facts[text]


def _has(unit, toks, grams):
    kind, t, _ = unit
    if kind == "quote":
        return any(tuple(t[i:i + 5]) in grams for i in range(max(1, len(t) - 4)))
    return all(x in toks for x in t)


def _fact_share(p, a, b):
    units = fact_units(p, a)
    btoks = tx.tokens(b)
    bset, grams = set(btoks), {tuple(btoks[i:i + 5]) for i in range(len(btoks) - 4)}
    total = sum(w for _, _, w in units)
    return sum(u[2] for u in units if _has(u, bset, grams)) / total if total else 0.0


def fact_idf_overlap(p):
    """Order-free: idf-weighted share of the suspect's names, numbers and quotes that the candidate also has. A
    listicle or Q&A reshapes and reorders the sentences, but each fact survives where it was."""
    return _fact_share(p, p.s["text"], p.c["text"])


def fact_back_overlap(p):
    """The same share in the other direction: a reformatted copy keeps nearly all of its source's facts, an article
    written from a fact sheet only the handful the sheet listed."""
    return _fact_share(p, p.c["text"], p.s["text"])


def fact_sentence_coverage(p):
    """Share of the suspect's fact-bearing sentences whose facts (by idf weight) are mostly in the candidate."""
    ss, _ = p.emb.doc(p.s["doc_id"], p.s["text"])
    ctoks = tx.tokens(p.c["text"])
    cset, grams = set(ctoks), {tuple(ctoks[i:i + 5]) for i in range(len(ctoks) - 4)}
    hit = n = 0
    for sent in ss:
        units = fact_units(p, sent)
        total = sum(w for _, _, w in units)
        if not total:
            continue
        n += 1
        hit += sum(u[2] for u in units if _has(u, cset, grams)) / total >= 0.5
    return hit / n if n else 0.0


# Not used by the default verifier; fit only where named (the format-change experiment).
OPTIONAL = {
    "fact_idf_overlap": fact_idf_overlap,
    "fact_back_overlap": fact_back_overlap,
    "fact_sentence_coverage": fact_sentence_coverage,
}

SIGNALS = {
    "rare_cos": rare_cos,
    "shingle_containment": shingle_containment,
    "align_coverage": align_coverage,
    "align_mean": align_mean,
    "align_order": align_order,
    "back_coverage": back_coverage,
    "ordered_coverage": ordered_coverage,
    "aligned_word_overlap": aligned_word_overlap,
    "quote_overlap": quote_overlap,
    "fact_overlap": fact_overlap,
}


def compute(pair, names=None):
    """The default signals, or exactly the named ones (which may include OPTIONAL signals)."""
    if names is None:
        return {k: f(pair) for k, f in SIGNALS.items()}
    return {k: f(pair) for k, f in {**SIGNALS, **OPTIONAL}.items() if k in names}
