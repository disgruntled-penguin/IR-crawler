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

    def __init__(self, s, c, feats, emb, retrieval=None):
        self.s, self.c = s, c
        self.feats = feats
        self.emb = emb
        self.retrieval = retrieval or {}
        self._align = None

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


SIGNALS = {
    "rare_cos": rare_cos,
    "shingle_containment": shingle_containment,
    "align_coverage": align_coverage,
    "align_mean": align_mean,
    "align_order": align_order,
    "quote_overlap": quote_overlap,
    "fact_overlap": fact_overlap,
}


def compute(pair, names=None):
    return {k: f(pair) for k, f in SIGNALS.items() if names is None or k in names}
