"""Ranked search with zone weighting and a static quality term: net(q, d) = relevance(q, d) + lambda * g(d)."""
import numpy as np

from . import text as tx

ZONE_WEIGHTS = {"title": 0.3, "body": 0.7}
LAMBDA = 0.3


def relevance(idx, query, k=200, zones=None):
    """Zone-weighted lnc.ltc cosine over title and body, scaled so the best hit scores 1."""
    zones = zones or ZONE_WEIGHTS
    toks = idx.tokens(query)
    counts = {}
    for t in tx.content_terms(toks) or toks:
        tid = idx.vocab.get(t)
        if tid is not None:
            counts[tid] = counts.get(tid, 0) + 1
    if not counts:
        return {}
    qvec = idx.query_vector(counts)
    acc = np.zeros(idx.N)
    for zone, wz in zones.items():
        z = np.zeros(idx.N)
        for t, wq in qvec.items():
            p = idx.post[zone].get(t)
            if p is not None:
                z[p.docs] += wq * (1 + np.log10(p.tfs))
        if zone == "body":
            z /= idx.norms
        else:
            z /= np.sqrt(np.maximum(idx.lengths["title"], 1))
        acc += wz * z
    top = np.argsort(-acc)[:k]
    best = acc[top[0]] or 1.0
    return {int(d): float(acc[d] / best) for d in top if acc[d] > 0}


def search(idx, query, g=None, lam=LAMBDA, k=10, origins=None, zones=None):
    """Top k (doc, net, relevance, g) tuples; g maps doc_id to originality (missing means 1)."""
    rel = relevance(idx, query, zones=zones)
    out = []
    for d, r in rel.items():
        if origins and idx.meta["origin"][d] not in origins:
            continue
        gd = 1.0 if g is None else g.get(idx.doc_ids[d], 1.0)
        out.append((d, r + lam * gd, r, gd))
    out.sort(key=lambda x: -x[1])
    return out[:k]
