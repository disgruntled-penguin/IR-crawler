"""Detection pipeline: retrieve candidates from the index, verify them, decide provenance."""
import math

import numpy as np

from .. import corpus, embed, index, store
from . import candidates, signals, verify

VERIFY_TOP = 5
TIE_MARGIN = 0.05


class Context:
    """Everything a detection run needs, loaded once."""

    def __init__(self, stemming=False, con=None):
        self.con = con or store.connect()
        self.idx = index.load(stemming)
        self.feats = corpus.load_features()
        self.emb = embed.get()
        rows = {r["doc_id"]: r for r in store.iter_docs(self.con)}
        self.docs = [rows.get(d) for d in self.idx.doc_ids]
        self.model = verify.load()

    def doc(self, n):
        return self.docs[n]

    def num(self, doc_id):
        return self.idx.doc_num[doc_id]


def date_of(d):
    p = d.get("published")
    return None if p is None or (isinstance(p, float) and math.isnan(p)) else p


def direction(pair):
    """When dates cannot decide, the derived side is the one mostly covered by the other."""
    _, _, sims = pair.alignment()
    fwd = float(np.mean(sims.max(axis=1) >= signals.ALIGN_THRESHOLD))
    back = float(np.mean(sims.max(axis=0) >= signals.ALIGN_THRESHOLD))
    return fwd, back


def provenance(s, c, pair):
    ds, dc = date_of(s), date_of(c)
    if ds is not None and dc is not None:
        if dc < ds:
            return "derived", f"source published {(ds - dc) / 3600:.1f}h earlier"
        return "uncertain", "candidate is not earlier"
    fwd, back = direction(pair)
    if fwd - back > 0.2:
        return "derived_undated", f"dates missing; suspect covered {fwd:.2f} vs {back:.2f} reverse"
    return "uncertain", f"dates missing and coverage symmetric ({fwd:.2f}/{back:.2f})"


def analyse(ctx, n, k_verify=VERIFY_TOP, model=None, use_quotes=True, signal_names=None, cands=None):
    """Return a report dict for document number n."""
    model = model or ctx.model
    s = ctx.doc(n)
    if cands is None:
        cands, stats = candidates.retrieve(ctx.idx, n, s["text"], use_quotes=use_quotes)
    else:
        stats = {}
    top = cands[:k_verify]
    ctx.emb.prefetch([(s["doc_id"], s["text"])] + [(ctx.doc(c["doc"])["doc_id"], ctx.doc(c["doc"])["text"]) for c in top])
    verified = []
    for c in top:
        cd = ctx.doc(c["doc"])
        pair = signals.Pair(s, cd, ctx.feats, ctx.emb, c)
        sig = signals.compute(pair, signal_names)
        prob = verify.score(sig, model)
        verdict, why = provenance(s, cd, pair)
        verified.append({"doc": c["doc"], "doc_id": cd["doc_id"], "url": cd["url"], "site": cd["site"],
                         "retrieval": c["score"], "signals": sig, "prob": prob, "verdict": verdict, "why": why})
    verified.sort(key=lambda v: -v["prob"])
    best = verified[0] if verified else None
    if best:
        # Copies of one original verify almost equally well; the earliest of them is the likelier origin.
        close = [v for v in verified if v["prob"] >= best["prob"] - TIE_MARGIN and v["verdict"] != "uncertain"
                 and v["prob"] >= model["threshold"]]
        if close:
            best = min(close, key=lambda v: date_of(ctx.doc(v["doc"])) or float("inf"))
    flagged = bool(best and best["prob"] >= model["threshold"] and best["verdict"] != "uncertain")
    return {"doc": n, "doc_id": s["doc_id"], "url": s["url"], "stats": stats, "candidates": cands,
            "verified": verified, "flagged": flagged, "source": best if flagged else None}
