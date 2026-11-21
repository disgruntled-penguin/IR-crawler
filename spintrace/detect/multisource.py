"""Multi-source tracing: one article stitched from several originals, recovered by explain-away retrieval.

The single-source pipeline keeps the strongest source and stops, so a second source hides behind the
first: its sentences are a minority of the suspect and its rare terms are outweighed in the query.
Here each round builds the query only from the sentences no found source explains yet (their rare
terms, names, numbers and quotes), verifies the candidates against that residual, keeps the best one
and marks the sentences it aligns to as covered. Rounds stop when little is left unexplained, when
nothing verifies, or after MAX_SOURCES.

The midterm verifier was fit on whole-document pairs, where a true source covers most of the suspect;
a 30% source of a mashup does not. The round verifier therefore scores coverage relative to the
residual (a 30% source covers almost all of what is left once the 70% source is explained), plus the
order, contiguity and inherited wording that separate copying from independent same-event reports.
"""
import json
import math
import time

import numpy as np

from .. import config, dedup, text as tx
from . import candidates, signals

MAX_SOURCES = 4
K_VERIFY = 5
MIN_NEW = 2
TIE_MARGIN = 0.05
MODEL_PATH = config.RESULTS / "multisource_verifier.json"

FEATURES = ["rare_cos", "res_coverage", "res_ordered", "res_run", "res_mean", "res_word_overlap",
            "res_fact_overlap", "res_quote", "res_containment", "back_coverage"]
# Used only until `spintrace eval-multi` has fit the round verifier on dev mashups and controls.
DEFAULT = {"features": FEATURES, "weights": [2.0, 3.0, 3.0, 2.0, 1.0, 3.0, 3.0, 0.5, 1.0, 1.0], "bias": -8.0,
           "threshold": 0.5, "cover": signals.ALIGN_THRESHOLD, "stop_share": 0.2, "source": "default"}


def load_model():
    if MODEL_PATH.exists():
        return json.loads(MODEL_PATH.read_text())
    return DEFAULT


def score(sig, model):
    z = model["bias"] + sum(w * sig.get(f, 0.0) for f, w in zip(model["features"], model["weights"]))
    return 1 / (1 + math.exp(-z))


class Suspect:
    """A text to trace. n is its index number when it is indexed (a crawled page), else None."""

    def __init__(self, doc_id, text, published=None, site=None, n=None, url=None, title=None):
        self.doc_id, self.text, self.n, self.url, self.title = doc_id, text, n, url, title
        self.published = None if published is None or (isinstance(published, float) and math.isnan(published)) else published
        self.site = site

    @classmethod
    def from_index(cls, ctx, n):
        d = ctx.doc(n)
        return cls(d["doc_id"], d["text"], d.get("published"), d["site"], n, d.get("url"), d.get("title"))

    def as_doc(self):
        return {"doc_id": self.doc_id, "text": self.text, "published": self.published, "site": self.site,
                "url": self.url}


def _run_length(idx_r, arg_r, hit_r):
    """Longest run of consecutive residual sentences that all align, with the candidate positions increasing."""
    best = cur = 0
    prev = None
    for i in range(len(idx_r)):
        if hit_r[i] and (prev is None or arg_r[i] > prev):
            cur += 1
        elif hit_r[i]:
            cur = 1
        else:
            cur = 0
        prev = arg_r[i] if hit_r[i] else None
        best = max(best, cur)
    return best


class CandidateCache:
    """Per-candidate text features reused across rounds and suspects."""

    def __init__(self, size=4000):
        self.size = size
        self.data = {}

    def get(self, doc_id, text):
        hit = self.data.get(doc_id)
        if hit is None:
            toks = tx.tokens(text)
            names, nums = tx.facts(text)
            hit = {"facts": names | nums, "grams": {tuple(toks[i:i + 5]) for i in range(len(toks) - 4)},
                   "shingles": dedup.shingles(text)}
            if len(self.data) >= self.size:
                self.data.pop(next(iter(self.data)))
            self.data[doc_id] = hit
        return hit


_cand_cache = CandidateCache()


def round_signals(pair, resid, retrieval, res_facts, res_quotes, res_shingles):
    """Signals of one (suspect residual, candidate) pair. Each is in [0, 1]."""
    best, arg, sims = pair.alignment()
    nr = int(resid.sum())
    if nr == 0:
        return {f: 0.0 for f in FEATURES}
    hit = resid & (best >= signals.ALIGN_THRESHOLD)
    ridx = np.nonzero(resid)[0]
    matched = arg[hit].tolist()
    ss, _ = pair.emb.doc(pair.s["doc_id"], pair.s["text"])
    cs, _ = pair.emb.doc(pair.c["doc_id"], pair.c["text"])
    wo = []
    for i in np.nonzero(hit)[0]:
        a = set(tx.content_terms(tx.tokens(ss[i])))
        b = set(tx.content_terms(tx.tokens(cs[arg[i]])))
        wo.append(len(a & b) / len(a | b) if a | b else 0.0)
    cf = _cand_cache.get(pair.c["doc_id"], pair.c["text"])
    qhit = 0
    for q in res_quotes:
        t = tx.tokens(q)
        if any(tuple(t[i:i + 5]) in cf["grams"] for i in range(max(1, len(t) - 4))):
            qhit += 1
    return {
        "rare_cos": float(retrieval.get("rare_cos", 0.0)),
        "res_coverage": float(hit.sum() / nr),
        "res_ordered": signals._lis(matched) / nr if matched else 0.0,
        "res_run": _run_length(ridx, arg[ridx], hit[ridx]) / nr,
        "res_mean": float(np.clip(best[resid].mean(), 0, 1)),
        "res_word_overlap": float(np.mean(wo)) if wo else 0.0,
        "res_fact_overlap": len(res_facts & cf["facts"]) / len(res_facts) if res_facts else 0.0,
        "res_quote": qhit / len(res_quotes) if res_quotes else float(retrieval.get("quote_frac", 0.0)),
        "res_containment": dedup.containment(res_shingles, cf["shingles"]),
        "back_coverage": float(np.mean(sims[resid].max(axis=0) >= signals.ALIGN_THRESHOLD)),
    }


def trace(ctx, sus, model=None, threshold=None, stop_share=None, cover=None, exclude=None, residual=True,
          use_facts=True, use_quotes=True, k_verify=K_VERIFY, max_sources=MAX_SOURCES, require_dates=True):
    """Explain-away rounds for one suspect. Returns the rounds and, via finalize(), sources with shares.

    threshold=0 and stop_share=0 run every round up to max_sources; finalize() with real values then gives
    exactly what an early-stopping run would have returned, which is how the evaluation tunes them on dev.
    """
    model = model or getattr(ctx, "ms_model", None) or load_model()
    threshold = model["threshold"] if threshold is None else threshold
    stop_share = model["stop_share"] if stop_share is None else stop_share
    cover = model["cover"] if cover is None else cover
    idx = ctx.idx
    sdoc = sus.as_doc()
    if hasattr(ctx.feats, "add"):
        ctx.feats.add(sus.doc_id, sus.text)
    sents, _ = ctx.emb.doc(sus.doc_id, sus.text)
    ns = len(sents)
    excl = np.zeros(idx.N, dtype=bool) if exclude is None else exclude.copy()
    if sus.n is not None:
        excl[sus.n] = True
    resid = np.ones(ns, dtype=bool)
    pairs = {}
    rounds = []
    stop = f"{max_sources} sources found"
    for r in range(max_sources):
        share_left = float(resid.mean())
        if r > 0 and (share_left < stop_share or resid.sum() < MIN_NEW):
            stop = f"unexplained share {share_left:.2f} below {stop_share:.2f}"
            break
        t0 = time.perf_counter()
        res_text = " ".join(s for s, keep in zip(sents, resid) if keep) if residual else sus.text
        cands, stats = candidates.retrieve_text(idx, res_text, sus.published, sus.site, exclude=excl,
                                                use_quotes=use_quotes, use_facts=use_facts,
                                                require_dates=require_dates)
        t_ret = time.perf_counter() - t0
        if not cands:
            stop = "no candidates retrieved"
            break
        top = cands[:k_verify]
        ctx.emb.prefetch([(ctx.doc(c["doc"])["doc_id"], ctx.doc(c["doc"])["text"]) for c in top])
        vtext = " ".join(s for s, keep in zip(sents, resid) if keep)
        names, nums = tx.facts(vtext)
        rfacts, rquotes, rsh = names | nums, tx.quotes(vtext), dedup.shingles(vtext)
        verified = []
        for c in top:
            cd = ctx.doc(c["doc"])
            pair = pairs.get(c["doc"])
            if pair is None:
                pair = pairs[c["doc"]] = signals.Pair(sdoc, cd, ctx.feats, ctx.emb, c)
            sig = round_signals(pair, resid, c, rfacts, rquotes, rsh)
            verified.append({"doc": c["doc"], "doc_id": cd["doc_id"], "url": cd.get("url"), "site": cd["site"],
                             "published": cd.get("published"), "retrieval": c["score"], "signals": sig,
                             "prob": score(sig, model)})
        verified.sort(key=lambda v: -v["prob"])
        best = verified[0]
        # Copies of one original verify almost equally well; the earliest of them is the likelier origin.
        close = [v for v in verified if v["prob"] >= best["prob"] - TIE_MARGIN]
        chosen = min(close, key=lambda v: v["published"] if v["published"] is not None
                     and not (isinstance(v["published"], float) and math.isnan(v["published"])) else float("inf"))
        bsim, barg, _ = pairs[chosen["doc"]].alignment()
        bsim = bsim.astype(np.float32)
        new = resid & (bsim >= cover)
        rounds.append({"round": r + 1, "share_left": share_left, "residual_sents": int(resid.sum()),
                       "query_terms": stats["query_terms"], "postings_touched": stats["postings_touched"],
                       "n_quotes": stats["n_quotes"], "candidates": cands, "verified": verified,
                       "chosen": chosen, "prob": best["prob"], "new_covered": np.nonzero(new)[0].tolist(),
                       "sims": bsim, "arg": barg, "t_retrieve": t_ret, "t_round": time.perf_counter() - t0})
        if best["prob"] < threshold:
            stop = f"best candidate p={best['prob']:.3f} below threshold {threshold:.2f}"
            break
        if new.sum() < MIN_NEW:
            stop = f"best candidate covers {int(new.sum())} new sentences (< {MIN_NEW})"
            break
        resid &= ~new
        excl[chosen["doc"]] = True
    return {"suspect": sus.doc_id, "n_sents": ns, "sents": sents, "rounds": rounds, "stop": stop,
            "params": {"threshold": threshold, "stop_share": stop_share, "cover": cover}}


def accepted_rounds(tr, threshold, stop_share):
    """The prefix of rounds an early-stopping run with these parameters keeps."""
    keep = []
    for rd in tr["rounds"]:
        if rd["round"] > 1 and (rd["share_left"] < stop_share or rd["residual_sents"] < MIN_NEW):
            break
        if rd["prob"] < threshold or len(rd["new_covered"]) < MIN_NEW:
            break
        keep.append(rd)
    return keep


def finalize(tr, threshold=None, stop_share=None, cover=None):
    """Sources with sentence shares: each suspect sentence goes to the found source it aligns to best."""
    p = tr["params"]
    threshold = p["threshold"] if threshold is None else threshold
    stop_share = p["stop_share"] if stop_share is None else stop_share
    cover = p["cover"] if cover is None else cover
    keep = accepted_rounds(tr, threshold, stop_share)
    ns = tr["n_sents"]
    owner = np.full(ns, -1)
    if keep:
        S = np.stack([rd["sims"] for rd in keep])
        best = S.argmax(axis=0)
        ok = S.max(axis=0) >= cover
        owner[ok] = best[ok]
    sources = []
    for j, rd in enumerate(keep):
        c = rd["chosen"]
        sources.append({"doc": c["doc"], "doc_id": c["doc_id"], "url": c["url"], "site": c["site"],
                        "round": rd["round"], "prob": c["prob"], "share": float(np.mean(owner == j)) if ns else 0.0,
                        "sentences": np.nonzero(owner == j)[0].tolist()})
    return {"sources": sources, "unexplained": float(np.mean(owner < 0)) if ns else 1.0, "n_sents": ns}


def trace_text(ctx, text, published=None, site=None, doc_id=None, **kw):
    """Store-free path: trace a text that is not in the index (a pasted article, a fresh page)."""
    import hashlib
    doc_id = doc_id or "text:" + hashlib.sha1(text.encode()).hexdigest()[:16]
    tr = trace(ctx, Suspect(doc_id, text, published, site), **kw)
    return tr, finalize(tr)
