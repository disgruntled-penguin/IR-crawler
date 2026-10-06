"""Evaluation runner: score every evaluation item with SpinTrace and each baseline, once, and cache it.

Items are suspects with a known answer:
  positives      synthetic rewrites, whose source is a Wikinews original in the same index
  original       the Wikinews originals themselves (nothing earlier to copy from)
  hard_negative  crawled articles with same-event coverage from another outlet published earlier
Rewrites of the same original are hidden from each other, so a rewrite cannot be traced to a sibling.
Splits are by original (or by suspect for negatives), so no story appears in both dev and test.
"""
import hashlib
import json
import pickle
import time
from collections import defaultdict

import numpy as np

from .. import config, dedup, index, store
from ..detect import candidates, pipeline, signals
from . import hardneg

RAW = config.DATA / "eval" / "raw.pkl"
TOP = 20


def split_of(key):
    return "dev" if int(hashlib.sha1(key.encode()).hexdigest(), 16) % 2 == 0 else "test"


def build_items(ctx):
    idx = ctx.idx
    items, groups = [], defaultdict(list)
    for row in store.iter_docs(ctx.con, "origin='synthetic'"):
        m = json.loads(row["meta"])
        if row["doc_id"] not in idx.doc_num or m["orig_doc"] not in idx.doc_num:
            continue
        n = idx.doc_num[row["doc_id"]]
        if m["level"] == "facts_only":
            # Same story, written only from the original's facts: must not be flagged as derived.
            items.append({"doc": n, "label": 0, "level": "facts_only", "source": None,
                          "same_event": idx.doc_num[m["orig_doc"]], "split": split_of(m["orig_doc"]),
                          "group": m["orig_doc"]})
            continue
        groups[m["orig_doc"]].append(n)
        items.append({"doc": n, "label": 1, "level": m["level"], "source": idx.doc_num[m["orig_doc"]],
                      "split": split_of(m["orig_doc"]), "group": m["orig_doc"]})
    for row in store.iter_docs(ctx.con, "origin='wikinews'"):
        if row["doc_id"] in idx.doc_num:
            items.append({"doc": idx.doc_num[row["doc_id"]], "label": 0, "level": "original", "source": None,
                          "split": split_of(row["doc_id"]), "group": row["doc_id"]})
    for a, b in hardneg.load():
        if a in idx.doc_num and b in idx.doc_num:
            items.append({"doc": idx.doc_num[a], "label": 0, "level": "hard_negative", "source": None,
                          "same_event": idx.doc_num[b], "split": split_of(a), "group": a})
    for it in items:
        it["exclude"] = [n for n in groups.get(it["group"], []) if n != it["doc"]]
    return items


def allowed_mask(idx, it):
    mask = candidates.earlier_mask(idx, it["doc"])
    if it["exclude"]:
        mask[it["exclude"]] = False
    return mask


def ranked(acc, mask, k=TOP):
    acc = np.where(mask, acc, -np.inf)
    top = np.argpartition(-acc, min(k, len(acc) - 1))[:k]
    top = top[np.argsort(-acc[top])]
    return [(int(d), float(acc[d])) for d in top if np.isfinite(acc[d]) and acc[d] > 0]


class Baselines:
    """Whole-document comparisons against every earlier document (the all-pairs approach)."""

    def __init__(self, ctx):
        self.ctx = ctx
        idx = ctx.idx
        feats = ctx.feats
        self.sigs = [feats[d]["minhash"] for d in idx.doc_ids]
        self.shingles = [feats[d]["shingles"] for d in idx.doc_ids]
        self.exact = defaultdict(list)
        for n, d in enumerate(idx.doc_ids):
            self.exact[feats[d]["exact"]].append(n)
        self.lsh = dedup.LSH()
        for n, s in enumerate(self.sigs):
            self.lsh.add(n, s)
        ctx.emb.prefetch([(d["doc_id"], d["text"]) for d in ctx.docs])
        self.dense = np.stack([ctx.emb.doc_vector(d["doc_id"], d["text"]) for d in ctx.docs])

    def run(self, it, mask):
        idx, n, out = self.ctx.idx, it["doc"], {}
        t = time.perf_counter()
        h = self.ctx.feats[idx.doc_ids[n]]["exact"]
        hits = [m for m in self.exact[h] if mask[m]]
        out["exact_hash"] = ([(m, 1.0) for m in hits], time.perf_counter() - t)
        t = time.perf_counter()
        cands = [m for m in self.lsh.query(self.sigs[n]) if mask[m]]
        r = sorted(((m, dedup.est_jaccard(self.sigs[n], self.sigs[m])) for m in cands), key=lambda x: -x[1])
        out["minhash_jaccard"] = (r[:TOP], time.perf_counter() - t)
        # The best case for any shingle method: exact containment against every earlier document, no LSH.
        t = time.perf_counter()
        sh = self.shingles[n]
        r = [(m, dedup.containment(sh, self.shingles[m])) for m in np.nonzero(mask)[0].tolist()] if sh else []
        r = sorted((x for x in r if x[1] > 0), key=lambda x: -x[1])
        out["shingle_containment"] = (r[:TOP], time.perf_counter() - t)
        t = time.perf_counter()
        terms = idx.doc_terms(n)
        out["tfidf_cosine"] = (ranked(idx.cosine_scores(idx.query_vector(terms)), mask), time.perf_counter() - t)
        t = time.perf_counter()
        acc = idx.bm25_scores(terms)
        out["bm25"] = (ranked(acc / (acc[n] or 1.0), mask), time.perf_counter() - t)
        t = time.perf_counter()
        out["dense_cosine"] = (ranked(self.dense @ self.dense[n], mask), time.perf_counter() - t)
        return out


def verify_candidates(ctx, it, cands, k=pipeline.VERIFY_TOP):
    s = ctx.doc(it["doc"])
    ctx.emb.prefetch([(ctx.doc(c["doc"])["doc_id"], ctx.doc(c["doc"])["text"]) for c in cands[:k]])
    rows = []
    for c in cands[:k]:
        cd = ctx.doc(c["doc"])
        pair = signals.Pair(s, cd, ctx.feats, ctx.emb, c)
        verdict, _ = pipeline.provenance(s, cd, pair)
        rows.append({"doc": c["doc"], "signals": signals.compute(pair), "verdict": verdict,
                     "published": pipeline.date_of(cd)})
    return rows


def run(stem_compare=True, log_every=250, sample=None, raw=RAW):
    ctx = pipeline.Context()
    idx = ctx.idx
    items = build_items(ctx)
    if sample:
        import random
        items = random.Random(0).sample(items, sample)
    base = Baselines(ctx)
    sidx = index.load(True) if stem_compare and index.index_path(True).exists() else None
    t_all = time.time()
    for i, it in enumerate(items):
        n = it["doc"]
        text = ctx.doc(n)["text"]
        mask = allowed_mask(idx, it)
        it["allowed"] = int(mask.sum())
        it["baselines"] = base.run(it, mask)
        it["touched_full"] = int(sum(idx.df[t] for t in idx.doc_terms(n)))
        excl = np.zeros(idx.N, dtype=bool)
        excl[it["exclude"]] = True
        t = time.perf_counter()
        cands, stats = candidates.retrieve(idx, n, text, k=TOP, exclude=excl)
        t_ret = time.perf_counter() - t
        t = time.perf_counter()
        it["verified"] = verify_candidates(ctx, it, cands)
        it["spintrace"] = {"cands": [(c["doc"], c["score"]) for c in cands], "t_retrieve": t_ret,
                           "t_verify": time.perf_counter() - t, "touched": stats["postings_touched"]}
        nq, _ = candidates.retrieve(idx, n, text, k=TOP, exclude=excl, use_quotes=False)
        it["no_quotes_cands"] = [c["doc"] for c in nq]
        if it["label"] == 1:
            allowed = mask
            it["budget"] = {}
            for mode, m in [("rare", 3), ("rare", 5), ("rare", 10), ("rare", 30), ("common", 10), ("random", 10)]:
                q = candidates.budget_query(idx, n, m, mode)
                top = idx.cosine(idx.query_vector(q), allowed, k=TOP)
                it["budget"][f"{mode}{m}"] = ([d for d, _ in top], int(sum(idx.df[t] for t in q)))
        nf, _ = candidates.retrieve(idx, n, text, k=TOP, exclude=excl, use_facts=False)
        it["no_facts_cands"] = [c["doc"] for c in nf]
        dense_cands = [{"doc": d, "rare_cos": 0.0, "quote_frac": 0.0, "score": s} for d, s in it["baselines"]["dense_cosine"][0]]
        it["dense_verified"] = verify_candidates(ctx, it, dense_cands)
        if sidx is not None:
            it["stem_cands"] = stem_candidates(idx, sidx, it, text)
        # Provenance without dates: which way does the coverage asymmetry point for the known pair?
        other = it["source"] if it["label"] == 1 else it.get("same_event")
        if other is not None:
            pair = signals.Pair(ctx.doc(n), ctx.doc(other), ctx.feats, ctx.emb)
            it["direction"] = pipeline.direction(pair)
        if (i + 1) % log_every == 0:
            ctx.emb.save()
            print(f"eval {i + 1}/{len(items)} {time.time() - t_all:.0f}s", flush=True)
    ctx.emb.save()
    raw.parent.mkdir(parents=True, exist_ok=True)
    meta = {"N": idx.N, "doc_ids": idx.doc_ids, "sites": idx.meta["site"].tolist(), "origins": idx.meta["origin"].tolist(),
            "built": time.time()}
    with raw.open("wb") as f:
        pickle.dump({"items": items, "meta": meta}, f)
    return items


def stem_candidates(idx, sidx, it, text):
    """Candidate list from the stemmed index, mapped back to the unstemmed index's doc numbers."""
    sn = sidx.doc_num.get(idx.doc_ids[it["doc"]])
    if sn is None:
        return None
    excl = np.zeros(sidx.N, dtype=bool)
    for m in it["exclude"]:
        j = sidx.doc_num.get(idx.doc_ids[m])
        if j is not None:
            excl[j] = True
    cands, _ = candidates.retrieve(sidx, sn, text, k=TOP, exclude=excl)
    return [idx.doc_num.get(sidx.doc_ids[c["doc"]], -1) for c in cands]
