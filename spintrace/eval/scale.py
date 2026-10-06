"""Scale experiment: the same test items against an index that also holds one CC-NEWS file.

Nothing is refit: the verifier and threshold are the ones fit on dev for the main index. The question is
whether rare-term retrieval still finds the source, what it costs, and whether detection holds, when the
index grows with thousands of real articles from the same days as the crawl.
"""
import csv
import time

import numpy as np

from .. import config, corpus, store
from ..detect import candidates, pipeline, signals, verify
from ..index import Index
from . import report

OUT = config.RESULTS / "scale.csv"


def build_index(con):
    feats = corpus.load_features()
    t0 = time.time()
    idx = Index().build(d for d in store.iter_docs(con) if d["doc_id"] in feats)
    return idx, time.time() - t0


def run():
    con = store.connect()
    data = report.load()
    main_ids = data["meta"]["doc_ids"]
    test = [it for it in data["items"] if it["split"] == "test"]
    model = verify.load()
    idx, secs = build_index(con)
    n_cc = int((idx.meta["origin"] == "ccnews").sum())
    print(f"scale index: {idx.N} docs ({n_cc} CC-NEWS), {len(idx.terms)} terms, built in {secs:.0f}s", flush=True)
    ctx = pipeline.Context(con=con, idx=idx)
    rows = []
    for i, it in enumerate(test):
        doc_id = main_ids[it["doc"]]
        n = idx.doc_num[doc_id]
        excl = np.zeros(idx.N, dtype=bool)
        for m in it["exclude"]:
            j = idx.doc_num.get(main_ids[m])
            if j is not None:
                excl[j] = True
        s = ctx.doc(n)
        t = time.perf_counter()
        cands, stats = candidates.retrieve(idx, n, s["text"], exclude=excl)
        t_ret = time.perf_counter() - t
        mask = candidates.earlier_mask(idx, n) & ~excl
        t = time.perf_counter()
        full = idx.cosine(idx.query_vector(idx.doc_terms(n)), mask, k=1)
        t_full = time.perf_counter() - t
        top = cands[:pipeline.VERIFY_TOP]
        ctx.emb.prefetch([(s["doc_id"], s["text"])] + [(ctx.doc(c["doc"])["doc_id"], ctx.doc(c["doc"])["text"]) for c in top])
        vs = []
        for c in top:
            cd = ctx.doc(c["doc"])
            pair = signals.Pair(s, cd, ctx.feats, ctx.emb, c)
            verdict, _ = pipeline.provenance(s, cd, pair)
            vs.append({"doc": c["doc"], "signals": signals.compute(pair), "verdict": verdict,
                       "published": pipeline.date_of(cd)})
        score, _ = report.spin_score({"verified": vs}, model)
        src = idx.doc_num.get(main_ids[it["source"]]) if it["label"] == 1 else None
        rows.append({"level": it["level"], "label": it["label"], "flag": score >= model["threshold"],
                     "r1": bool(cands) and cands[0]["doc"] == src, "r20": src in [c["doc"] for c in cands],
                     "full_r1": bool(full) and full[0][0] == src, "touched": stats["postings_touched"],
                     "touched_full": int(sum(idx.df[t] for t in idx.doc_terms(n))),
                     "ms": 1000 * t_ret, "ms_full": 1000 * t_full, "allowed": stats["allowed"]})
        if (i + 1) % 250 == 0:
            ctx.emb.save()
            print(f"scale {i + 1}/{len(test)}", flush=True)
    ctx.emb.save()
    out = summarise(rows, idx.N, n_cc)
    with OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    return out


def summarise(rows, n_docs, n_cc):
    negs = [r for r in rows if r["label"] == 0]
    out = []
    for level in report.LEVELS:
        pos = [r for r in rows if r["level"] == level]
        if not pos:
            continue
        p, r, f = report.prf([x["flag"] for x in pos + negs], [x["label"] for x in pos + negs])
        out.append({"level": level, "index_docs": n_docs, "ccnews_docs": n_cc, "n": len(pos),
                    "recall@1": float(np.mean([x["r1"] for x in pos])), "recall@20": float(np.mean([x["r20"] for x in pos])),
                    "full_doc_tfidf_recall@1": float(np.mean([x["full_r1"] for x in pos])), "f1": f,
                    "postings_touched": float(np.mean([x["touched"] for x in pos])),
                    "postings_full_doc": float(np.mean([x["touched_full"] for x in pos])),
                    "ms_retrieve": float(np.mean([x["ms"] for x in pos])), "ms_full_doc": float(np.mean([x["ms_full"] for x in pos])),
                    "earlier_docs": float(np.mean([x["allowed"] for x in pos]))})
    for level in report.NEG_LEVELS:
        ns = [r for r in negs if r["level"] == level]
        if ns:
            out.append({"level": f"fpr_{level}", "index_docs": n_docs, "ccnews_docs": n_cc, "n": len(ns),
                        "f1": float(np.mean([x["flag"] for x in ns]))})
    return out
