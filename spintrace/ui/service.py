"""Data layer of the web app: the same index, search and detection code as the CLI, returned as plain dicts."""
import csv
import math
import threading

from .. import config, index, store
from ..detect import pipeline, signals, scan
from ..search import search as ranked_search

LOCK = threading.Lock()
_state = {}


def _clean(x):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None
    return x


def _state_get():
    """Index and g(d), loaded once. SQLite connections are per request, since each request has its own thread."""
    if not _state:
        con = store.connect()
        _state.update(idx=index.load(), g=scan.originality(con))
        con.close()
    return _state


def _source_of(con, doc_id):
    r = con.execute("""SELECT e.source, e.prob, e.verdict, d.site, d.title FROM edges e
                       JOIN docs d ON d.doc_id=e.source WHERE e.suspect=? ORDER BY e.prob DESC LIMIT 1""",
                    (doc_id,)).fetchone()
    if not r:
        return None
    return {"doc_id": r["source"], "prob": round(r["prob"], 3), "verdict": r["verdict"], "site": r["site"],
            "title": r["title"]}


def search(query, use_g=True, k=10, crawl_only=True, lam=None):
    """Ranked results with relevance, originality g(d) and, for copies, the original they trace to."""
    query = query.strip()
    if not query:
        return {"results": [], "n_docs": None}
    with LOCK:
        st = _state_get()
        idx, con = st["idx"], store.connect()
        kw = {} if lam is None else {"lam": lam}
        hits = ranked_search(idx, query, st["g"] if use_g else None, k=k,
                             origins={"crawl"} if crawl_only else None, **kw)
        out = []
        for rank, (d, net, rel, gd) in enumerate(hits, 1):
            doc_id = idx.doc_ids[d]
            row = con.execute("SELECT title, url, site, published, n_words FROM docs WHERE doc_id=?",
                              (doc_id,)).fetchone()
            out.append({"rank": rank, "doc_id": doc_id, "title": row["title"] or row["url"], "url": row["url"],
                        "site": row["site"], "published": _clean(row["published"]), "words": row["n_words"],
                        "net": round(net, 3), "rel": round(rel, 3), "g": round(gd, 3),
                        "copy_of": _source_of(con, doc_id)})
        con.close()
        return {"results": out, "n_docs": idx.N}


def integrity():
    """Per-site share of crawled articles flagged as derived (the table written by `spintrace scan`)."""
    path = config.RESULTS / "domain_integrity.csv"
    if not path.exists():
        return []
    rows = list(csv.DictReader(path.open()))
    for r in rows:
        r["articles"] = int(r["articles"])
        r["flagged_derived"] = int(r["flagged_derived"])
        r["derived_share"] = float(r["derived_share"])
    return rows


def alignment_rows(best, arg, ss, cs, threshold, limit=40):
    """Per suspect sentence: its best-matching sentence in the source, the cosine, and whether it counts as aligned."""
    return [{"i": i, "sim": round(float(best[i]), 3), "aligned": bool(best[i] >= threshold),
             "suspect": ss[i], "source": cs[int(arg[i])], "source_i": int(arg[i])}
            for i in range(min(limit, len(ss)))]


def _context():
    """Detection context (index, features, embedder, verifier), built on first use."""
    if "ctx" not in _state:
        idx = _state_get()["idx"]
        con = store.connect()
        _state["ctx"] = pipeline.Context(con=con, idx=idx)
        con.close()
    return _state["ctx"]


def _brief(d, **extra):
    return {"doc_id": d["doc_id"], "site": d["site"], "title": d["title"] or d["url"], "url": d["url"],
            "published": _clean(d["published"]), **extra}


def trace(key):
    """Trace one document the way `spintrace suspect` does: query, candidates, verification, alignment, verdict."""
    with LOCK:
        ctx = _context()
        con = store.connect()
        row = con.execute("SELECT doc_id FROM docs WHERE doc_id=? OR url=?", (key, key)).fetchone()
        con.close()
        if not row or row[0] not in ctx.idx.doc_num:
            return {"error": "that document is not in the index (it may have been crawled after the last build)"}
        n = ctx.num(row[0])
        s = ctx.doc(n)
        rep = pipeline.analyse(ctx, n)
        st = rep["stats"]
        cands = []
        for c in rep["candidates"][:10]:
            d = ctx.doc(c["doc"])
            cands.append(_brief(d, score=round(c["score"], 3), rare_cos=round(c["rare_cos"], 3),
                                quote_frac=round(c["quote_frac"], 2)))
        verified = [_brief(ctx.doc(v["doc"]), prob=round(v["prob"], 3), verdict=v["verdict"], why=v["why"],
                           signals={k: round(x, 3) for k, x in v["signals"].items()}) for v in rep["verified"]]
        focus = rep["source"] or (rep["verified"][0] if rep["verified"] else None)
        align = None
        if focus:
            c = ctx.doc(focus["doc"])
            best, arg, _ = signals.Pair(s, c, ctx.feats, ctx.emb).alignment()
            ss, _ = ctx.emb.doc(s["doc_id"], s["text"])
            cs, _ = ctx.emb.doc(c["doc_id"], c["text"])
            align = {"source": _brief(c), "threshold": signals.ALIGN_THRESHOLD, "n_suspect": len(ss),
                     "rows": alignment_rows(best, arg, ss, cs, signals.ALIGN_THRESHOLD)}
        return {"suspect": _brief(s, words=s["n_words"]),
                "query": {"terms": [[t, w] for t, w in st["query_terms"]], "quotes": st["n_quotes"],
                          "allowed": st["allowed"], "n_docs": ctx.idx.N, "postings": st["postings_touched"]},
                "candidates": cands, "verified": verified, "alignment": align,
                "verdict": {"flagged": rep["flagged"], "threshold": round(ctx.model["threshold"], 3),
                            "weights": ctx.model["source"],
                            "source": _brief(ctx.doc(rep["source"]["doc"]), prob=round(rep["source"]["prob"], 3),
                                             verdict=rep["source"]["verdict"], why=rep["source"]["why"])
                            if rep["flagged"] else None}}
