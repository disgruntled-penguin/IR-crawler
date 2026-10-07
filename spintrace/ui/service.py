"""Data layer of the web app: the same index, search and detection code as the CLI, returned as plain dicts."""
import csv
import math
import threading

from .. import config, index, store
from ..detect import scan
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
