"""Do originals outrank their copies in search? Ranked retrieval with and without originality g(d).

Synthetic queries: the title of each Wikinews original that has rewrites in the index. Graded relevance:
the original 2, its rewrites 1 (same content, but derivative), everything else 0. lambda is chosen on dev
queries and reported on test queries. Live queries are written out for the team to judge.
"""
import csv
import json
import math

import numpy as np

from .. import config, index, store
from ..detect import scan
from ..search import search
from .run import split_of

LIVE_QUERIES = ["Francis Halzen Nobel physics neutrinos", "Kenya first Ebola case", "Paramount Warner Bros merger",
                "Smriti Mandhana India captain", "flydubai co-pilot attack", "Mistral open model", "Jim Bakker dies",
                "DRDO high altitude parachute", "Bluesky domain", "Amazon Prime Day deals", "France school protests",
                "election commission opposition march", "Iran Strait of Hormuz", "Supreme Court climate case oil",
                "OpenAI mathematicians"]
JUDGE = config.ROOT / "evaldata" / "search_judgments.csv"
LAMBDAS = [0.0, 0.1, 0.2, 0.3, 0.5, 0.8]


def ndcg(gains, ideal, k=10):
    dcg = sum(g / math.log2(i + 2) for i, g in enumerate(gains[:k]))
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(sorted(ideal, reverse=True)[:k]))
    return dcg / idcg if idcg else 0.0


def synthetic_queries(con, idx):
    copies = {}
    for r in store.iter_docs(con, "origin='synthetic'"):
        m = json.loads(r["meta"])
        copies.setdefault(m["orig_doc"], []).append(r["doc_id"])
    out = []
    for r in store.iter_docs(con, "origin='wikinews'"):
        if r["doc_id"] in idx.doc_num and len(copies.get(r["doc_id"], [])) >= 2:
            rel = {r["doc_id"]: 2, **{c: 1 for c in copies[r["doc_id"]]}}
            out.append({"query": r["title"], "rel": rel, "original": r["doc_id"], "split": split_of(r["doc_id"])})
    return out


# Synthetic rewrites carry no headline, so the title zone would favour originals by itself; body only here.
BODY_ONLY = {"body": 1.0}


def run_queries(idx, queries, g, lam):
    rows = []
    for q in queries:
        res = search(idx, q["query"], g, lam=lam, k=10, zones=BODY_ONLY)
        ids = [idx.doc_ids[d] for d, *_ in res]
        gains = [q["rel"].get(d, 0) for d in ids]
        rows.append({"p@10": sum(x > 0 for x in gains) / 10, "ndcg@10": ndcg(gains, list(q["rel"].values())),
                     "original_first": float(bool(ids) and ids[0] == q["original"]),
                     "original_share_top10": (sum(x == 2 for x in gains) / max(1, sum(x > 0 for x in gains)))})
    return {k: float(np.mean([r[k] for r in rows])) for k in rows[0]} if rows else {}


def main():
    con = store.connect()
    idx = index.load()
    g = scan.originality(con)
    qs = synthetic_queries(con, idx)
    dev = [q for q in qs if q["split"] == "dev"]
    test = [q for q in qs if q["split"] == "test"]
    best = max(LAMBDAS, key=lambda lam: run_queries(idx, dev, g, lam)["ndcg@10"])
    rows = [{"setting": "relevance only (no g)", "lambda": 0.0, **run_queries(idx, test, None, 0.0)},
            {"setting": "net score with g(d)", "lambda": best, **run_queries(idx, test, g, best)}]
    rows[0]["n_queries"] = rows[1]["n_queries"] = len(test)
    config.RESULTS.mkdir(exist_ok=True)
    with (config.RESULTS / "search_eval.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows([{k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()} for r in rows])
    write_live(con, idx, g, best)
    return rows


def write_live(con, idx, g, lam):
    """Top 10 for each live query, with and without g, as a sheet for the two judges."""
    old = {}
    if JUDGE.exists():
        old = {(r["query"], r["doc_id"]): r for r in csv.DictReader(JUDGE.open())}
    with JUDGE.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["query", "doc_id", "url", "rank_no_g", "rank_with_g", "g", "judge1", "judge2"])
        for q in LIVE_QUERIES:
            a = [idx.doc_ids[d] for d, *_ in search(idx, q, None, lam=0.0, k=10)]
            b = [idx.doc_ids[d] for d, *_ in search(idx, q, g, lam=lam, k=10)]
            for d in dict.fromkeys(a + b):
                url = con.execute("SELECT url FROM docs WHERE doc_id=?", (d,)).fetchone()[0]
                prev = old.get((q, d), {})
                w.writerow([q, d, url, a.index(d) + 1 if d in a else "", b.index(d) + 1 if d in b else "",
                            f"{g.get(d, 1.0):.2f}", prev.get("judge1", ""), prev.get("judge2", "")])
