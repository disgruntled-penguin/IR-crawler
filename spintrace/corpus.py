"""Corpus assembly and per-document features shared by later stages."""
import json
import pickle
import time

from . import config, dedup, store
from .eval import rewrites, wikinews

FEATURES = config.INDEX_DIR / "features.pkl"


def ingest_eval(con=None):
    """Mix Wikinews originals and their synthetic rewrites into the crawled store."""
    con = con or store.connect()
    n = 0
    for line in wikinews.OUT.open():
        o = json.loads(line)
        store.save_doc(con, {
            "doc_id": store.doc_id_for(o["url"]), "url": o["url"], "host": "en.wikinews.org", "site": "wikinews.org",
            "kind": "original", "origin": "wikinews", "title": o["title"], "text": o["text"],
            "published": o["published"], "date_source": "wikinews", "fetched_at": None,
        })
        n += 1
    m = 0
    if rewrites.OUT.exists():
        for line in rewrites.OUT.open():
            r = json.loads(line)
            url = f"https://{r['farm']}/article/{r['rid']}"
            store.save_doc(con, {
                "doc_id": store.doc_id_for(url), "url": url, "host": r["farm"], "site": r["farm"],
                "kind": "suspect", "origin": "synthetic", "title": None, "text": r["text"],
                "published": r["published"], "date_source": "synthetic", "fetched_at": None,
                "meta": {"level": r["level"], "orig_doc": store.doc_id_for(r["orig_url"]), "rid": r["rid"]},
            })
            m += 1
    con.commit()
    return n, m


def build_features(con=None):
    """Exact hash, shingle set and MinHash signature for every document, keyed by doc_id."""
    global _feats
    con = con or store.connect()
    old = load_features() if FEATURES.exists() else {}
    feats = {}
    t0 = time.time()
    for row in con.execute("SELECT doc_id, text, content_hash FROM docs"):
        prev = old.get(row["doc_id"])
        if prev and prev["content_hash"] == row["content_hash"]:
            feats[row["doc_id"]] = prev
            continue
        sh = dedup.shingles(row["text"])
        feats[row["doc_id"]] = {
            "content_hash": row["content_hash"], "exact": dedup.exact_hash(row["text"]),
            "shingles": sh, "minhash": dedup.minhash(sh),
        }
    with FEATURES.open("wb") as f:
        pickle.dump(feats, f, protocol=pickle.HIGHEST_PROTOCOL)
    _feats = feats
    return feats, time.time() - t0


_feats = None


def load_features():
    global _feats
    if _feats is None:
        with FEATURES.open("rb") as f:
            _feats = pickle.load(f)
    return _feats
