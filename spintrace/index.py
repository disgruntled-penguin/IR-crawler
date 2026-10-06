"""Positional inverted index with zones, plus tf-idf (lnc.ltc) and BM25 scoring.

Postings per (zone, term) are three arrays: sorted doc numbers, term frequencies, and the
concatenated positions (sliced with the cumulative tf). A forward index of body term counts
supports building a query from a whole document.
"""
import heapq
import math
import pickle
import time
from collections import Counter, defaultdict

import numpy as np

from . import config, store, text as tx

ZONES = ("title", "body", "quote")
BM25_K1, BM25_B = 1.2, 0.75


class Postings:
    __slots__ = ("docs", "tfs", "pos", "offs")

    def __init__(self, entries):
        self.docs = np.fromiter((d for d, _ in entries), dtype=np.int32, count=len(entries))
        self.tfs = np.fromiter((len(p) for _, p in entries), dtype=np.int32, count=len(entries))
        self.offs = np.concatenate([[0], np.cumsum(self.tfs)]).astype(np.int64)
        self.pos = np.fromiter((x for _, p in entries for x in p), dtype=np.int32, count=int(self.offs[-1]))

    def positions(self, i):
        return self.pos[self.offs[i]:self.offs[i + 1]]


class Index:
    def __init__(self, stemming=False):
        self.stemming = stemming
        self.vocab = {}
        self.terms = []
        self.doc_ids = []
        self.meta = {}
        self.post = {z: {} for z in ZONES}
        self.fwd = []
        self.lengths = {}
        self.norms = None
        self.df = None

    @property
    def N(self):
        return len(self.doc_ids)

    def tid(self, term, add=False):
        t = self.vocab.get(term)
        if t is None and add:
            t = self.vocab[term] = len(self.terms)
            self.terms.append(term)
        return t

    def tokens(self, text):
        return tx.tokens(text, stemming=self.stemming)

    def build(self, docs):
        """docs: iterable of dicts with doc_id, title, text, published, site, kind, origin."""
        t0 = time.time()
        raw = {z: defaultdict(list) for z in ZONES}
        lengths = {z: [] for z in ZONES}
        published, sites, kinds, origins = [], [], [], []
        for d in docs:
            n = len(self.doc_ids)
            self.doc_ids.append(d["doc_id"])
            published.append(d.get("published") or np.nan)
            sites.append(d["site"])
            kinds.append(d["kind"])
            origins.append(d["origin"])
            zones = {"title": d.get("title") or "", "body": d["text"], "quote": " . ".join(tx.quotes(d["text"]))}
            for z, content in zones.items():
                toks = self.tokens(content)
                lengths[z].append(len(toks))
                positions = defaultdict(list)
                for i, t in enumerate(toks):
                    positions[self.tid(t, add=True)].append(i)
                for t, p in positions.items():
                    raw[z][t].append((n, p))
                if z == "body":
                    tids = np.fromiter(positions.keys(), dtype=np.int32, count=len(positions))
                    tfs = np.fromiter((len(p) for p in positions.values()), dtype=np.int32, count=len(positions))
                    order = np.argsort(tids)
                    self.fwd.append((tids[order], tfs[order]))
        for z in ZONES:
            self.post[z] = {t: Postings(e) for t, e in raw[z].items()}
            self.lengths[z] = np.array(lengths[z], dtype=np.int32)
        self.meta = {
            "published": np.array(published, dtype=np.float64), "site": np.array(sites),
            "kind": np.array(kinds), "origin": np.array(origins),
        }
        self.doc_num = {d: i for i, d in enumerate(self.doc_ids)}
        self.df = np.zeros(len(self.terms), dtype=np.int32)
        for t, p in self.post["body"].items():
            self.df[t] = len(p.docs)
        self.idf = np.log10(self.N / np.maximum(self.df, 1))
        # lnc document vectors: log tf, no idf, cosine length normalisation.
        self.norms = np.array([math.sqrt(float(np.sum((1 + np.log10(tfs)) ** 2))) or 1.0 for _, tfs in self.fwd])
        self.avg_len = float(self.lengths["body"].mean()) if self.N else 0.0
        self.build_seconds = time.time() - t0
        return self

    def postings(self, term, zone="body"):
        t = self.vocab.get(term)
        return self.post[zone].get(t) if t is not None else None

    def phrase(self, words, zone="body"):
        """Docs containing the exact token sequence: intersect postings, then check consecutive positions."""
        toks = self.tokens(" ".join(words)) if isinstance(words, (list, tuple)) else self.tokens(words)
        if not toks:
            return set()
        plist = [self.postings(t, zone) for t in toks]
        if any(p is None for p in plist):
            return set()
        # Intersect starting from the rarest term (query optimisation by increasing df).
        order = sorted(range(len(toks)), key=lambda i: len(plist[i].docs))
        common = plist[order[0]].docs
        for i in order[1:]:
            common = np.intersect1d(common, plist[i].docs, assume_unique=True)
            if not len(common):
                return set()
        hits = set()
        for d in common:
            starts = None
            for k, p in enumerate(plist):
                i = int(np.searchsorted(p.docs, d))
                s = set((p.positions(i) - k).tolist())
                starts = s if starts is None else starts & s
                if not starts:
                    break
            if starts:
                hits.add(int(d))
        return hits

    def query_vector(self, term_counts):
        """ltc query weights over term ids: log tf times idf, cosine normalised."""
        w = {t: (1 + math.log10(c)) * self.idf[t] for t, c in term_counts.items() if self.idf[t] > 0 and c > 0}
        norm = math.sqrt(sum(v * v for v in w.values())) or 1.0
        return {t: v / norm for t, v in w.items()}

    def cosine_scores(self, qvec):
        """Term-at-a-time lnc.ltc cosine: one accumulator per document, touched only via postings."""
        acc = np.zeros(self.N, dtype=np.float64)
        for t, wq in qvec.items():
            p = self.post["body"].get(t)
            if p is None:
                continue
            acc[p.docs] += wq * (1 + np.log10(p.tfs))
        return acc / self.norms

    def cosine(self, qvec, allowed=None, k=10):
        """Top k by lnc.ltc cosine, selected with a heap."""
        return self._top(self.cosine_scores(qvec), allowed, k)

    def bm25(self, term_counts, allowed=None, k=10):
        return self._top(self.bm25_scores(term_counts), allowed, k)

    def bm25_scores(self, term_counts):
        acc = np.zeros(self.N, dtype=np.float64)
        L = self.lengths["body"].astype(np.float64)
        for t, qtf in term_counts.items():
            p = self.post["body"].get(t)
            if p is None:
                continue
            idf = math.log(1 + (self.N - len(p.docs) + 0.5) / (len(p.docs) + 0.5))
            tf = p.tfs.astype(np.float64)
            acc[p.docs] += idf * tf * (BM25_K1 + 1) / (tf + BM25_K1 * (1 - BM25_B + BM25_B * L[p.docs] / self.avg_len))
        return acc

    def _top(self, acc, allowed, k):
        if allowed is not None:
            acc = np.where(allowed, acc, 0.0)
        nz = np.nonzero(acc)[0]
        best = heapq.nlargest(k, nz.tolist(), key=acc.__getitem__)
        return [(d, float(acc[d])) for d in best]

    def doc_terms(self, n):
        tids, tfs = self.fwd[n]
        return dict(zip(tids.tolist(), tfs.tolist()))

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            return pickle.load(f)


def index_path(stemming=False):
    return config.INDEX_DIR / ("index_stem.pkl" if stemming else "index.pkl")


def build_from_store(stemming=False, con=None, only=None):
    """Build over the store; only, if given, is the doc_id snapshot to index (the live crawl keeps adding)."""
    con = con or store.connect()
    docs = store.iter_docs(con)
    if only is not None:
        docs = (d for d in docs if d["doc_id"] in only)
    idx = Index(stemming=stemming).build(docs)
    idx.save(index_path(stemming))
    return idx


_cache = {}


def load(stemming=False):
    if stemming not in _cache:
        _cache[stemming] = Index.load(index_path(stemming))
    return _cache[stemming]
