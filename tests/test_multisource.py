import zlib

import numpy as np

from spintrace import text as tx
from spintrace.detect import multisource, pipeline
from spintrace.index import Index

A = ['Health Minister Aden Duale said a 41-year-old Kenyan citizen died in Kisumu on Tuesday.',
     'The patient had travelled from Kampala to Kisumu by bus on 12 March.',
     'Duale said 214 contacts were traced across Kisumu and Siaya counties.',
     '"We have activated our emergency teams in Kisumu," Duale told reporters.',
     'Laboratory tests at the Kenya Medical Research Institute confirmed the infection.',
     'Schools in Kisumu will remain open while contact tracing continues.']
B = ['Norwegian skier Johannes Klaebo won the 50 km race at Holmenkollen on Sunday.',
     'Klaebo finished in 2 hours 1 minute, ahead of Harald Amundsen and Simen Krueger.',
     'It was the eleventh World Cup victory of the season for the Trondheim athlete.',
     'About 30,000 spectators lined the Holmenkollen course in light snow.',
     '"This race means everything to Norwegian skiers," Klaebo said after the finish.',
     'The World Cup final takes place in Lahti next weekend.']
FILLER = ['markets', 'football', 'weather', 'cooking', 'gardening', 'traffic', 'music', 'films']


class BagEmbedder:
    """Deterministic stand-in for the sentence model: hashed bag of content words, so equal sentences match."""

    def __init__(self):
        self.cache = {}

    def doc(self, doc_id, text):
        if doc_id not in self.cache:
            sents = tx.sentences(text) or [text]
            vecs = np.zeros((len(sents), 256), dtype=np.float32)
            for i, s in enumerate(sents):
                for t in tx.content_terms(tx.tokens(s)):
                    vecs[i, zlib.crc32(t.encode()) % 256] += 1
            vecs /= np.maximum(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-9)
            self.cache[doc_id] = (sents, vecs)
        return self.cache[doc_id]

    def prefetch(self, items):
        for d, t in items:
            self.doc(d, t)


class ToyContext:
    def __init__(self):
        docs = [{"doc_id": "a", "text": " ".join(A), "published": 100.0, "site": "a.com"},
                {"doc_id": "b", "text": " ".join(B), "published": 120.0, "site": "b.com"}]
        for i, w in enumerate(FILLER * 3):
            docs.append({"doc_id": f"f{i}", "text": f"A story number {i} about {w} and the local {w} club. "
                         f"Residents discussed {w} for an hour on Monday evening.", "published": 50.0,
                         "site": f"f{i}.com"})
        for d in docs:
            d.update(title="", kind="original", origin="crawl", url=f"https://{d['site']}/{d['doc_id']}")
        self.rows = docs
        self.idx = Index().build(docs)
        self.emb = BagEmbedder()
        self.feats = pipeline.TextFeatures(self)

    def doc(self, n):
        return self.rows[n]

    def num(self, doc_id):
        return self.idx.doc_num[doc_id]


MODEL = dict(multisource.DEFAULT, threshold=0.5, stop_share=0.2, cover=0.62)


def found(ctx, text):
    tr = multisource.trace(ctx, multisource.Suspect("s", text, 500.0, "farm.example"), MODEL)
    fin = multisource.finalize(tr)
    return [ctx.idx.doc_ids[s["doc"]] for s in fin["sources"]], fin


def test_splice_of_two_originals_recovers_both_with_their_shares():
    ctx = ToyContext()
    srcs, fin = found(ctx, " ".join(A[:3] + B[:3]))
    assert sorted(srcs) == ["a", "b"]
    assert [round(s["share"], 2) for s in fin["sources"]] == [0.5, 0.5]
    assert fin["unexplained"] == 0.0


def test_single_source_rewrite_yields_one_source():
    ctx = ToyContext()
    srcs, fin = found(ctx, " ".join(A[1:]))
    assert srcs == ["a"]
    assert fin["sources"][0]["share"] == 1.0


def test_shares_sum_to_at_most_one_and_boilerplate_stays_unexplained():
    ctx = ToyContext()
    text = " ".join(B[:4] + A[:2] + ["Subscribe to our newsletter for the latest stories every single morning."])
    srcs, fin = found(ctx, text)
    total = sum(s["share"] for s in fin["sources"]) + fin["unexplained"]
    assert abs(total - 1.0) < 1e-9
    assert sum(s["share"] for s in fin["sources"]) <= 1.0
    assert fin["unexplained"] > 0
    assert set(srcs) == {"a", "b"}


def test_later_articles_are_never_sources():
    ctx = ToyContext()
    tr = multisource.trace(ctx, multisource.Suspect("s", " ".join(A[:3] + B[:3]), 110.0, "farm.example"), MODEL)
    srcs = [s["doc_id"] for s in multisource.finalize(tr)["sources"]]
    assert "b" not in srcs and "a" in srcs


def test_finalize_matches_an_early_stopping_run():
    ctx = ToyContext()
    sus = multisource.Suspect("s", " ".join(A[:3] + B[:3]), 500.0, "farm.example")
    full = multisource.trace(ctx, sus, MODEL, threshold=0.0, stop_share=0.0)
    early = multisource.trace(ctx, sus, MODEL)
    a = multisource.finalize(full, MODEL["threshold"], MODEL["stop_share"], MODEL["cover"])
    b = multisource.finalize(early)
    assert [s["doc"] for s in a["sources"]] == [s["doc"] for s in b["sources"]]
    assert a["unexplained"] == b["unexplained"]
