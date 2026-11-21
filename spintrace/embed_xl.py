"""Multilingual sentence embeddings (paraphrase-multilingual-MiniLM-L12-v2) for cross-language verification.

The model was trained so that a sentence and its translation map to nearby unit vectors. SpinTrace
uses it only on the handful of candidates the anchor query has already retrieved, plus one head
vector per English document for the dense baseline.
"""
import pickle
import sqlite3

import numpy as np

from . import config, text as tx, text_indic as ti

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
CACHE = config.INDEX_DIR / "xl_emb.pkl"
DENSE = config.INDEX_DIR / "xl_dense.pkl"
MAX_SENTS = 40
HEAD_CHARS = 700


def connect_ro():
    """Read-only handle on the store: the live crawler owns writes (WAL lets both run)."""
    con = sqlite3.connect(f"file:{config.DB_PATH}?mode=ro", uri=True, timeout=60)
    con.row_factory = sqlite3.Row
    return con


def split(text):
    """Sentences in the text's own script."""
    if tx.script_of(text) == "latn":
        return tx.sentences(text)[:MAX_SENTS] or [text[:500]]
    return ti.sentences(text)[:MAX_SENTS] or [text[:500]]


def head(text, title=None):
    t = ((title + ". ") if title else "") + " ".join(text.split())
    return t[:HEAD_CHARS]


class Embedder:
    def __init__(self, cache=CACHE):
        self.path = cache
        self.model = None
        self.cache = {}
        self.dirty = False
        if cache.exists():
            with cache.open("rb") as f:
                self.cache = pickle.load(f)

    def _model(self):
        if self.model is None:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(MODEL_NAME, device="cpu")
        return self.model

    def encode(self, texts, batch_size=32):
        return self._model().encode(texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False)

    def prefetch(self, items):
        """Batch-encode the sentences of many documents; items are (doc_id, text)."""
        todo = [(d, t) for d, t in items if d not in self.cache]
        if not todo:
            return
        all_sents, spans = [], []
        for d, t in todo:
            sents = split(t)
            spans.append((d, len(all_sents), len(all_sents) + len(sents)))
            all_sents.extend(sents)
        embs = self.encode(all_sents).astype(np.float16)
        for d, a, b in spans:
            self.cache[d] = (all_sents[a:b], embs[a:b])
        self.dirty = True

    def doc(self, doc_id, text):
        if doc_id not in self.cache:
            self.prefetch([(doc_id, text)])
        return self.cache[doc_id]

    def save(self):
        if self.dirty:
            with self.path.open("wb") as f:
                pickle.dump(self.cache, f, protocol=pickle.HIGHEST_PROTOCOL)
            self.dirty = False


_emb = None


def get():
    global _emb
    if _emb is None:
        _emb = Embedder()
    return _emb


def build_dense(doc_ids, con=None, batch=256, log_every=2000):
    """One head vector per document (title and opening, truncated by the model at 128 word pieces),
    the dense baseline's corpus side. Resumable: vectors already in the cache are kept."""
    con = con or connect_ro()
    have = {}
    if DENSE.exists():
        with DENSE.open("rb") as f:
            have = pickle.load(f)
    emb = get()
    todo = [d for d in doc_ids if d not in have]
    for i in range(0, len(todo), batch):
        chunk = todo[i:i + batch]
        rows = {r["doc_id"]: r for r in con.execute(
            f"SELECT doc_id, title, text FROM docs WHERE doc_id IN ({','.join('?' * len(chunk))})", chunk)}
        ids = [d for d in chunk if d in rows]
        vecs = emb.encode([head(rows[d]["text"], rows[d]["title"]) for d in ids], batch_size=64).astype(np.float16)
        have.update(zip(ids, vecs))
        if (i // batch) % max(1, log_every // batch) == 0 or i + batch >= len(todo):
            with DENSE.open("wb") as f:
                pickle.dump(have, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"dense heads {len(have)}/{len(doc_ids)}", flush=True)
    return have


def load_dense():
    with DENSE.open("rb") as f:
        return pickle.load(f)
