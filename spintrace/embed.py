"""Sentence embeddings (all-MiniLM-L6-v2) with a per-document cache.

The model maps each sentence to a unit vector so that paraphrases land close together; SpinTrace
uses it only to verify candidates the inverted index has already retrieved.
"""
import pickle

import numpy as np

from . import config, text as tx

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CACHE = config.INDEX_DIR / "emb.pkl"
MAX_SENTS = 60


class Embedder:
    def __init__(self):
        self.model = None
        self.cache = {}
        self.dirty = False
        if CACHE.exists():
            with CACHE.open("rb") as f:
                self.cache = pickle.load(f)

    def _model(self):
        if self.model is None:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(MODEL_NAME, device="cpu")
        return self.model

    def encode(self, texts):
        return self._model().encode(texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False)

    def doc(self, doc_id, text):
        """(sentences, embeddings) for a document, computed once."""
        hit = self.cache.get(doc_id)
        if hit is None:
            sents = tx.sentences(text)[:MAX_SENTS] or [text[:500]]
            hit = (sents, self.encode(sents).astype(np.float16))
            self.cache[doc_id] = hit
            self.dirty = True
        return hit

    def prefetch(self, items):
        """Batch-encode many documents at once; items are (doc_id, text)."""
        todo = [(d, t) for d, t in items if d not in self.cache]
        if not todo:
            return
        all_sents, spans = [], []
        for d, t in todo:
            sents = tx.sentences(t)[:MAX_SENTS] or [t[:500]]
            spans.append((d, len(all_sents), len(all_sents) + len(sents)))
            all_sents.extend(sents)
        embs = self.encode(all_sents).astype(np.float16)
        for d, a, b in spans:
            self.cache[d] = (all_sents[a:b], embs[a:b])
        self.dirty = True

    def doc_vector(self, doc_id, text):
        """Mean of sentence vectors, renormalised: the dense-retrieval baseline's document vector."""
        _, e = self.doc(doc_id, text)
        v = e.astype(np.float32).mean(axis=0)
        return v / (np.linalg.norm(v) or 1.0)

    def save(self):
        if self.dirty:
            with CACHE.open("wb") as f:
                pickle.dump(self.cache, f, protocol=pickle.HIGHEST_PROTOCOL)
            self.dirty = False


_emb = None


def get():
    global _emb
    if _emb is None:
        _emb = Embedder()
    return _emb
