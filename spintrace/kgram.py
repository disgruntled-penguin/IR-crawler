"""Tolerant retrieval of names across scripts: a k-gram index over phonetic keys (IIR chapter 3).

Every English document's names (text.facts) are split into words and reduced to phonetic keys.
The k-gram index maps each 3-gram of '$key$' to the keys containing it. A transliterated Hindi
name is looked up by k-gram Jaccard first (cheap, through postings), then confirmed by edit
distance, and each surviving key leads to the English index terms spelled that way.
"""
import pickle
import time
from collections import Counter, defaultdict

import numpy as np

from . import config, text as tx, text_indic as ti, translit as tr

K = 3
MIN_JACCARD = 0.3
MAX_EDIT = 2
MIN_INDIC_DOCS = 50
INDIC_DF_BEFORE = 1795285800  # 2026-11-22 00:00 IST
PATH = config.INDEX_DIR / "xl_kgram.pkl"


def grams(s, k=K):
    s = f"${s}$"
    return {s[i:i + k] for i in range(max(1, len(s) - k + 1))}


def levenshtein(a, b):
    """Edit distance by dynamic programming, one row at a time."""
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


class KGramIndex:
    def __init__(self):
        self.keys = []
        self.key_id = {}
        self.post = defaultdict(list)
        self.terms = []

    def add(self, key, term, count=1):
        kid = self.key_id.get(key)
        if kid is None:
            kid = self.key_id[key] = len(self.keys)
            self.keys.append(key)
            self.terms.append(Counter())
            for g in grams(key):
                self.post[g].append(kid)
        self.terms[kid][term] += count

    def freeze(self):
        self.post = dict(self.post)
        self.ngrams = [len(grams(k)) for k in self.keys]
        return self

    def lookup(self, q, min_jaccard=MIN_JACCARD, max_edit=MAX_EDIT, limit=5):
        """Keys close to q: [(key, jaccard, edit distance, {term: count})], best first.

        Short keys get at most one edit, since two edits turn any 3-letter key into any other.
        """
        gq = grams(q)
        overlap = Counter()
        for g in sorted(gq):
            for kid in self.post.get(g, ()):
                overlap[kid] += 1
        max_edit = min(max_edit, 1 if len(q) <= 3 else 2)
        out = []
        for kid, ov in overlap.items():
            jac = ov / (len(gq) + self.ngrams[kid] - ov)
            if jac < min_jaccard:
                continue
            ed = levenshtein(q, self.keys[kid])
            if ed <= max_edit:
                out.append((self.keys[kid], jac, ed, self.terms[kid]))
        out.sort(key=lambda x: (x[2], -x[1], -sum(x[3].values()), x[0]))
        return out[:limit]

    def postings(self, gram):
        return [self.keys[k] for k in self.post.get(gram, ())]


def name_words(text):
    """Single-word name parts from text.facts, the strings a translation has to transliterate."""
    names, _ = tx.facts(text)
    out = []
    for n in sorted(names):
        for w in tx.tokens(n):
            if not w[0].isdigit() and len(w) >= 2 and w not in tx.STOPWORDS:
                out.append(w)
    return out


class Lexicon:
    """The English side of the bridge for one index: plain and voicing-folded k-gram indexes over name
    keys, the exact name words (for the no-fuzzy ablation), and which documents are in Latin script."""

    def __init__(self, doc_ids):
        self.doc_ids = list(doc_ids)
        self.kg = KGramIndex()
        self.kg_fold = KGramIndex()
        self.exact = Counter()
        self.latin = None
        self.indic_df = {}
        self.indic_n = {}
        self.build_seconds = 0.0

    def lookup(self, key, fold=False, **kw):
        return (self.kg_fold if fold else self.kg).lookup(key, **kw)

    def source_idf(self, token):
        """(idf, df) of an Indic word over the crawled pages in its script, or (None, 0) without enough pages."""
        script = tx.INDIC_BLOCKS.get(ord(token[0]) & ~0x7F)
        n = self.indic_n.get(script, 0)
        if n < MIN_INDIC_DOCS:
            return None, 0
        df = self.indic_df[script].get(token, 0)
        return float(np.log10((n + 1) / (df + 1))), df


def indic_df(lex, con, before=INDIC_DF_BEFORE):
    """Document frequency of every word in the crawled Indic-script pages, per script: a word that is
    common in Hindi news is a common word, whatever English name it happens to sound like. Only pages
    fetched before a fixed time count, so a rebuild gives the same table while the crawl goes on."""
    dfs, ns = defaultdict(Counter), Counter()
    for (text,) in con.execute("SELECT text FROM docs WHERE origin='crawl' AND fetched_at < ?", (before,)):
        if not any("\u0900" <= ch <= "\u0d7f" for ch in text[:400]):
            continue
        script = tx.script_of(text)
        if script == "latn":
            continue
        ns[script] += 1
        dfs[script].update({t for t in ti.tokens(text) if ti.is_indic(t)})
    lex.indic_df, lex.indic_n = dict(dfs), dict(ns)


def build(idx, con, path=PATH, log_every=5000):
    t0 = time.time()
    lex = Lexicon(idx.doc_ids)
    latin = []
    for i in range(0, idx.N, 500):
        chunk = idx.doc_ids[i:i + 500]
        rows = {r["doc_id"]: r["text"] for r in con.execute(
            f"SELECT doc_id, text FROM docs WHERE doc_id IN ({','.join('?' * len(chunk))})", chunk)}
        for d in chunk:
            text = rows.get(d, "")
            is_latin = tx.script_of(text) == "latn"
            latin.append(is_latin)
            if not is_latin:
                continue
            for w in name_words(text):
                lex.exact[w] += 1
                k = tr.key(w)
                if len(k) >= 2:
                    lex.kg.add(k, w)
                    lex.kg_fold.add(tr.key(w, fold_voicing=True), w)
        if (i // 500) % max(1, log_every // 500) == 0:
            print(f"lexicon {i}/{idx.N} keys={len(lex.kg.keys)} {time.time() - t0:.0f}s", flush=True)
    lex.latin = np.array(latin, dtype=bool)
    indic_df(lex, con)
    lex.kg.freeze()
    lex.kg_fold.freeze()
    lex.build_seconds = time.time() - t0
    with open(path, "wb") as f:
        pickle.dump(lex, f, protocol=pickle.HIGHEST_PROTOCOL)
    return lex


_cache = {}


def load(path=PATH):
    if path not in _cache:
        if not path.exists():
            raise SystemExit(f"no k-gram lexicon at {path}; run `python -m spintrace build-xl` first")
        with open(path, "rb") as f:
            _cache[path] = pickle.load(f)
    return _cache[path]
