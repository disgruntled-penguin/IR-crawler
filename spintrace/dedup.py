"""The textbook "content seen?" check: exact hash, word shingles, Jaccard, MinHash and LSH banding."""
import hashlib
import zlib
from collections import defaultdict

import numpy as np

from . import text as tx

SHINGLE_K = 5
N_PERM = 128
BANDS, ROWS = 32, 4
PRIME = (1 << 31) - 1
_rng = np.random.RandomState(358)
A = _rng.randint(1, PRIME, size=N_PERM).astype(np.uint64)
B = _rng.randint(0, PRIME, size=N_PERM).astype(np.uint64)


def exact_hash(text):
    return hashlib.sha1(" ".join(tx.tokens(text)).encode()).hexdigest()


def shingles(text, k=SHINGLE_K):
    """Set of hashed word k-grams over the shared token stream (stopwords kept, no stemming)."""
    toks = tx.tokens(text)
    if len(toks) < k:
        return {zlib.crc32(" ".join(toks).encode())} if toks else set()
    return {zlib.crc32(" ".join(toks[i:i + k]).encode()) for i in range(len(toks) - k + 1)}


def jaccard(a, b):
    return len(a & b) / len(a | b) if a and b else 0.0


def containment(a, b):
    """Share of a's shingles that also occur in b; asymmetric, so a short copy inside a long original scores high."""
    return len(a & b) / len(a) if a else 0.0


def minhash(sh):
    """Signature: for each of N_PERM hash functions (a*x + b) mod p, the minimum over the shingle set."""
    if not sh:
        return np.full(N_PERM, PRIME, dtype=np.uint64)
    x = np.fromiter(sh, dtype=np.uint64, count=len(sh)) % PRIME
    return ((np.outer(A, x) + B[:, None]) % PRIME).min(axis=1)


def est_jaccard(s1, s2):
    return float(np.mean(s1 == s2))


class LSH:
    """Banding: signatures that agree on all ROWS of any band become candidate pairs."""

    def __init__(self):
        self.buckets = [defaultdict(list) for _ in range(BANDS)]
        self.sigs = {}

    def add(self, key, sig):
        self.sigs[key] = sig
        for b in range(BANDS):
            self.buckets[b][sig[b * ROWS:(b + 1) * ROWS].tobytes()].append(key)

    def query(self, sig):
        out = set()
        for b in range(BANDS):
            out.update(self.buckets[b].get(sig[b * ROWS:(b + 1) * ROWS].tobytes(), ()))
        return out

    def pairs(self):
        seen = set()
        for buckets in self.buckets:
            for keys in buckets.values():
                if 1 < len(keys) < 200:
                    for i in range(len(keys)):
                        for j in range(i + 1, len(keys)):
                            p = (keys[i], keys[j]) if keys[i] < keys[j] else (keys[j], keys[i])
                            if p not in seen:
                                seen.add(p)
                                yield p
