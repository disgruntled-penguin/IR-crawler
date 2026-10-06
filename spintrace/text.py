"""The single text pipeline shared by the index, shingling, queries and the detection signals."""
import re
import unicodedata
from functools import lru_cache

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

STOPWORDS = frozenset(ENGLISH_STOP_WORDS) | {"said", "says", "according", "also", "would", "could", "told"}

QUOTE_MAP = str.maketrans({"“": '"', "”": '"', "„": '"', "‘": "'", "’": "'", "–": "-", "—": "-"})
TOKEN = re.compile(r"\d+(?:[.,]\d+)*|[^\W\d_]+(?:'[^\W\d_]+)?")
QUOTE = re.compile(r'"([^"]{20,400})"')
SENT_SPLIT = re.compile(r"(?<=[.!?])[\"']?\s+(?=[\"']?[A-Z0-9])")
CAPS = re.compile(r"\b[A-Z][a-zA-Z'\-]+(?:\s+(?:of|the|de|al|bin|von|van|for)?\s*[A-Z][a-zA-Z'\-]+)*")
NUMBER = re.compile(r"\b\d+(?:[.,]\d+)*\b")


def clean(text):
    return unicodedata.normalize("NFKC", text).translate(QUOTE_MAP)


def fold(text):
    """Case and accent folding: "Teherán" and "teheran" become one term."""
    text = unicodedata.normalize("NFKD", clean(text).lower())
    return "".join(c for c in text if not unicodedata.combining(c))


def _norm_token(t):
    # Numbers keep their value: "1,394" and "1394" index as the same term; "3.5" keeps its decimal.
    if t[0].isdigit():
        return t.replace(",", "")
    return t[:-2] if t.endswith("'s") else t


@lru_cache(maxsize=200_000)
def stem(t):
    return _porter().stem(t)


@lru_cache(maxsize=1)
def _porter():
    from nltk.stem import PorterStemmer
    return PorterStemmer()


def tokens(text, stemming=False):
    """Lowercased tokens with positions preserved (stopwords kept so phrase queries work)."""
    out = [_norm_token(t) for t in TOKEN.findall(fold(text))]
    if stemming:
        out = [t if t[0].isdigit() else stem(t) for t in out]
    return out


def content_terms(toks):
    return [t for t in toks if t not in STOPWORDS and (len(t) > 1 or t.isdigit())]


def sentences(text):
    text = re.sub(r"\s+", " ", clean(text)).strip()
    return [s.strip() for s in SENT_SPLIT.split(text) if len(s.split()) >= 4]


def quotes(text, min_words=5):
    """Direct quotes: the exact phrases a rewrite usually has to keep."""
    return [q.strip() for q in QUOTE.findall(clean(text)) if len(q.split()) >= min_words]


def facts(text):
    """Names and numbers: capitalised spans not starting a sentence, and numeric values.

    A light heuristic standing in for NER; it keeps the property that matters here, that these
    strings survive paraphrase.
    """
    t = clean(text)
    names = set()
    for m in CAPS.finditer(t):
        span = m.group(0)
        start = m.start()
        prev = t[max(0, start - 2):start]
        if (start == 0 or prev.strip() in (".", "!", "?", '"', "")) and " " not in span:
            continue
        words = span.split()
        while words and words[0].lower() in STOPWORDS:
            words = words[1:]
        name = re.sub(r"'s\b", "", fold(" ".join(words)))
        if words and name not in STOPWORDS:
            names.add(name)
    nums = {n.replace(",", "") for n in NUMBER.findall(t) if len(n.replace(",", "")) >= 2}
    return names, nums
