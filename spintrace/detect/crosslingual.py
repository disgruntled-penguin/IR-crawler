"""Cross-language derivative detection: an Indic-script suspect against the English index.

A translation shares no words with its source, so shingles, tf-idf and quote phrases have nothing
to match. What survives translation is numbers, dates and names, so the suspect's query to the
English index is built from those anchors:
  numbers  canonical values, looked up as the English index terms that spell them
  dates    month terms, and 'day month' / 'month day' as positional phrase queries
  latin    Latin-script tokens left in the text (acronyms, brands)
  names    transliterated, reduced to a phonetic key, matched through the k-gram index to the
           English name words, which are then index terms
The ranked candidates are verified with anchor overlaps and a multilingual sentence alignment.
"""
import math
import time
from collections import Counter

import numpy as np

from .. import config, embed_xl, kgram, text as tx, text_indic as ti, translit as tr
from . import candidates, pipeline, signals

WEIGHTS = {"number": 1.0, "date": 0.5, "latin": 1.0, "name": 1.0, "weekday": 0.2}
DATE_PHRASE_WEIGHT = 0.2
WINDOW_DAYS = 30
TOP_K = 20
VERIFY_TOP = 5
NAME_MATCHES = 3
TERMS_PER_KEY = 2
EDIT_CONF = {0: 1.0, 1: 0.5, 2: 0.25}
ALIGN_THRESHOLD = 0.6
IDF_GAP = 1.0
MIN_SRC_DF = 3
SCALES = [(1e3, "thousand"), (1e5, "lakh"), (1e6, "million"), (1e7, "crore"), (1e9, "billion")]
MODEL_PATH = config.RESULTS / "verifier_xl.json"
FEATURES = ["xl_cos", "number_overlap", "name_overlap", "date_overlap", "latin_overlap", "align_coverage",
            "align_mean", "align_order", "ordered_coverage", "back_coverage"]
DEFAULT = {"features": FEATURES, "weights": [2.0, 2.0, 2.0, 0.5, 0.5, 3.0, 2.0, 1.0, 3.0, 1.0], "bias": -7.0,
           "threshold": 0.5, "source": "default"}
NOT_NAMES = set(ti.MULTIPLIERS) | set(ti.NUMBER_WORDS) | set(ti.MONTH_WORDS) | set(ti.WEEKDAYS)


class Options:
    """Switches for the ablations; the defaults are the full system."""

    def __init__(self, names=True, numbers=True, dates=True, latin=True, fuzzy="kgram", idf_check=True):
        self.names, self.numbers, self.dates, self.latin = names, numbers, dates, latin
        self.fuzzy = fuzzy
        self.idf_check = idf_check


FULL = Options()


class Anchors:
    """What a translation keeps, extracted from the suspect once."""

    def __init__(self, text, lex, idx, opts=FULL):
        toks = ti.tokens(text)
        self.numbers = Counter(ti.numbers(text, toks)) if opts.numbers else Counter()
        self.dates = sorted(set(ti.dates(text, toks))) if opts.dates else []
        self.weekdays = Counter(ti.weekday_terms(toks)) if opts.dates else Counter()
        self.latin = Counter(ti.latin_tokens(text)) if opts.latin else Counter()
        self.names = []
        self.toks = toks
        if opts.names:
            seen = {}
            for t in toks:
                if not ti.is_indic(t) or t in ti.HI_STOPWORDS or t in NOT_NAMES:
                    continue
                if t in seen:
                    seen[t]["tf"] += 1
                    continue
                fold = 0x0B80 <= ord(t[0]) < 0x0C00
                latin = tr.word(t)
                k = tr.key(latin, fold)
                if len(k) < 2:
                    continue
                src_idf, src_df = lex.source_idf(t) if opts.idf_check else (None, 0)
                seen[t] = {"token": t, "translit": latin, "key": k, "fold": fold, "tf": 1, "src_idf": src_idf,
                           "src_df": src_df, "matches": match_name(lex, latin, k, fold, opts.fuzzy)}
            for n in seen.values():
                n["matches"] = [m for m in n["matches"] if any(plausible(idx, n, t) for t, _ in m[3].most_common(TERMS_PER_KEY))]
            self.names = [n for n in seen.values() if n["matches"]]

    def counts(self):
        return {"numbers": len(self.numbers), "dates": len(self.dates), "latin": len(self.latin),
                "names": len(self.names)}


def match_name(lex, latin, k, fold, mode="kgram"):
    """[(english key, jaccard, edit distance, {term: count})] for one transliterated token.

    mode 'kgram' is the full tolerant lookup; 'key' keeps only exact key matches; 'exact' needs the
    transliteration itself to be an English name word (the no-fuzzy ablation).
    """
    if mode == "exact":
        c = lex.exact.get(latin)
        return [(latin, 1.0, 0, Counter({latin: c}))] if c else []
    hits = lex.lookup(k, fold=fold, limit=NAME_MATCHES)
    if mode == "key":
        hits = [h for h in hits if h[2] == 0]
    return hits


def number_forms(value):
    """English index terms that can spell a canonical number: itself, and its mantissa before a scale
    word ('35' of '35 million', '3.5' of '3.5 crore'). Mantissas are weaker evidence."""
    v = float(value)
    forms = {value: 1.0}
    for scale, _ in SCALES:
        if v >= scale:
            m = v / scale
            if abs(m * 100 - round(m * 100)) < 1e-6 and m < 1000:
                forms.setdefault(ti.canon(m), 0.5)
    return forms


def plausible(idx, name, term):
    """A name keeps roughly its document frequency across languages. A word common in the source
    language (seen in MIN_SRC_DF or more pages) that sounds like a much rarer English name is a common
    word, not that name: रात (night) is not Rohit, प्रधान (prime) is not Pepperdine."""
    if name["src_idf"] is None or name["src_df"] < MIN_SRC_DF:
        return True
    tid = idx.vocab.get(term)
    return tid is None or float(idx.idf[tid]) - name["src_idf"] <= IDF_GAP


def build_query(idx, anchors):
    """{term id: weight} with weight = anchor-type weight x match confidence x idf, cosine normalised;
    and a list of (term, idf, anchor type, source) for display."""
    raw, why = Counter(), {}

    def add(term, w, kind, src):
        tid = idx.vocab.get(term)
        if tid is None or idx.idf[tid] <= 0 or w <= 0:
            return
        raw[tid] += w
        why.setdefault(tid, (kind, src))

    for num, c in anchors.numbers.items():
        for form, conf in number_forms(num).items():
            add(form, WEIGHTS["number"] * conf * (1 + math.log10(c)), "number", num)
    for m, d in anchors.dates:
        add(ti.MONTHS[m - 1], WEIGHTS["date"], "date", f"{d}/{m}")
        add(str(d), WEIGHTS["date"], "date", f"{d}/{m}")
    for w, c in anchors.weekdays.items():
        add(w, WEIGHTS["weekday"], "weekday", w)
    for t, c in anchors.latin.items():
        for term in tx.tokens(t):
            add(term, WEIGHTS["latin"] * (1 + math.log10(c)), "latin", t)
    for n in anchors.names:
        confs = [(m, EDIT_CONF.get(m[2], 0.0) * m[1]) for m in n["matches"]]
        total = max(1.0, sum(c for _, c in confs))
        for (key, _, _, terms), conf in confs:
            for term, _ in terms.most_common(TERMS_PER_KEY):
                if not plausible(idx, n, term):
                    continue
                add(term, WEIGHTS["name"] * (1 + math.log10(n["tf"])) * conf / total, "name", n["token"])
    q = {t: w * float(idx.idf[t]) for t, w in raw.items()}
    norm = math.sqrt(sum(v * v for v in q.values())) or 1.0
    q = {t: v / norm for t, v in q.items()}
    shown = sorted(q, key=lambda t: -q[t])
    return q, [(idx.terms[t], round(float(idx.idf[t]), 3), why[t][0], why[t][1], round(q[t], 3)) for t in shown]


def date_hits(idx, dates, allowed):
    """Share of the suspect's dates each document contains as the phrase 'July 26' or '26 July'."""
    hits, touched = Counter(), 0
    for m, d in dates:
        found = set()
        for words in ([ti.MONTHS[m - 1], str(d)], [str(d), ti.MONTHS[m - 1]]):
            found |= idx.phrase(words)
        touched += sum(len(p.docs) for p in (idx.postings(ti.MONTHS[m - 1]), idx.postings(str(d))) if p is not None)
        for x in found:
            if allowed[x]:
                hits[x] += 1
    return {x: c / len(dates) for x, c in hits.items()}, touched


def allowed_mask(idx, lex, published, site=None, exclude=None, window_days=WINDOW_DAYS):
    """Earlier English documents from other sites inside the time window; undated documents stay in."""
    mask = candidates.mask_before(idx, published, site) & lex.latin
    if published is not None and not np.isnan(published) and window_days:
        pub = idx.meta["published"]
        mask &= np.isnan(pub) | (pub >= published - window_days * 86400)
    if exclude is not None:
        mask &= ~exclude
    return mask


def retrieve(idx, lex, text, published=None, site=None, exclude=None, k=TOP_K, opts=FULL, anchors=None):
    """Return (candidates, stats, anchors); each candidate is dict(doc, cos, date_frac, score)."""
    t0 = time.perf_counter()
    anchors = anchors or Anchors(text, lex, idx, opts)
    allowed = allowed_mask(idx, lex, published, site, exclude)
    qvec, shown = build_query(idx, anchors)
    cos = dict(idx.cosine(qvec, allowed, k=k)) if qvec else {}
    dh, d_touched = date_hits(idx, anchors.dates, allowed) if anchors.dates else ({}, 0)
    docs = set(cos) | set(dh)
    cands = [{"doc": d, "cos": cos.get(d, 0.0), "date_frac": dh.get(d, 0.0)} for d in docs]
    for c in cands:
        c["score"] = c["cos"] + DATE_PHRASE_WEIGHT * c["date_frac"]
    cands.sort(key=lambda c: -c["score"])
    touched = int(sum(len(idx.post["body"][t].docs) for t in qvec if t in idx.post["body"])) + d_touched
    stats = {"anchors": anchors.counts(), "query_terms": shown, "postings_touched": touched,
             "allowed": int(allowed.sum()), "ms": 1000 * (time.perf_counter() - t0)}
    return cands[:k], stats, anchors


class CandidateFacts:
    """Anchors of an English candidate, computed lazily and cached per document."""

    def __init__(self, text):
        toks = ti.tokens(text)
        self.numbers = set(ti.numbers(text, toks))
        self.dates = set(ti.dates(text, toks))
        self.tokens = set(tx.tokens(text))
        words = kgram.name_words(text)
        self.keys = {tr.key(w) for w in words} | {tr.key(w, True) for w in words}


class Pair:
    """A suspect and one English candidate; alignment() has the shape signals.Pair gives, so
    pipeline.provenance and pipeline.direction work on it unchanged."""

    def __init__(self, s, c, emb, anchors, cfacts, retrieval=None):
        self.s, self.c = s, c
        self.emb = emb
        self.anchors = anchors
        self.cf = cfacts
        self.retrieval = retrieval or {}
        self._align = None

    def alignment(self):
        if self._align is None:
            _, es = self.emb.doc(self.s["doc_id"], self.s["text"])
            _, ec = self.emb.doc(self.c["doc_id"], self.c["text"])
            sims = es.astype(np.float32) @ ec.astype(np.float32).T
            self._align = (sims.max(axis=1), sims.argmax(axis=1), sims)
        return self._align


def xl_cos(p):
    return float(p.retrieval.get("cos", 0.0))


def number_overlap(p):
    s = set(p.anchors.numbers)
    return len(s & p.cf.numbers) / len(s) if s else 0.0


def name_overlap(p):
    """Share of the suspect's matched names whose key is within one edit of a candidate name key."""
    names = p.anchors.names
    if not names:
        return 0.0
    hit = 0
    for n in names:
        k = n["key"]
        if k in p.cf.keys or any(abs(len(k) - len(c)) <= 1 and kgram.levenshtein(k, c) <= 1 for c in p.cf.keys
                                 if len(k) >= 4):
            hit += 1
    return hit / len(names)


def date_overlap(p):
    s = set(p.anchors.dates)
    return len(s & p.cf.dates) / len(s) if s else 0.0


def latin_overlap(p):
    s = set(p.anchors.latin)
    return len([t for t in s if t in p.cf.tokens]) / len(s) if s else 0.0


def align_coverage(p):
    best, _, _ = p.alignment()
    return float(np.mean(best >= ALIGN_THRESHOLD))


def align_mean(p):
    best, _, _ = p.alignment()
    return float(np.clip(best.mean(), 0, 1))


def align_order(p):
    best, arg, _ = p.alignment()
    matched = arg[best >= ALIGN_THRESHOLD].tolist()
    return signals._lis(matched) / len(matched) if len(matched) >= 2 else 0.0


def ordered_coverage(p):
    best, arg, _ = p.alignment()
    matched = arg[best >= ALIGN_THRESHOLD].tolist()
    return signals._lis(matched) / len(best) if len(best) else 0.0


def back_coverage(p):
    _, _, sims = p.alignment()
    return float(np.mean(sims.max(axis=0) >= ALIGN_THRESHOLD))


SIGNALS = {"xl_cos": xl_cos, "number_overlap": number_overlap, "name_overlap": name_overlap,
           "date_overlap": date_overlap, "latin_overlap": latin_overlap, "align_coverage": align_coverage,
           "align_mean": align_mean, "align_order": align_order, "ordered_coverage": ordered_coverage,
           "back_coverage": back_coverage}


def compute(pair):
    return {k: f(pair) for k, f in SIGNALS.items()}


def load_model():
    import json
    if MODEL_PATH.exists():
        return json.loads(MODEL_PATH.read_text())
    return DEFAULT


def save_model(model):
    import json
    config.RESULTS.mkdir(exist_ok=True)
    MODEL_PATH.write_text(json.dumps(model, indent=2))


class Context:
    """The English index, its name lexicon, the multilingual embedder and a read-only store."""

    def __init__(self, idx=None, lex=None, con=None, emb=None, model=None, lex_path=kgram.PATH):
        from .. import index
        self.idx = idx or index.load(False)
        self.lex = lex or kgram.load(lex_path)
        if list(self.lex.doc_ids[:50]) != list(self.idx.doc_ids[:50]) or len(self.lex.doc_ids) != self.idx.N:
            raise SystemExit("the k-gram lexicon was built for another index; run `python -m spintrace build-xl`")
        self.con = con or embed_xl.connect_ro()
        self.emb = emb or embed_xl.get()
        self.model = model or load_model()
        self._rows = {}
        self._facts = {}

    def row(self, n):
        """Store row of index document n (title, text, url, site, published), fetched on demand."""
        r = self._rows.get(n)
        if r is None:
            r = self.con.execute("SELECT doc_id, url, site, title, text, published, meta FROM docs WHERE doc_id=?",
                                 (self.idx.doc_ids[n],)).fetchone()
            r = self._rows[n] = dict(r) if r else {"doc_id": self.idx.doc_ids[n], "url": "", "site": "", "title": "",
                                                   "text": "", "published": None, "meta": "{}"}
        return r

    def facts(self, n):
        f = self._facts.get(n)
        if f is None:
            f = self._facts[n] = CandidateFacts(self.row(n)["text"])
        return f


def verify(ctx, s, cands, anchors, k=VERIFY_TOP):
    """Signals and provenance for the top k candidates; s is a dict with doc_id, text, published."""
    top = cands[:k]
    ctx.emb.prefetch([(s["doc_id"], s["text"])] + [(ctx.row(c["doc"])["doc_id"], ctx.row(c["doc"])["text"]) for c in top])
    rows = []
    for c in top:
        cd = ctx.row(c["doc"])
        pair = Pair(s, cd, ctx.emb, anchors, ctx.facts(c["doc"]), c)
        verdict, why = pipeline.provenance(s, cd, pair)
        rows.append({"doc": c["doc"], "doc_id": cd["doc_id"], "url": cd["url"], "site": cd["site"],
                     "retrieval": c["score"], "signals": compute(pair), "verdict": verdict, "why": why,
                     "published": pipeline.date_of(cd)})
    return rows


def decide(verified, model):
    """Best verified candidate; among near ties the earliest is the likelier origin (as in pipeline.analyse)."""
    for v in verified:
        v["prob"] = score(v["signals"], model)
    verified.sort(key=lambda v: -v["prob"])
    best = verified[0] if verified else None
    if best:
        close = [v for v in verified if v["prob"] >= best["prob"] - pipeline.TIE_MARGIN
                 and v["verdict"] != "uncertain" and v["prob"] >= model["threshold"]]
        if close:
            best = min(close, key=lambda v: v["published"] if v["published"] is not None else float("inf"))
    flagged = bool(best and best["prob"] >= model["threshold"] and best["verdict"] != "uncertain")
    return flagged, (best if flagged else None)


def score(sig, model):
    z = model["bias"] + sum(w * sig.get(f, 0.0) for f, w in zip(model["features"], model["weights"]))
    return 1 / (1 + math.exp(-z))


def analyse_text(ctx, s, exclude=None, opts=FULL, model=None):
    """Full report for a suspect dict (doc_id, text, published, site, url) that need not be indexed."""
    model = model or ctx.model
    cands, stats, anchors = retrieve(ctx.idx, ctx.lex, s["text"], s.get("published"), s.get("site"), exclude,
                                     opts=opts)
    t = time.perf_counter()
    verified = verify(ctx, s, cands, anchors)
    flagged, best = decide(verified, model)
    stats["verify_ms"] = 1000 * (time.perf_counter() - t)
    return {"doc_id": s["doc_id"], "url": s.get("url"), "stats": stats, "anchors": anchors, "candidates": cands,
            "verified": verified, "flagged": flagged, "source": best, "crosslingual": True}


def analyse(pctx, n):
    """Entry point from pipeline.analyse for an indexed document that is not in Latin script."""
    if not kgram.PATH.exists():
        s = pctx.doc(n)
        return {"doc": n, "doc_id": s["doc_id"], "url": s["url"], "stats": {"skipped": "no cross-language lexicon"},
                "candidates": [], "verified": [], "flagged": False, "source": None, "crosslingual": True}
    ctx = _context_for(pctx)
    s = pctx.doc(n)
    excl = np.zeros(ctx.idx.N, dtype=bool)
    excl[n] = True
    rep = analyse_text(ctx, {**s, "published": pipeline.date_of(s)}, exclude=excl)
    rep["doc"] = n
    return rep


_ctx = {}


def _context_for(pctx):
    key = id(pctx.idx)
    if key not in _ctx:
        _ctx[key] = Context(idx=pctx.idx)
    return _ctx[key]


def script_of_doc(d):
    """The script a stored document was tagged with at crawl time, or detected from its text."""
    import json
    try:
        s = json.loads(d.get("meta") or "{}").get("script")
    except (TypeError, ValueError):
        s = None
    return s or tx.script_of(d.get("text") or "")
