"""Cross-language evaluation: Indic rewrites of Wikinews originals traced back to the English index.

Items (from evaldata/rewrites_xl.jsonl, whatever rows exist):
  hi_translation, hi_seo, hi_summary   label 1, the true source is the Wikinews original
  hi_facts_only                        label 0, an independent Hindi article on the same event; the
                                       original is in the index and must not be called the source
  source_absent                        label 0, each Hindi derivative again with its source (and the
                                       source's English rewrites) masked out; nothing should be flagged
  ta/te/bn_translation                 label 1, a generality check on other scripts
Splits follow the midterm: by original, through run.split_of. Everything tuned (verifier weights,
thresholds, anchor weights) uses dev only; every number reported is test.

Methods: SpinTrace-XL; translate-then-detect (the LLM back-translations of test items, run through
the English baselines and English SpinTrace); multilingual dense cosine; Latin-tokens-only retrieval.
"""
import csv
import json
import os
import pickle
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from .. import config, dedup, embed, embed_xl, store
from ..detect import candidates, crosslingual as xl, pipeline, signals, verify
from . import report, xl_rewrites
from .run import TOP, ranked, split_of

ROWS = Path(os.environ.get("SPINTRACE_XL_ROWS", xl_rewrites.OUT))
RAW = config.DATA / "eval" / "xl_raw.pkl"
DEDUP = config.DATA / "eval" / "xl_dedup.pkl"
BT_EMB = config.INDEX_DIR / "xl_bt_emb.pkl"
OUT = config.RESULTS
HI_LEVELS = ["hi_translation", "hi_seo", "hi_summary"]
OTHER_LEVELS = ["ta_translation", "te_translation", "bn_translation"]
NEG_LEVELS = ["hi_facts_only", "source_absent"]
BT_BASELINES = ["exact_hash", "minhash_jaccard", "tfidf_cosine", "bm25"]
# Every method searches all earlier documents from other sites. The live system's 30-day window would
# shrink that pool from about 9,000 to about 300 here (the crawl is mostly Sep-Oct 2026), which makes the
# test easy, so the window is reported as its own variant instead.
WINDOW = None
VARIANTS = {
    "full": xl.Options(window_days=WINDOW),
    "no_names": xl.Options(names=False, window_days=WINDOW),
    "no_numbers": xl.Options(numbers=False, window_days=WINDOW),
    "no_dates": xl.Options(dates=False, window_days=WINDOW),
    "key_exact": xl.Options(fuzzy="key", window_days=WINDOW),
    "translit_exact": xl.Options(fuzzy="exact", window_days=WINDOW),
    "no_idf_check": xl.Options(idf_check=False, window_days=WINDOW),
    "window_30d": xl.Options(),
    "latin_only": xl.Options(names=False, numbers=False, dates=False, window_days=WINDOW),
}
BOOT = 1000


def load_rows():
    if not ROWS.exists():
        raise SystemExit(f"no cross-language rewrites at {ROWS}")
    return [json.loads(l) for l in ROWS.open() if l.strip()]


def synthetic_siblings(con, idx):
    """English synthetic rewrites in the index, by the original they were made from."""
    out = defaultdict(list)
    for doc_id, meta in con.execute("SELECT doc_id, meta FROM docs WHERE origin='synthetic'"):
        n = idx.doc_num.get(doc_id)
        if n is not None:
            out[json.loads(meta)["orig_doc"]].append(n)
    return out


def build_items(ctx, rows):
    idx = ctx.idx
    sib = synthetic_siblings(ctx.con, idx)
    by_rid = {r["rid"]: r for r in rows}
    items = []
    for r in rows:
        level = r["level"]
        if level.startswith("bt:") or level == "mash_llm" or not r.get("text"):
            continue
        src = store.doc_id_for(r["orig_url"])
        if src not in idx.doc_num:
            continue
        n = idx.doc_num[src]
        base = {"rid": r["rid"], "lang": r["lang"], "text": r["text"], "published": r["published"], "site": r["farm"],
                "split": split_of(src), "group": src, "exclude": sib.get(src, []), "orig": n}
        if level == "hi_facts_only":
            items.append({**base, "level": level, "label": 0, "source": None, "same_event": n})
            continue
        items.append({**base, "level": level, "label": 1, "source": n})
        if level in HI_LEVELS:
            items.append({**base, "level": "source_absent", "of": level, "label": 0, "source": None,
                          "exclude": base["exclude"] + [n]})
    bt = {}
    for r in rows:
        if r["level"].startswith("bt:") and r.get("src_rid") in by_rid and r.get("text"):
            bt[r["src_rid"]] = r
    for it in items:
        b = bt.get(it["rid"])
        if b is not None:
            it["bt"] = {"rid": b["rid"], "text": b["text"], "seconds": b.get("seconds"),
                        "src_seconds": by_rid[it["rid"]].get("seconds")}
    return items


def exclude_mask(idx, it):
    m = np.zeros(idx.N, dtype=bool)
    if it["exclude"]:
        m[it["exclude"]] = True
    return m


def suspect(it):
    return {"doc_id": "xl:" + it["rid"], "text": it["text"], "published": it["published"], "site": it["site"],
            "url": f"https://{it['site']}/article/{it['rid']}"}


class Dense:
    """Multilingual dense cosine over every document's head vector (the one corpus-wide embedding)."""

    def __init__(self, ctx):
        heads = embed_xl.load_dense()
        dim = len(next(iter(heads.values())))
        self.mat = np.zeros((ctx.idx.N, dim), dtype=np.float32)
        self.have = np.zeros(ctx.idx.N, dtype=bool)
        for n, d in enumerate(ctx.idx.doc_ids):
            v = heads.get(d)
            if v is not None:
                self.mat[n] = v
                self.have[n] = True
        if self.have.mean() < 0.99:
            raise SystemExit(f"dense heads cover {self.have.mean():.0%} of the index; finish `build-xl --dense` first")
        self.emb = ctx.emb

    def run(self, text, mask):
        t = time.perf_counter()
        q = self.emb.encode([embed_xl.head(text)])[0].astype(np.float32)
        r = ranked(self.mat @ q, mask & self.have)
        return r, time.perf_counter() - t


class BackTranslation:
    """The translate-then-detect pipeline: an LLM translates the suspect back to English, then the
    English methods run on it unchanged, with the thresholds they were given on the English dev split."""

    def __init__(self, ctx):
        self.ctx = ctx
        idx = ctx.idx
        d = load_dedup(idx)
        self.exact = defaultdict(list)
        for n, h in enumerate(d["exact"]):
            self.exact[h].append(n)
        self.sigs = d["minhash"]
        self.lsh = dedup.LSH()
        for n in range(idx.N):
            self.lsh.add(n, self.sigs[n])
        self.emb = _Embedder()
        self.model = verify.load()

    def run(self, text, published, site, mask):
        idx, out = self.ctx.idx, {}
        t = time.perf_counter()
        hits = [m for m in self.exact[dedup.exact_hash(text)] if mask[m]]
        out["exact_hash"] = ([(m, 1.0) for m in hits], time.perf_counter() - t)
        t = time.perf_counter()
        sh = dedup.shingles(text)
        sig = dedup.minhash(sh)
        cands = [m for m in self.lsh.query(sig) if mask[m]]
        r = sorted(((m, dedup.est_jaccard(sig, self.sigs[m])) for m in cands), key=lambda x: -x[1])
        out["minhash_jaccard"] = (r[:TOP], time.perf_counter() - t)
        terms = candidates.text_terms(idx, text)
        t = time.perf_counter()
        out["tfidf_cosine"] = (ranked(idx.cosine_scores(idx.query_vector(terms)), mask), time.perf_counter() - t)
        t = time.perf_counter()
        acc = idx.bm25_scores(terms)
        out["bm25"] = (ranked(acc / (bm25_self(idx, terms, len(idx.tokens(text))) or 1.0), mask),
                       time.perf_counter() - t)
        t = time.perf_counter()
        cands, _ = candidates.retrieve_text(idx, text, published, site, k=TOP, exclude=~mask)
        s = {"doc_id": "bt:" + str(hash(text)), "text": text, "published": published}
        feats = {s["doc_id"]: {"shingles": sh}}
        rows = []
        top = cands[:pipeline.VERIFY_TOP]
        self.emb.prefetch([(s["doc_id"], text)] + [(self.ctx.row(c["doc"])["doc_id"], self.ctx.row(c["doc"])["text"])
                                                   for c in top])
        for c in top:
            cd = self.ctx.row(c["doc"])
            feats.setdefault(cd["doc_id"], {"shingles": dedup.shingles(cd["text"])})
            pair = signals.Pair(s, cd, feats, self.emb, c)
            verdict, _ = pipeline.provenance(s, cd, pair)
            rows.append({"doc": c["doc"], "signals": signals.compute(pair), "verdict": verdict,
                         "published": pipeline.date_of(cd)})
        out["spintrace_en"] = (rows, [(c["doc"], c["score"]) for c in cands], time.perf_counter() - t)
        return out


class _Embedder(embed.Embedder):
    """The English verifier's embedder with its own small cache, so the midterm cache is not loaded."""

    def __init__(self):
        self.model = None
        self.cache = {}
        self.dirty = False
        if BT_EMB.exists():
            with BT_EMB.open("rb") as f:
                self.cache = pickle.load(f)

    def save(self):
        if self.dirty:
            with BT_EMB.open("wb") as f:
                pickle.dump(self.cache, f, protocol=pickle.HIGHEST_PROTOCOL)
            self.dirty = False


def bm25_self(idx, terms, length):
    """BM25 of a text against itself, so its scores are on the same 0-1 scale as run.Baselines' bm25."""
    s = 0.0
    for t, tf in terms.items():
        p = idx.post["body"].get(t)
        df = len(p.docs) if p is not None else 0
        idf = np.log(1 + (idx.N - df + 0.5) / (df + 0.5))
        s += idf * tf * (1.2 + 1) / (tf + 1.2 * (1 - 0.75 + 0.75 * length / idx.avg_len))
    return s


def load_dedup(idx):
    """Exact hashes and MinHash signatures of every indexed document, taken from the midterm features once."""
    if DEDUP.exists():
        with DEDUP.open("rb") as f:
            d = pickle.load(f)
        if d["doc_ids"] == idx.doc_ids:
            return d
    from .. import corpus
    feats = corpus.load_features()
    d = {"doc_ids": list(idx.doc_ids), "exact": [feats[x]["exact"] for x in idx.doc_ids],
         "minhash": np.stack([feats[x]["minhash"] for x in idx.doc_ids])}
    corpus._feats = None
    DEDUP.parent.mkdir(parents=True, exist_ok=True)
    with DEDUP.open("wb") as f:
        pickle.dump(d, f, protocol=pickle.HIGHEST_PROTOCOL)
    return d


def dense_candidates(ranking):
    return [{"doc": d, "cos": 0.0, "date_frac": 0.0, "score": s} for d, s in ranking]


def run(variants=tuple(VARIANTS), log_every=20, raw=RAW, limit=None, fresh=False, use_dense=True):
    """Score every item with every method and cache it; items already cached are reused unless fresh."""
    ctx = xl.Context()
    idx = ctx.idx
    items = build_items(ctx, load_rows())
    if limit:
        items = random.Random(358).sample(items, min(limit, len(items)))
    cached = {}
    if raw.exists() and not fresh:
        with raw.open("rb") as f:
            for it in pickle.load(f)["items"]:
                cached[(it["rid"], it["level"])] = it
    dense = Dense(ctx) if use_dense and embed_xl.DENSE.exists() else None
    bt = None
    t0 = time.time()
    for i, it in enumerate(items):
        old = cached.get((it["rid"], it["level"]))
        if (old is not None and all(v in old.get("runs", {}) for v in variants) and ("bt" not in it or "bt_out" in old)
                and (dense is None or "dense" in old)):
            items[i] = old
            continue
        s = suspect(it)
        mask = xl.allowed_mask(idx, ctx.lex, it["published"], it["site"], exclude_mask(idx, it), WINDOW)
        it["allowed"] = int(mask.sum())
        it["runs"] = {}
        full_anchors = None
        for name in variants:
            opts = VARIANTS[name]
            t = time.perf_counter()
            cands, stats, anchors = xl.retrieve(idx, ctx.lex, it["text"], it["published"], it["site"],
                                                exclude_mask(idx, it), opts=opts)
            t_ret = time.perf_counter() - t
            run_ = {"cands": [(c["doc"], c["score"]) for c in cands], "t_retrieve": t_ret,
                    "touched": stats["postings_touched"], "anchors": stats["anchors"]}
            if name != "latin_only":
                t = time.perf_counter()
                run_["verified"] = xl.verify(ctx, s, cands, anchors)
                run_["t_verify"] = time.perf_counter() - t
            it["runs"][name] = run_
            if name == "full":
                full_anchors = anchors
        if dense is not None:
            full_anchors = full_anchors or xl.Anchors(it["text"], ctx.lex, idx)
            r, secs = dense.run(it["text"], mask)
            it["dense"] = (r, secs)
            t = time.perf_counter()
            it["runs"]["dense_verify"] = {"cands": r, "verified": xl.verify(ctx, s, dense_candidates(r), full_anchors),
                                          "t_verify": time.perf_counter() - t}
        if "bt" in it:
            bt = bt or BackTranslation(ctx)
            it["bt_out"] = bt.run(it["bt"]["text"], it["published"], it["site"], mask)
        if (i + 1) % log_every == 0:
            ctx.emb.save()
            if bt:
                bt.emb.save()
            print(f"eval-xl {i + 1}/{len(items)} {time.time() - t0:.0f}s", flush=True)
    ctx.emb.save()
    if bt:
        bt.emb.save()
    for it in items:
        for run_ in it["runs"].values():
            for v in run_.get("verified") or []:
                v.pop("why", None)
    raw.parent.mkdir(parents=True, exist_ok=True)
    with raw.open("wb") as f:
        pickle.dump({"items": items, "N": idx.N, "built": time.time()}, f)
    return items


def load(raw=RAW):
    with raw.open("rb") as f:
        return pickle.load(f)["items"]


def boot_ci(pred, labels, stat="f1", B=BOOT, seed=358):
    """95% percentile bootstrap interval of precision, recall or F1 over resampled items."""
    pred, labels = np.asarray(pred, dtype=bool), np.asarray(labels)
    if not len(pred):
        return float("nan"), float("nan")
    rng = np.random.RandomState(seed)
    k = {"precision": 0, "recall": 1, "f1": 2}[stat]
    vals = []
    for _ in range(B):
        ix = rng.randint(0, len(pred), len(pred))
        vals.append(report.prf(pred[ix], labels[ix])[k])
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def boot_mean(x, B=BOOT, seed=358):
    x = np.asarray(x, dtype=float)
    if not len(x):
        return float("nan"), float("nan")
    rng = np.random.RandomState(seed)
    vals = [x[rng.randint(0, len(x), len(x))].mean() for _ in range(B)]
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def set_keys(items):
    """Expose each variant's verified list as it['v_<variant>'], the shape report.fit_model expects."""
    for it in items:
        for name, r in it["runs"].items():
            if "verified" in r:
                it["v_" + name] = r["verified"]
        if "bt_out" in it:
            it["v_bt"] = it["bt_out"]["spintrace_en"][0]


class Method:
    """score(it) -> (score, predicted source); rank(it) -> ranked doc list; applies(it) -> bool."""

    def __init__(self, name, score, rank, applies=lambda it: True, threshold=None):
        self.name, self.score, self.rank, self.applies, self.threshold = name, score, rank, applies, threshold


def _verified_rank(it, key, model):
    vs = it.get(key) or []
    order = [v["doc"] for v in sorted(vs, key=lambda v: -verify.score(v["signals"], model))]
    return order + [d for d, _ in it["runs"].get(key[2:], {}).get("cands", []) if d not in order]


def _top(lst):
    return (lst[0][1], lst[0][0]) if lst else (0.0, None)


def english_thresholds():
    """The English methods' thresholds, fit on the English dev split in the midterm evaluation."""
    out = {}
    path = config.RESULTS / "detection_by_level.csv"
    if path.exists():
        for r in csv.DictReader(path.open()):
            out.setdefault(r["method"], float(r["threshold"]))
    return out


def xl_method(name, variant, model):
    key = "v_" + variant
    return Method(name, lambda it: report.spin_score(it, model, key), lambda it: _verified_rank(it, key, model),
                  lambda it: key in it)


def methods(models):
    ms = [xl_method("spintrace_xl", "full", models["full"])]
    ms.append(Method("dense_cosine", lambda it: _top(it["dense"][0]), lambda it: [d for d, _ in it["dense"][0]],
                     lambda it: "dense" in it))
    ms.append(Method("latin_only", lambda it: _top(it["runs"]["latin_only"]["cands"]),
                     lambda it: [d for d, _ in it["runs"]["latin_only"]["cands"]], lambda it: "latin_only" in it["runs"]))
    eng = english_thresholds()
    en_model = verify.load()

    def has_bt(it):
        return "bt_out" in it

    def bt_rank(it):
        vs = sorted(it["v_bt"], key=lambda v: -verify.score(v["signals"], en_model))
        order = [v["doc"] for v in vs]
        return order + [d for d, _ in it["bt_out"]["spintrace_en"][1] if d not in order]

    ms.append(Method("bt_spintrace_en", lambda it: report.spin_score(it, en_model, "v_bt"), bt_rank, has_bt,
                     en_model["threshold"]))
    for b in BT_BASELINES:
        ms.append(Method("bt_" + b, lambda it, b=b: _top(it["bt_out"][b][0]),
                         lambda it, b=b: [d for d, _ in it["bt_out"][b][0]], has_bt, eng.get(b)))
    return ms


def fit_models(dev, variants):
    out = {}
    for v in variants:
        items = [it for it in dev if ("v_" + v) in it and it["level"] in HI_LEVELS + NEG_LEVELS]
        out[v] = report.fit_model(items, xl.FEATURES, key="v_" + v)
    return out


def threshold_for(m, dev):
    if m.threshold is not None:
        return m.threshold
    group = [it for it in dev if m.applies(it) and it["level"] in HI_LEVELS + NEG_LEVELS]
    return report.best_threshold([m.score(it)[0] for it in group], [it["label"] for it in group])


def detection_rows(m, thr, test, levels):
    scored = [(it, *m.score(it)) for it in test if m.applies(it)]
    negs = [x for x in scored if x[0]["label"] == 0]
    rows = []
    for level in levels:
        pos = [x for x in scored if x[0]["level"] == level]
        if not pos:
            continue
        group = pos + negs
        pred = [s >= thr for _, s, _ in group]
        lab = [it["label"] for it, _, _ in group]
        p, r, f = report.prf(pred, lab)
        lo, hi = boot_ci(pred, lab)
        rows.append({"method": m.name, "level": level, "n_pos": len(pos), "n_neg": len(negs), "threshold": thr,
                     "precision": p, "recall": r, "f1": f, "f1_lo": lo, "f1_hi": hi,
                     "source_correct": float(np.mean([s >= thr and src == it["source"] for it, s, src in pos]))})
    for level in NEG_LEVELS:
        ns = [s >= thr for it, s, _ in negs if it["level"] == level]
        if ns:
            lo, hi = boot_mean(ns)
            rows.append({"method": m.name, "level": "fpr_" + level, "n_pos": 0, "n_neg": len(ns), "threshold": thr,
                         "precision": None, "recall": None, "f1": float(np.mean(ns)), "f1_lo": lo, "f1_hi": hi,
                         "source_correct": None})
    return rows


def ranking_rows(m, test, levels, ks=(1, 5, 20)):
    rows = []
    pos = [it for it in test if it["label"] == 1 and m.applies(it)]
    for level in levels + ["all_hi"]:
        group = [it for it in pos if it["level"] == level or (level == "all_hi" and it["level"] in HI_LEVELS)]
        if not group:
            continue
        ranks = []
        for it in group:
            lst = m.rank(it)
            ranks.append(lst.index(it["source"]) + 1 if it["source"] in lst else None)
        p1 = [bool(r == 1) for r in ranks]
        lo, hi = boot_mean(p1)
        row = {"method": m.name, "level": level, "n": len(group), "p@1": float(np.mean(p1)), "p@1_lo": lo,
               "p@1_hi": hi, "mrr": float(np.mean([1 / r if r else 0 for r in ranks]))}
        for k in ks[1:]:
            row[f"recall@{k}"] = float(np.mean([bool(r and r <= k) for r in ranks]))
        rows.append(row)
    return rows


def efficiency_rows(items):
    rows = []
    full = [it["runs"]["full"] for it in items]
    rows.append({"method": "spintrace_xl",
                 "ms_per_suspect": 1000 * np.mean([r["t_retrieve"] + r["t_verify"] for r in full]),
                 "ms_retrieve": 1000 * np.mean([r["t_retrieve"] for r in full]),
                 "postings_touched": float(np.mean([r["touched"] for r in full])),
                 "docs_verified": float(xl.VERIFY_TOP), "llm_seconds": 0.0})
    lat = [it["runs"]["latin_only"] for it in items if "latin_only" in it["runs"]]
    if lat:
        rows.append({"method": "latin_only", "ms_per_suspect": 1000 * np.mean([r["t_retrieve"] for r in lat]),
                     "ms_retrieve": 1000 * np.mean([r["t_retrieve"] for r in lat]),
                     "postings_touched": float(np.mean([r["touched"] for r in lat])), "llm_seconds": 0.0})
    dn = [it for it in items if "dense" in it]
    if dn:
        rows.append({"method": "dense_cosine", "ms_per_suspect": 1000 * np.mean([it["dense"][1] for it in dn]),
                     "docs_compared": float(np.mean([it["allowed"] for it in dn])), "llm_seconds": 0.0})
    bt = [it for it in items if "bt_out" in it]
    if bt:
        llm = float(np.mean([it["bt"]["seconds"] or 0.0 for it in bt]))
        for b in BT_BASELINES:
            det = 1000 * np.mean([it["bt_out"][b][1] for it in bt])
            rows.append({"method": "bt_" + b, "ms_per_suspect": det + 1000 * llm, "ms_detect_only": det,
                         "llm_seconds": llm})
        det = 1000 * np.mean([it["bt_out"]["spintrace_en"][2] for it in bt])
        rows.append({"method": "bt_spintrace_en", "ms_per_suspect": det + 1000 * llm, "ms_detect_only": det,
                     "llm_seconds": llm})
    return rows


def macro(rows, levels=HI_LEVELS):
    fs = [r["f1"] for r in rows if r["level"] in levels]
    return float(np.mean(fs)) if fs else float("nan")


def ablation_rows(dev, test, models):
    """One change at a time: each retrieval variant with its own verifier refit on dev, then the full
    retrieval with one group of verifier signals removed."""
    out = []

    def add(name, m, variant):
        thr = threshold_for(m, dev)
        det = detection_rows(m, thr, test, HI_LEVELS)
        rank = [r for r in ranking_rows(m, test, HI_LEVELS) if r["level"] == "all_hi"]
        f = {r["level"]: r["f1"] for r in det}
        touched = [it["runs"][variant]["touched"] for it in test if "touched" in it["runs"].get(variant, {})]
        out.append({"variant": name, "macro_f1": macro(det), "p@1": rank[0]["p@1"] if rank else float("nan"),
                    "recall@5": rank[0].get("recall@5") if rank else float("nan"),
                    "fpr_hi_facts_only": f.get("fpr_hi_facts_only"), "fpr_source_absent": f.get("fpr_source_absent"),
                    "postings_touched": float(np.mean(touched)) if touched else None})

    for v in models:
        add(v, xl_method(v, v, models[v]), v)
    groups = {"no_alignment_signals": [f for f in xl.FEATURES if f.startswith(("align", "ordered", "back"))],
              "no_anchor_signals": ["number_overlap", "number_back", "name_overlap", "name_back", "date_overlap",
                                    "latin_overlap"],
              "no_retrieval_score": ["xl_cos"]}
    dv = [it for it in dev if it["level"] in HI_LEVELS + NEG_LEVELS and "v_full" in it]
    for name, drop in groups.items():
        model = report.fit_model(dv, [f for f in xl.FEATURES if f not in drop], key="v_full")
        add(name, xl_method(name, "full", model), "full")
    return out


LABELS = {"spintrace_xl": "SpinTrace-XL", "dense_cosine": "Multilingual dense cosine",
          "latin_only": "Latin tokens only", "bt_spintrace_en": "Back-translate + SpinTrace (EN)",
          "bt_tfidf_cosine": "Back-translate + tf-idf cosine", "bt_bm25": "Back-translate + BM25",
          "bt_minhash_jaccard": "Back-translate + MinHash Jaccard", "bt_exact_hash": "Back-translate + exact hash"}
COLORS = {"spintrace_xl": "#2a78d6", "dense_cosine": "#4a3aa7", "latin_only": "#52514e", "bt_spintrace_en": "#1baf7a",
          "bt_bm25": "#e87ba4", "bt_tfidf_cosine": "#eda100", "bt_minhash_jaccard": "#008300", "bt_exact_hash": "#eb6834"}


def charts(det, rank, levels):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for rows, key, path, title, ylabel in (
            (det, "f1", OUT / "crosslingual_f1_by_level.png", "Cross-language detection F1 by level (test split)", "F1"),
            (rank, "p@1", OUT / "crosslingual_p_at_1_by_level.png", "Original ranked first by level (test split)",
             "P@1")):
        fig, ax = plt.subplots(figsize=(10, 4.8), dpi=150)
        for m in LABELS:
            ys = [next((r[key] for r in rows if r["method"] == m and r["level"] == l), np.nan) for l in levels]
            if all(np.isnan(y) for y in ys):
                continue
            ax.plot(levels, ys, marker="o", markersize=7, linewidth=2.6 if m == "spintrace_xl" else 1.8,
                    color=COLORS[m], zorder=5 if m == "spintrace_xl" else 2, label=LABELS[m])
        ax.set_ylim(-0.02, 1.02)
        report._style(ax, title, "suspect level", ylabel)
        ax.legend(frameon=False, fontsize=8, loc="center left", bbox_to_anchor=(1.01, 0.5))
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)


def main(raw=RAW):
    items = load(raw)
    set_keys(items)
    dev = [it for it in items if it["split"] == "dev"]
    test = [it for it in items if it["split"] == "test"]
    variants = [v for v in list(VARIANTS) + ["dense_verify"] if any("v_" + v in it for it in dev)]
    models = fit_models(dev, variants)
    xl.save_model(models["full"])
    levels = HI_LEVELS + OTHER_LEVELS
    det, rank = [], []
    for m in methods(models):
        if not any(m.applies(it) for it in test):
            continue
        thr = threshold_for(m, dev)
        det += detection_rows(m, thr, test, levels)
        rank += ranking_rows(m, test, levels)
        if m.name.startswith("bt_"):
            # No Hindi dev item was back-translated, so these use the English dev thresholds; the threshold
            # that is best on test itself is an upper bound on what any tuning could give them.
            group = [it for it in test if m.applies(it) and it["level"] in HI_LEVELS + NEG_LEVELS]
            oracle = report.best_threshold([m.score(it)[0] for it in group], [it["label"] for it in group])
            det += [{**r, "method": m.name + "@oracle"} for r in detection_rows(m, oracle, test, levels)]
    abl = ablation_rows(dev, test, models)
    eff = efficiency_rows(test)
    OUT.mkdir(exist_ok=True)
    report.write_csv(OUT / "crosslingual_detection.csv", det)
    report.write_csv(OUT / "crosslingual_ranking.csv", rank)
    report.write_csv(OUT / "crosslingual_ablations.csv", abl)
    report.write_csv(OUT / "crosslingual_efficiency.csv", eff)
    counts = defaultdict(lambda: defaultdict(int))
    for it in items:
        counts[it["split"]][it["level"]] += 1
    counts = {k: dict(v) for k, v in counts.items()}
    (OUT / "crosslingual_sets.json").write_text(json.dumps(counts, indent=2))
    charts(det, rank, HI_LEVELS)
    return det, rank, abl, eff, counts, models["full"]
