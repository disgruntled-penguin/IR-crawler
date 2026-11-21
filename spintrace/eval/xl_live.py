"""SpinTrace-XL on the live crawl: every crawled Indian-language page against the English crawl.

Two English indexes cover the two crawls. Pages crawled before the main index was built are searched
in that index (crawled documents only). Pages from the restarted crawl are searched in a small index
of the English pages crawled in the same days (LIVE_DAYS before the newest page), built from a
read-only snapshot of the store; a full rebuild of all 35k English pages does not fit in memory next
to the generation queue on this machine. Nobody judges these pairs: the output is unjudged counts,
seeded-random examples, and whether each flagged suspect credits the outlet it was traced to.
"""
import csv
import json
import random
import time

import numpy as np

from .. import config, embed_xl, index, kgram, text as tx, text_indic as ti
from ..detect import crosslingual as xl, pipeline

INDEX = config.INDEX_DIR / "xl_live_index.pkl"
LEXICON = config.INDEX_DIR / "xl_live_kgram.pkl"
OUT = config.ROOT / "evaldata" / "live_xl.csv"
SUMMARY = config.RESULTS / "crosslingual_live.json"
LIVE_DAYS = 3
ALIASES = {
    "reuters": ["reuters", "रॉयटर्स", "रायटर्स", "रॉयटर", "রয়টার্স", "రాయిటర్స్", "ராய்ட்டர்ஸ்"],
    "pti": ["pti", "पीटीआई", "পিটিআই", "పీటీఐ"],
    "ani": ["ani", "एएनआई", "এএনআই", "ఏఎన్ఐ"],
    "ians": ["ians", "आईएएनएस", "আইএএনএস"],
    "afp": ["afp", "एएफपी", "এএফপি"],
    "ap": ["associated press", "एसोसिएटेड प्रेस"],
    "bbc": ["bbc", "बीबीसी", "বিবিসি", "బీబీసీ", "பிபிசி"],
    "cnn": ["cnn", "सीएनएन"],
    "ndtv": ["ndtv", "एनडीटीवी"],
    "hindustantimes": ["hindustan times", "हिंदुस्तान टाइम्स", "हिन्दुस्तान टाइम्स"],
    "indiatimes": ["times of india", "टाइम्स ऑफ इंडिया"],
    "thehindu": ["the hindu", "द हिंदू"],
    "indianexpress": ["indian express", "इंडियन एक्सप्रेस"],
    "aljazeera": ["al jazeera", "अल जज़ीरा", "अल जजीरा"],
    "theguardian": ["guardian", "गार्डियन"],
    "nytimes": ["new york times", "न्यूयॉर्क टाइम्स"],
}
AGENCIES = ("reuters", "pti", "ani", "ians", "afp", "ap")


def is_indic_doc(text, meta):
    try:
        s = json.loads(meta or "{}").get("script")
    except ValueError:
        s = None
    if s:
        return s != "latn", s
    if not any("ऀ" <= ch <= "ൿ" for ch in text[:400]):
        return False, "latn"
    s = tx.script_of(text)
    return s != "latn", s


def newest_fetch(con):
    return con.execute("SELECT MAX(fetched_at) FROM docs WHERE origin='crawl'").fetchone()[0]


def recent_english(con, since):
    for d in con.execute("SELECT doc_id, title, text, published, site, kind, origin, meta FROM docs "
                         "WHERE origin='crawl' AND fetched_at >= ? ORDER BY doc_id", (since,)):
        d = dict(d)
        if not is_indic_doc(d["text"], d["meta"])[0]:
            yield d


def build(con=None):
    """The live English index over the last LIVE_DAYS of crawling, and its name lexicon."""
    con = con or embed_xl.connect_ro()
    since = newest_fetch(con) - LIVE_DAYS * 86400
    idx = index.Index().build(recent_english(con, since))
    idx.live_since = since
    idx.save(INDEX)
    lex = kgram.build(idx, con, path=LEXICON)
    return idx, lex


def suspects(con):
    out = []
    for d in con.execute("SELECT doc_id, url, site, title, text, published, fetched_at, meta FROM docs "
                         "WHERE origin='crawl'"):
        ok, script = is_indic_doc(d["text"], d["meta"])
        if ok:
            out.append({**dict(d), "script": script, "published": pipeline.date_of(dict(d))})
    return out


def mentions(text, alias):
    t = ti.normalize(text).lower()
    if alias.isascii():
        return f" {alias} " in f" {' '.join(tx.tokens(t))} "
    return ti.normalize(alias) in t


def credits(text, site):
    """Which agencies the suspect names, and whether it names the outlet it was traced to."""
    named = sorted(a for a in AGENCIES if any(mentions(text, x) for x in ALIASES[a]))
    stem = site.split(".")[0].lower() if site else ""
    own = ALIASES.get(stem, [stem] if len(stem) > 3 else [])
    return named, bool(own) and any(mentions(text, x) for x in own)


def contexts(con):
    """(live context, its start time, main-index context with non-crawl documents masked out)."""
    if not INDEX.exists() or not LEXICON.exists():
        build(con)
    live_idx = index.Index.load(INDEX)
    live = xl.Context(idx=live_idx, lex=kgram.load(LEXICON), con=con)
    main = xl.Context(con=con, emb=live.emb)
    not_crawl = main.idx.meta["origin"] != "crawl"
    return live, live_idx.live_since, main, not_crawl


def run(seed=7, examples=5):
    con = embed_xl.connect_ro()
    live, since, main, not_crawl = contexts(con)
    rows = []
    sus = suspects(con)
    t0 = time.time()
    for i, s in enumerate(sus):
        recent = (s["fetched_at"] or 0) >= since
        ctx = live if recent else main
        excl = None if recent else not_crawl.copy()
        if not recent and s["doc_id"] in main.idx.doc_num:
            excl[main.idx.doc_num[s["doc_id"]]] = True
        rep = xl.analyse_text(ctx, s, exclude=excl)
        best = rep["source"] or (max(rep["verified"], key=lambda v: v["prob"]) if rep["verified"] else None)
        src = ctx.row(best["doc"]) if best else None
        agencies, own = credits(s["text"], src["site"] if src else "")
        lag = (s["published"] - src["published"]) / 3600 if src and s["published"] and src["published"] else None
        a = rep["stats"]["anchors"]
        rows.append({"suspect_url": s["url"], "suspect_site": s["site"], "script": s["script"],
                     "suspect_published": s["published"], "index": "live" if recent else "main",
                     "source_url": src["url"] if src else "", "source_site": src["site"] if src else "",
                     "prob": round(best["prob"], 4) if best else 0.0, "verdict": best["verdict"] if best else "",
                     "flagged": int(rep["flagged"]), "n_numbers": a["numbers"], "n_dates": a["dates"],
                     "n_latin": a["latin"], "n_names": a["names"], "postings_touched": rep["stats"]["postings_touched"],
                     "lag_hours": round(lag, 1) if lag is not None else "", "credits_source": int(own),
                     "agencies_named": " ".join(agencies)})
        if (i + 1) % 25 == 0:
            live.emb.save()
            print(f"live-xl {i + 1}/{len(sus)} flagged={sum(r['flagged'] for r in rows)} {time.time() - t0:.0f}s",
                  flush=True)
    live.emb.save()
    OUT.parent.mkdir(exist_ok=True)
    with OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["suspect_url"])
        w.writeheader()
        w.writerows(rows)
    summary = summarise(rows, {"live": live.idx.N, "main_crawl": int((~not_crawl).sum())})
    rng = random.Random(seed)
    flagged = [r for r in rows if r["flagged"]]
    summary["examples_flagged"] = rng.sample(flagged, min(examples, len(flagged)))
    summary["examples_any"] = rng.sample(rows, min(examples, len(rows)))
    SUMMARY.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def summarise(rows, sizes):
    flagged = [r for r in rows if r["flagged"]]
    by_script, by_site = {}, {}
    for r in rows:
        for table, key in ((by_script, r["script"]), (by_site, r["suspect_site"])):
            t = table.setdefault(key, {"suspects": 0, "flagged": 0})
            t["suspects"] += 1
            t["flagged"] += r["flagged"]
    lags = [r["lag_hours"] for r in flagged if r["lag_hours"] != ""]
    return {"english_index_docs": sizes, "suspects": len(rows), "flagged": len(flagged),
            "flagged_derived": sum(r["verdict"] == "derived" for r in flagged),
            "flagged_undated": sum(r["verdict"] == "derived_undated" for r in flagged),
            "flagged_crediting_source": sum(r["credits_source"] for r in flagged),
            "flagged_naming_agency": sum(bool(r["agencies_named"]) for r in flagged),
            "median_lag_hours": float(np.median(lags)) if lags else None,
            "by_script": by_script, "by_site": by_site, "judged": False}
