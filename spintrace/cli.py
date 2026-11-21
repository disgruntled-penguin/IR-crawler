"""SpinTrace command line: python -m spintrace <command>."""
import argparse
import datetime as dt
import json
import math
import textwrap

from . import config


def _when(t):
    if t is None or (isinstance(t, float) and math.isnan(t)):
        return "undated"
    return dt.datetime.fromtimestamp(t, dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _resolve(con, key):
    row = con.execute("SELECT doc_id FROM docs WHERE doc_id=? OR url=?", (key, key)).fetchone()
    if not row:
        raise SystemExit(f"no document matches {key!r}")
    return row[0]


def cmd_crawl(args):
    from .crawl.crawler import Crawler, setup_logging
    setup_logging(config.LOG_DIR / "crawl.log")
    Crawler(args.seeds, max_pages=args.max_pages).run(hours=args.hours)


def cmd_export(args):
    """Write crawled URLs and per-document features (no article text) for the repository."""
    import csv
    import numpy as np
    from . import corpus, store
    con = store.connect()
    feats = corpus.load_features()
    rows = [r for r in con.execute("SELECT doc_id, url, site, kind, published, date_source, n_words, content_hash "
                                   "FROM docs WHERE origin IN ('crawl','newsguard') ORDER BY doc_id")]
    path = config.ROOT / "evaldata" / "crawl_urls.csv"
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["doc_id", "url", "site", "kind", "published", "date_source", "n_words", "sha1_text"])
        w.writerows([tuple(r) for r in rows])
    ids = [r["doc_id"] for r in rows if r["doc_id"] in feats]
    np.savez_compressed(config.ROOT / "evaldata" / "crawl_minhash.npz", doc_ids=np.array(ids),
                        minhash=np.stack([feats[d]["minhash"] for d in ids]).astype(np.uint32))
    print(f"exported {len(rows)} URLs and {len(ids)} MinHash signatures to evaldata/")


def cmd_recrawl(args):
    """Re-fetch the exported article URLs politely, to rebuild the corpus from a clean clone."""
    import csv
    from .crawl.crawler import Crawler, setup_logging
    setup_logging(config.LOG_DIR / "recrawl.log")
    rows = list(csv.DictReader(open(args.urls)))
    urls = [(r["url"], r["kind"]) for r in rows if not r["url"].startswith("https://web.archive.org")]
    if args.limit:
        urls = urls[:args.limit]
    Crawler(args.seeds, url_list=urls).run()


def cmd_newsguard(args):
    """Fetch both sides of each NewsGuard-documented pair from the Wayback Machine into the store."""
    from .eval import realpairs
    for pair_id, status in realpairs.load():
        print(f"{pair_id}: {status}")


def cmd_ccnews(args):
    from .eval import ccnews
    kept, seen = ccnews.load(args.max)
    print(f"stored {kept} English articles from {seen} HTML records in {ccnews.WARC.name}")


def cmd_scale(args):
    """Re-run the test items against an index that also holds the CC-NEWS documents (no refitting)."""
    from .eval import scale
    for r in scale.run():
        print("  ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in r.items()))
    print(f"written to {scale.OUT}")


def cmd_build(args):
    """Ingest evaluation documents, compute dedup features and build the index(es)."""
    from . import corpus, index, store
    con = store.connect()
    if not args.no_eval:
        print("ingested wikinews=%d synthetic=%d" % corpus.ingest_eval(con))
    feats, secs = corpus.build_features(con)
    print(f"features for {len(feats)} docs in {secs:.1f}s")
    only = set(feats)
    if not args.with_ccnews:
        # CC-NEWS documents are for the scale experiment only (`python -m spintrace scale`).
        only -= {r[0] for r in con.execute("SELECT doc_id FROM docs WHERE origin='ccnews'")}
    for stem in ([False, True] if args.stem else [False]):
        idx = index.build_from_store(stemming=stem, con=con, only=only)
        print(f"index stemming={stem}: {idx.N} docs, {len(idx.terms)} terms, built in {idx.build_seconds:.1f}s")


def cmd_postings(args):
    from . import index
    idx = index.load(args.stem)
    term = idx.tokens(args.term)[0]
    t = idx.vocab.get(term)
    if t is None:
        raise SystemExit(f"term {term!r} not in dictionary")
    print(f"term={term!r} df={idx.df[t]} N={idx.N} idf=log10(N/df)={idx.idf[t]:.3f}")
    for zone in ("title", "body", "quote"):
        p = idx.post[zone].get(t)
        if p is None:
            continue
        print(f"zone={zone} postings={len(p.docs)}")
        for i in range(min(args.limit, len(p.docs))):
            d = int(p.docs[i])
            print(f"  doc#{d} {idx.doc_ids[d]} {idx.meta['site'][d]:<22} tf={p.tfs[i]} positions={p.positions(i)[:12].tolist()}")


def cmd_pair(args):
    from . import corpus, dedup, store
    con = store.connect()
    a, b = _resolve(con, args.a), _resolve(con, args.b)
    f = corpus.load_features()
    sa, sb = f[a]["shingles"], f[b]["shingles"]
    print(f"A={a} shingles={len(sa)}   B={b} shingles={len(sb)}   k={dedup.SHINGLE_K}")
    print(f"exact hash equal: {f[a]['exact'] == f[b]['exact']}")
    print(f"|A∩B|={len(sa & sb)} |A∪B|={len(sa | sb)} Jaccard={dedup.jaccard(sa, sb):.3f}")
    print(f"containment A in B={dedup.containment(sa, sb):.3f}  B in A={dedup.containment(sb, sa):.3f}")
    ma, mb = f[a]["minhash"], f[b]["minhash"]
    bands = sum(bool((ma[i * dedup.ROWS:(i + 1) * dedup.ROWS] == mb[i * dedup.ROWS:(i + 1) * dedup.ROWS]).all())
                for i in range(dedup.BANDS))
    print(f"MinHash estimate ({dedup.N_PERM} perms)={dedup.est_jaccard(ma, mb):.3f}  LSH bands agreeing={bands}/{dedup.BANDS}"
          f" -> {'candidate pair' if bands else 'not a candidate'}")


def cmd_suspect(args):
    from .detect import pipeline, signals
    ctx = pipeline.Context(stemming=args.stem)
    doc_id = _resolve(ctx.con, args.doc)
    n = ctx.num(doc_id)
    s = ctx.doc(n)
    print(f"SUSPECT {doc_id} {s['site']} published={_when(s['published'])}\n  {s['url']}\n  {s['title']}")
    rep = pipeline.analyse(ctx, n)
    st = rep["stats"]
    print(f"\n1. QUERY: {len(st['query_terms'])} rarest terms (index elimination), quotes={st['n_quotes']}, "
          f"earlier docs allowed={st['allowed']}/{ctx.idx.N}, postings touched={st['postings_touched']}")
    print("   " + ", ".join(f"{t}({w})" for t, w in st["query_terms"]))
    print("\n2. CANDIDATES (rare-term cosine + quote phrase hits)")
    for i, c in enumerate(rep["candidates"][:10]):
        d = ctx.doc(c["doc"])
        print(f"   {i + 1:>2}. score={c['score']:.3f} cos={c['rare_cos']:.3f} quotes={c['quote_frac']:.2f} "
              f"{d['site']:<22} {_when(d['published'])} {(d['title'] or d['url'])[:60]}")
    print("\n3. VERIFICATION of the top candidates")
    for v in rep["verified"]:
        sig = " ".join(f"{k}={x:.2f}" for k, x in v["signals"].items())
        print(f"   p={v['prob']:.3f} {v['site']:<22} {sig}\n      provenance: {v['verdict']} ({v['why']})")
    best = rep["verified"][0] if rep["verified"] else None
    if best:
        c = ctx.doc(best["doc"])
        pair = signals.Pair(s, c, ctx.feats, ctx.emb)
        b, arg, _ = pair.alignment()
        ss, _ = ctx.emb.doc(s["doc_id"], s["text"])
        cs, _ = ctx.emb.doc(c["doc_id"], c["text"])
        print(f"\n4. SENTENCE ALIGNMENT with {c['site']} (threshold {signals.ALIGN_THRESHOLD})")
        for i in range(min(args.sentences, len(ss))):
            mark = "==" if b[i] >= signals.ALIGN_THRESHOLD else "  "
            print(f"   {mark} s{i} -> c{arg[i]} sim={b[i]:.2f}\n      S: {textwrap.shorten(ss[i], 110)}\n"
                  f"      C: {textwrap.shorten(cs[arg[i]], 110)}")
    verdict = (f"DERIVED from {rep['source']['url']} (p={rep['source']['prob']:.3f}, {rep['source']['verdict']})"
               if rep["flagged"] else "NOT FLAGGED as derivative")
    print(f"\n5. VERDICT: {verdict}  [threshold {ctx.model['threshold']:.3f}, weights from {ctx.model['source']}]")
    ctx.emb.save()


def cmd_sources(args):
    """Every source of one suspect, round by round: residual query, candidates, verification, coverage, shares."""
    import numpy as np
    from .detect import multisource
    from .eval import ms_eval
    if not args.doc and not args.file:
        raise SystemExit("give a doc_id, URL or mashup id, or --file")
    ctx = ms_eval.context()
    if args.file:
        import hashlib
        text = open(args.file, encoding="utf-8").read()
        sus = multisource.Suspect("text:" + hashlib.sha1(text.encode()).hexdigest()[:16], text)
        gold, excl, note = None, None, f"  [text from {args.file}, not in the index; no date filter]"
    else:
        sus, gold, excl, note = ms_eval.suspect_for_key(ctx, args.doc, args.xl)
    model = ctx.ms_model
    print(f"SUSPECT {sus.doc_id} {sus.site} published={_when(sus.published)}{note}")
    if sus.url:
        print(f"  {sus.url}\n  {sus.title}")
    for g, share in (gold or {}).items():
        d = ctx.doc(g)
        print(f"  true source: {d['doc_id']} {d['site']} share={share if share is None else round(share, 2)} "
              f"{(d['title'] or d['url'])[:70]}")
    tr = multisource.trace(ctx, sus, exclude=excl)
    sents = tr["sents"]
    print(f"  {tr['n_sents']} sentences; round verifier from {model['source']}: threshold={tr['params']['threshold']:.2f} "
          f"stop below {tr['params']['stop_share']:.2f} unexplained, cover at sim >= {tr['params']['cover']:.2f}")
    covered = np.zeros(tr["n_sents"], dtype=bool)
    for rd in tr["rounds"]:
        print(f"\nROUND {rd['round']}: unexplained {rd['residual_sents']}/{tr['n_sents']} sentences ({rd['share_left']:.0%})")
        if covered.any():
            print(f"   covered so far: s{', s'.join(str(i) for i in np.nonzero(covered)[0])}")
        print(f"   residual query: {len(rd['query_terms'])} rarest terms, quotes={rd['n_quotes']}, "
              f"postings touched={rd['postings_touched']}")
        print("   " + ", ".join(f"{t}({w})" for t, w in rd["query_terms"]))
        print("   candidates (rare-term cosine + quote phrase hits):")
        for i, c in enumerate(rd["candidates"][:args.k]):
            d = ctx.doc(c["doc"])
            mark = " <- true source" if gold and c["doc"] in gold else ""
            print(f"   {i + 1:>2}. score={c['score']:.3f} cos={c['rare_cos']:.3f} quotes={c['quote_frac']:.2f} "
                  f"{d['site']:<20} {_when(d['published'])} {(d['title'] or d['url'])[:50]}{mark}")
        print("   verification against the unexplained sentences:")
        for v in rd["verified"]:
            sig = " ".join(f"{k.replace('res_', '')}={x:.2f}" for k, x in v["signals"].items())
            print(f"     p={v['prob']:.3f} {v['site']:<20} {sig}")
        c = rd["chosen"]
        ok = rd["prob"] >= tr["params"]["threshold"] and len(rd["new_covered"]) >= multisource.MIN_NEW
        print(f"   {'CHOSEN' if ok else 'REJECTED'}: {c['doc_id']} {c['site']} p={c['prob']:.3f}, "
              f"covers {len(rd['new_covered'])} unexplained sentences")
        cs, _ = ctx.emb.doc(c["doc_id"], ctx.doc(c["doc"])["text"])
        for i in rd["new_covered"][:args.sentences]:
            print(f"      == s{i} -> c{rd['arg'][i]} sim={rd['sims'][i]:.2f}\n         S: {textwrap.shorten(sents[i], 100)}"
                  f"\n         C: {textwrap.shorten(cs[rd['arg'][i]], 100)}")
        if ok:
            covered[rd["new_covered"]] = True
    print(f"\nSTOP: {tr['stop']}")
    fin = multisource.finalize(tr)
    print("FINAL SHARES (each sentence goes to the found source it aligns to best):")
    for s in fin["sources"]:
        mark = "" if not gold else (" (true source)" if s["doc"] in gold else " (not a true source)")
        print(f"   {s['share']:.0%}  round {s['round']}  p={s['prob']:.3f}  {s['site']:<20} {s['url']}{mark}")
    print(f"   {fin['unexplained']:.0%}  unexplained")
    ctx.emb.save()


def cmd_eval_multi(args):
    from pathlib import Path
    from .eval import ms_eval
    xl = Path(args.xl) if args.xl else None
    if args.live:
        ms_eval.live(args.sample, xl)
    else:
        ms_eval.main(fresh=args.fresh, xl=xl, ablations=not args.no_ablations, smoke=args.smoke)


def cmd_format_eval(args):
    from .eval import formats
    formats.main(fresh=args.fresh)


def cmd_demo(args):
    """Trace randomly drawn test cases (seeded, never hand-picked), including a miss and a false alarm if any."""
    import random
    from .detect import verify
    from .eval import report
    data = report.load()
    meta = data["meta"]
    test = [it for it in data["items"] if it["split"] == "test"]
    model = verify.load()
    rng = random.Random(args.seed)
    picks = []
    for level in ("seo", "summary", "facts_only", "hard_negative"):
        group = [it for it in test if it["level"] == level]
        if group:
            picks.append((f"random {level}", rng.choice(group)))
    misses = [it for it in test if it["label"] == 1 and report.spin_score(it, model)[0] < model["threshold"]]
    alarms = [it for it in test if it["label"] == 0 and report.spin_score(it, model)[0] >= model["threshold"]]
    if misses:
        picks.append(("a missed rewrite (limitation)", rng.choice(misses)))
    if alarms:
        picks.append(("a false alarm (limitation)", rng.choice(alarms)))
    for why, it in picks:
        print("=" * 100 + f"\nDEMO CASE: {why}; ground truth: {'derived' if it['label'] else 'not derived'}"
              + (f" from {meta['doc_ids'][it['source']]}" if it["label"] else ""))
        cmd_suspect(argparse.Namespace(doc=meta["doc_ids"][it["doc"]], sentences=args.sentences, stem=False))


def cmd_scan(args):
    from .detect import pipeline, scan
    ctx = pipeline.Context()
    n, flagged = scan.run(ctx, args.where, multi=args.multi)
    print(f"scanned {n} docs, flagged {flagged}")
    print(f"live pairs for judging: {scan.export_flagged(ctx.con)} -> {scan.FLAGGED}")
    for r in scan.domain_report(ctx.con)[:15]:
        print(f"  {r['site']:<26} {r['flagged_derived']:>4}/{r['articles']:<5} {r['derived_share']:.3f}  {r['top_sources']}")


def cmd_search(args):
    from . import index, store
    from .detect import scan
    from .search import search
    idx = index.load()
    con = store.connect()
    g = None if args.no_g else scan.originality(con)
    scan.originality(con)
    for rank, (d, net, rel, gd) in enumerate(search(idx, args.query, g, lam=args.lam, k=args.k, origins=None), 1):
        doc_id = idx.doc_ids[d]
        row = con.execute("SELECT title, url, published FROM docs WHERE doc_id=?", (doc_id,)).fetchone()
        src = con.execute("SELECT d.site FROM edges e JOIN docs d ON d.doc_id=e.source WHERE e.suspect=? "
                          "ORDER BY e.prob DESC LIMIT 1", (doc_id,)).fetchone()
        note = f"  [copy of {src[0]}]" if src else ""
        print(f"{rank:>2}. net={net:.3f} rel={rel:.3f} g={gd:.2f} {idx.meta['site'][d]:<20} {_when(row['published'])} "
              f"{(row['title'] or row['url'])[:70]}{note}")


def cmd_eval(args):
    """Score every evaluation item, fit on dev, report test metrics and write tables and charts."""
    from .eval import hardneg, report, run
    if args.mine or not hardneg.OUT.exists():
        print(f"hard negatives mined: {hardneg.mine()}")
    if not args.report_only:
        run.run(stem_compare=not args.no_stem, sample=args.sample)
    det, rank, abl, ret, eff, counts, model = report.main()
    print("eval sets:", json.dumps(counts))
    print(f"verifier fit on dev: threshold={model['threshold']:.3f}")
    print("\nDetection F1 on test, by rewrite level")
    methods = list(dict.fromkeys(r["method"] for r in det))
    levels = [l for l in report.LEVELS if any(r["level"] == l for r in det)]
    print(f"{'method':<20}" + "".join(f"{l:>9}" for l in levels) + f"{'FPR hard':>10}{'FPR facts':>10}")
    for m in methods:
        f1 = {r["level"]: r["f1"] for r in det if r["method"] == m}
        print(f"{m:<20}" + "".join(f"{f1.get(l, float('nan')):>9.3f}" for l in levels)
              + f"{f1.get('fpr_hard_negative', float('nan')):>10.3f}{f1.get('fpr_facts_only', float('nan')):>10.3f}")
    print("\nRanking of the true original (test positives, all levels)")
    for r in rank:
        if r["level"] == "all":
            print(f"{r['method']:<18} P@1={r['recall@1']:.3f} MRR={r['mrr']:.3f} R@5={r['recall@5']:.3f} R@20={r['recall@20']:.3f}")
    print("\nAblations (macro F1 over levels)")
    for r in abl:
        print(f"  {r['variant']:<40} {r['macro_f1']:.3f}  FPR hard={r.get('fpr_hard_negative', float('nan')):.3f}"
              f"  FPR facts={r.get('fpr_facts_only', float('nan')):.3f}")
    from .detect import pipeline
    from .eval import realpairs, summary
    print("\nNewsGuard-documented pairs:")
    for r in realpairs.evaluate(pipeline.Context()):
        print("  " + "  ".join(f"{k}={v}" for k, v in r.items()))
    summary.build()
    print(f"\ntables and charts written to {config.RESULTS} (summary.md)")


def cmd_search_eval(args):
    from .eval import search_eval
    for r in search_eval.main():
        print("  ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in r.items()))
    print(f"live queries for judging: {search_eval.JUDGE}")
    from .eval import summary
    summary.build()


def cmd_judge(args):
    from .eval import judged
    if args.metrics:
        print(json.dumps(judged.metrics(), indent=2))
    else:
        judged.label(args.sheet, args.judge)


def cmd_judge_ui(args):
    from .ui import judge_server
    judge_server.serve(args.host, args.port)


def cmd_judge_pack(args):
    from .eval import judgepack
    n, wanted, size = judgepack.build(args.hardneg)
    print(f"{n} of {wanted} documents written to {judgepack.PACK} ({size / 1e6:.1f} MB); send it privately, never commit it")


def cmd_judge_merge(args):
    from .eval import judgepack
    print(judgepack.merge(args.file, args.sheet, args.judge))


def cmd_app(args):
    from .ui import app_server
    app_server.serve(args.host, args.port)


def cmd_crawl_report(args):
    from .crawl import report
    rep = report.write()
    for k, v in rep.items():
        if k not in ("min_gap_seconds_by_host", "docs_by_site", "delay_violation_examples"):
            print(f"{k}: {v}")
    print(f"written to {config.RESULTS / 'crawl_report.json'}\n")
    print("\n".join(report.examples()))
    from .eval import summary
    summary.build()


def main(argv=None):
    p = argparse.ArgumentParser(prog="spintrace")
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("crawl", help="polite focused crawl from the seed list")
    c.add_argument("--seeds", default=str(config.SEEDS / "seeds.csv"))
    c.add_argument("--hours", type=float)
    c.add_argument("--max-pages", type=int)
    c.set_defaults(fn=cmd_crawl)

    c = sub.add_parser("export", help="write crawled URLs and MinHash features (no texts) to evaldata/")
    c.set_defaults(fn=cmd_export)

    c = sub.add_parser("recrawl", help="politely re-fetch the exported URLs into a fresh store")
    c.add_argument("--urls", default=str(config.ROOT / "evaldata" / "crawl_urls.csv"))
    c.add_argument("--seeds", default=str(config.SEEDS / "seeds.csv"))
    c.add_argument("--limit", type=int)
    c.set_defaults(fn=cmd_recrawl)

    c = sub.add_parser("newsguard", help="load the NewsGuard-documented farm pairs from the Wayback Machine")
    c.set_defaults(fn=cmd_newsguard)

    c = sub.add_parser("ccnews", help="store English articles from the downloaded CC-NEWS file (scale experiment)")
    c.add_argument("--max", type=int, default=20000)
    c.set_defaults(fn=cmd_ccnews)

    c = sub.add_parser("scale", help="test items against an index that also holds CC-NEWS (no refitting)")
    c.set_defaults(fn=cmd_scale)

    c = sub.add_parser("build", help="ingest eval docs, compute shingles/MinHash, build the index")
    c.add_argument("--stem", action="store_true", help="also build a Porter-stemmed index")
    c.add_argument("--no-eval", action="store_true", help="index crawled pages only")
    c.add_argument("--with-ccnews", action="store_true", help="also index the CC-NEWS documents")
    c.set_defaults(fn=cmd_build)

    c = sub.add_parser("postings", help="print the postings of a term")
    c.add_argument("term")
    c.add_argument("--limit", type=int, default=10)
    c.add_argument("--stem", action="store_true")
    c.set_defaults(fn=cmd_postings)

    c = sub.add_parser("pair", help="shingle overlap, Jaccard and MinHash for two documents")
    c.add_argument("a")
    c.add_argument("b")
    c.set_defaults(fn=cmd_pair)

    c = sub.add_parser("suspect", help="trace one document: query, candidates, alignment, verdict")
    c.add_argument("doc", help="doc_id or URL")
    c.add_argument("--sentences", type=int, default=6)
    c.add_argument("--stem", action="store_true")
    c.set_defaults(fn=cmd_suspect)

    c = sub.add_parser("sources", help="every source of a mashup, round by round, with sentence shares")
    c.add_argument("doc", nargs="?", help="doc_id, URL, mashup id (mash:...) or LLM mashup id (mashllm:...)")
    c.add_argument("--file", help="trace the text in this file instead (not in the index)")
    c.add_argument("-k", type=int, default=8, help="candidates to print per round")
    c.add_argument("--sentences", type=int, default=4, help="covered sentences to print per round")
    c.add_argument("--xl", help="path of rewrites_xl.jsonl, for LLM mashups")
    c.set_defaults(fn=cmd_sources)

    c = sub.add_parser("eval-multi", help="multi-source evaluation: mashups, controls, baselines, ablations")
    c.add_argument("--fresh", action="store_true", help="recompute every cached pass")
    c.add_argument("--no-ablations", action="store_true")
    c.add_argument("--smoke", type=int, help="run on this many random items per split only")
    c.add_argument("--xl", help="path of rewrites_xl.jsonl, for LLM mashups")
    c.add_argument("--live", action="store_true", help="trace the flagged live articles instead")
    c.add_argument("--sample", type=int, default=300, help="live articles to trace")
    c.set_defaults(fn=cmd_eval_multi)

    c = sub.add_parser("format-eval", help="format changes (listicle, Q&A): midterm verifier against refits on dev")
    c.add_argument("--fresh", action="store_true", help="rescore every item instead of using the cached run")
    c.set_defaults(fn=cmd_format_eval)

    c = sub.add_parser("demo", help="trace seeded random test cases, including a miss and a false alarm")
    c.add_argument("--seed", type=int, default=7)
    c.add_argument("--sentences", type=int, default=4)
    c.set_defaults(fn=cmd_demo)

    c = sub.add_parser("scan", help="run detection over the corpus and store the copy graph")
    c.add_argument("--where", default="origin='crawl'")
    c.add_argument("--multi", action="store_true", help="also add an edge for every further verified source")
    c.set_defaults(fn=cmd_scan)

    c = sub.add_parser("search", help="ranked search with originality g(d)")
    c.add_argument("query")
    c.add_argument("-k", type=int, default=10)
    c.add_argument("--lam", type=float, default=0.3)
    c.add_argument("--no-g", action="store_true")
    c.set_defaults(fn=cmd_search)

    c = sub.add_parser("eval", help="synthetic and hard-negative evaluation against baselines")
    c.add_argument("--report-only", action="store_true", help="recompute metrics from the cached run")
    c.add_argument("--mine", action="store_true", help="re-mine hard negatives from the current crawl")
    c.add_argument("--no-stem", action="store_true")
    c.add_argument("--sample", type=int)
    c.set_defaults(fn=cmd_eval)

    c = sub.add_parser("search-eval", help="do originals outrank copies? P@10 and nDCG@10 with and without g(d)")
    c.set_defaults(fn=cmd_search_eval)

    c = sub.add_parser("judge", help="label flagged live pairs, hard negatives or search results (team judges)")
    c.add_argument("sheet", choices=["live", "hardneg", "search"])
    c.add_argument("--judge", type=int, choices=[1, 2], default=1)
    c.add_argument("--metrics", action="store_true", help="print precision and agreement from the labels so far")
    c.set_defaults(fn=cmd_judge)

    c = sub.add_parser("judge-ui", help="web interface for the same judgments, in the browser")
    c.add_argument("--host", default="127.0.0.1", help="0.0.0.0 to let a teammate judge from another machine on the network")
    c.add_argument("--port", type=int, default=8765)
    c.set_defaults(fn=cmd_judge_ui)

    c = sub.add_parser("judge-pack", help="write the texts a remote judge needs (private, not for git)")
    c.add_argument("--hardneg", type=int, default=100, help="how many hard-negative rows to include")
    c.set_defaults(fn=cmd_judge_pack)

    c = sub.add_parser("judge-merge", help="copy another judge's labels from their copy of a sheet into ours")
    c.add_argument("sheet", choices=["live", "hardneg", "search"])
    c.add_argument("file", help="their copy of the sheet csv")
    c.add_argument("--judge", type=int, choices=[1, 2], default=2)
    c.set_defaults(fn=cmd_judge_merge)

    c = sub.add_parser("app", help="web app: search with originality and the per-site integrity table")
    c.add_argument("--host", default="127.0.0.1")
    c.add_argument("--port", type=int, default=8766)
    c.set_defaults(fn=cmd_app)

    c = sub.add_parser("crawl-report", help="politeness and correctness statistics of the crawl")
    c.set_defaults(fn=cmd_crawl_report)

    args = p.parse_args(argv)
    args.fn(args)
