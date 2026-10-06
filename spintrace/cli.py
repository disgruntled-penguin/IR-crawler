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


def cmd_build(args):
    """Ingest evaluation documents, compute dedup features and build the index(es)."""
    from . import corpus, index, store
    con = store.connect()
    if not args.no_eval:
        print("ingested wikinews=%d synthetic=%d" % corpus.ingest_eval(con))
    feats, secs = corpus.build_features(con)
    print(f"features for {len(feats)} docs in {secs:.1f}s")
    for stem in ([False, True] if args.stem else [False]):
        idx = index.build_from_store(stemming=stem, con=con, only=set(feats))
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


def cmd_scan(args):
    from .detect import pipeline, scan
    ctx = pipeline.Context()
    n, flagged = scan.run(ctx, args.where)
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
    for rank, (d, net, rel, gd) in enumerate(search(idx, args.query, g, lam=args.lam, k=args.k, origins=None), 1):
        row = con.execute("SELECT title, url, published FROM docs WHERE doc_id=?", (idx.doc_ids[d],)).fetchone()
        print(f"{rank:>2}. net={net:.3f} rel={rel:.3f} g={gd:.2f} {idx.meta['site'][d]:<20} {_when(row['published'])} "
              f"{(row['title'] or row['url'])[:70]}")


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
    print(f"{'method':<18}" + "".join(f"{l:>10}" for l in levels) + f"{'FPR hard':>10}")
    for m in methods:
        f1 = {r["level"]: r["f1"] for r in det if r["method"] == m}
        print(f"{m:<18}" + "".join(f"{f1.get(l, float('nan')):>10.3f}" for l in levels)
              + f"{f1.get('fpr_hard_negative', float('nan')):>10.3f}")
    print("\nRanking of the true original (test positives, all levels)")
    for r in rank:
        if r["level"] == "all":
            print(f"{r['method']:<18} P@1={r['recall@1']:.3f} MRR={r['mrr']:.3f} R@5={r['recall@5']:.3f} R@20={r['recall@20']:.3f}")
    print("\nAblations (macro F1 over levels)")
    for r in abl:
        print(f"  {r['variant']:<38} {r['macro_f1']:.3f}  FPR hard={r.get('fpr_hard_negative', float('nan')):.3f}")
    print(f"\ntables and charts written to {config.RESULTS}")


def cmd_crawl_report(args):
    from .crawl import report
    rep = report.write()
    for k, v in rep.items():
        if k not in ("min_gap_seconds_by_host", "docs_by_site", "delay_violation_examples"):
            print(f"{k}: {v}")
    print(f"written to {config.RESULTS / 'crawl_report.json'}")


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

    c = sub.add_parser("build", help="ingest eval docs, compute shingles/MinHash, build the index")
    c.add_argument("--stem", action="store_true", help="also build a Porter-stemmed index")
    c.add_argument("--no-eval", action="store_true", help="index crawled pages only")
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

    c = sub.add_parser("scan", help="run detection over the corpus and store the copy graph")
    c.add_argument("--where", default="origin='crawl'")
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

    c = sub.add_parser("crawl-report", help="politeness and correctness statistics of the crawl")
    c.set_defaults(fn=cmd_crawl_report)

    args = p.parse_args(argv)
    args.fn(args)
