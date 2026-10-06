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


def cmd_build(args):
    """Ingest evaluation documents, compute dedup features and build the index(es)."""
    from . import corpus, index, store
    con = store.connect()
    if not args.no_eval:
        print("ingested wikinews=%d synthetic=%d" % corpus.ingest_eval(con))
    feats, secs = corpus.build_features(con)
    print(f"features for {len(feats)} docs in {secs:.1f}s")
    for stem in ([False, True] if args.stem else [False]):
        idx = index.build_from_store(stemming=stem, con=con)
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


def main(argv=None):
    p = argparse.ArgumentParser(prog="spintrace")
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("crawl", help="polite focused crawl from the seed list")
    c.add_argument("--seeds", default=str(config.SEEDS / "seeds.csv"))
    c.add_argument("--hours", type=float)
    c.add_argument("--max-pages", type=int)
    c.set_defaults(fn=cmd_crawl)

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

    args = p.parse_args(argv)
    args.fn(args)
