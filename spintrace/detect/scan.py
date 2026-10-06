"""Run detection over every document and store the copy graph; originality g(d) comes from it."""
import csv
import json
import time
from collections import Counter, defaultdict

from .. import config, store
from . import pipeline

SCHEMA = """
CREATE TABLE IF NOT EXISTS edges (
    suspect TEXT NOT NULL,
    source TEXT NOT NULL,
    prob REAL NOT NULL,
    verdict TEXT,
    why TEXT,
    signals TEXT,
    PRIMARY KEY (suspect, source)
);
"""


def run(ctx, where="1=1", log_every=200):
    """Analyse each matching document and keep its verified edges above threshold."""
    con = ctx.con
    con.executescript(SCHEMA)
    ids = [r[0] for r in con.execute(f"SELECT doc_id FROM docs WHERE {where}")]
    t0 = time.time()
    flagged = 0
    for i, doc_id in enumerate(ids):
        if doc_id not in ctx.idx.doc_num:
            continue
        rep = pipeline.analyse(ctx, ctx.num(doc_id))
        con.execute("DELETE FROM edges WHERE suspect=?", (doc_id,))
        if rep["flagged"]:
            flagged += 1
            v = rep["source"]
            con.execute("INSERT OR REPLACE INTO edges VALUES (?,?,?,?,?,?)",
                        (doc_id, v["doc_id"], v["prob"], v["verdict"], v["why"], json.dumps(v["signals"])))
        if (i + 1) % log_every == 0:
            con.commit()
            ctx.emb.save()
            print(f"scan {i + 1}/{len(ids)} flagged={flagged} {time.time() - t0:.0f}s", flush=True)
    con.commit()
    ctx.emb.save()
    return len(ids), flagged


def originality(con):
    """g(d) = 1 - derivative probability of d's strongest source edge; 1 for documents with no source."""
    con.executescript(SCHEMA)
    return {r[0]: 1.0 - r[1] for r in con.execute("SELECT suspect, MAX(prob) FROM edges GROUP BY suspect")}


def root(con, doc_id, limit=10):
    """Follow source edges back to the earliest known origin."""
    chain = [doc_id]
    while len(chain) <= limit:
        r = con.execute("SELECT source FROM edges WHERE suspect=? ORDER BY prob DESC LIMIT 1", (chain[-1],)).fetchone()
        if not r or r[0] in chain:
            break
        chain.append(r[0])
    return chain


FLAGGED = config.ROOT / "evaldata" / "live_flagged.csv"


def export_flagged(con):
    """Flagged live pairs for both judges to label (derived / not_derived); URLs and scores only."""
    rows = con.execute("""SELECT e.suspect, e.source, e.prob, e.verdict, e.why, e.signals, s.url AS surl, o.url AS ourl,
                          s.site AS ssite, o.site AS osite FROM edges e JOIN docs s ON s.doc_id=e.suspect
                          JOIN docs o ON o.doc_id=e.source WHERE s.origin='crawl' ORDER BY e.prob DESC""").fetchall()
    old = {}
    if FLAGGED.exists():
        old = {(r["suspect"], r["source"]): r for r in csv.DictReader(FLAGGED.open())}
    with FLAGGED.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["suspect", "source", "suspect_url", "source_url", "prob", "verdict", "why", "align_coverage",
                    "shingle_containment", "judge1", "judge2"])
        for r in rows:
            sig = json.loads(r["signals"])
            prev = old.get((r["suspect"], r["source"]), {})
            w.writerow([r["suspect"], r["source"], r["surl"], r["ourl"], f"{r['prob']:.3f}", r["verdict"], r["why"],
                        f"{sig.get('align_coverage', 0):.2f}", f"{sig.get('shingle_containment', 0):.2f}",
                        prev.get("judge1", ""), prev.get("judge2", "")])
    return len(rows)


def domain_report(con):
    """Per-site integrity: share of a site's crawled articles flagged as derived, and whom they derive from."""
    total = Counter(r[0] for r in con.execute("SELECT site FROM docs WHERE origin='crawl'"))
    flagged = Counter()
    sources = defaultdict(Counter)
    for ssite, osite in con.execute("""SELECT s.site, o.site FROM edges e JOIN docs s ON s.doc_id=e.suspect
                                       JOIN docs o ON o.doc_id=e.source WHERE s.origin='crawl'"""):
        flagged[ssite] += 1
        sources[ssite][osite] += 1
    rows = []
    for site, n in total.most_common():
        rows.append({"site": site, "articles": n, "flagged_derived": flagged[site],
                     "derived_share": round(flagged[site] / n, 4),
                     "top_sources": "; ".join(f"{k} ({v})" for k, v in sources[site].most_common(3))})
    rows.sort(key=lambda r: -r["derived_share"])
    path = config.RESULTS / "domain_integrity.csv"
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return rows
