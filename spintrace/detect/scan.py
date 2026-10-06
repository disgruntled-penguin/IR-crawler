"""Run detection over every document and store the copy graph; originality g(d) comes from it."""
import json
import time

from .. import store
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
