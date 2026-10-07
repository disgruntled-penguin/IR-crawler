"""Hand a judge the texts they need without the crawl database, and merge their labels back.

The pack holds crawled full texts, so it stays out of git (see .gitignore) and is sent to the other judge privately.
"""
import csv
import json

from .. import config, store
from . import judged

PACK = config.ROOT / "evaldata" / "judge_pack.json"
KEYS = {"live": ("suspect", "source"), "hardneg": ("suspect", "earlier"), "search": ("query", "doc_id")}
DOC_COLS = {"live": ("suspect", "source"), "hardneg": ("suspect", "earlier"), "search": ("doc_id",)}


def build(hardneg_rows=100):
    """Write the texts behind the live, search and first hardneg_rows hard-negative rows."""
    ids = set()
    for sheet, cols in DOC_COLS.items():
        rows = judged.read_rows(sheet)
        for r in rows[:hardneg_rows] if sheet == "hardneg" else rows:
            ids.update(r[c] for c in cols)
    con = store.connect()
    docs = {}
    for d in ids:
        r = con.execute("SELECT doc_id, url, site, title, published, text FROM docs WHERE doc_id=?", (d,)).fetchone()
        if r:
            docs[d] = dict(r)
    PACK.write_text(json.dumps({"docs": docs}))
    return len(docs), len(ids), PACK.stat().st_size


def load():
    return json.loads(PACK.read_text())["docs"] if PACK.exists() else {}


def merge(path, sheet, judge=2):
    """Copy one judge's labels from another copy of a sheet into ours; never overwrites a different existing label."""
    col, key = f"judge{judge}", KEYS[sheet]
    theirs = {tuple(r[k] for k in key): r[col] for r in csv.DictReader(open(path)) if r.get(col)}
    rows = judged.read_rows(sheet)
    added = clash = 0
    for r in rows:
        v = theirs.get(tuple(r[k] for k in key))
        if not v:
            continue
        if not r[col]:
            r[col] = v
            added += 1
        elif r[col] != v:
            clash += 1
    with judged.SHEETS[sheet].open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return {"labels_in_file": len(theirs), "added": added, "clashes_kept_ours": clash}
