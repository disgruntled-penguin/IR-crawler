"""NewsGuard-documented farm rewrites: fetch both sides from the Wayback Machine and trace them.

Only URLs are committed (evaldata/newsguard_pairs.csv); texts are re-fetched by this loader.
Requests go one at a time with a delay, and raw snapshots (id_) are requested so the archive's
toolbar is not mistaken for article text.
"""
import csv
import re
import time

import requests

from .. import config, store
from ..crawl import extract
from ..crawl.urlnorm import host_of, site_of

PAIRS = config.ROOT / "evaldata" / "newsguard_pairs.csv"
DELAY = 5.0
WAYBACK = re.compile(r"^https://web\.archive\.org/web/(\d{14})[a-z_]*/(.+)$")


def _snapshot(session, url):
    """Return (raw_html, original_url, capture_epoch) for a Wayback URL or a live URL's closest snapshot."""
    m = WAYBACK.match(url)
    if not m:
        r = session.get("https://archive.org/wayback/available", params={"url": url}, timeout=60)
        time.sleep(DELAY)
        snap = r.json().get("archived_snapshots", {}).get("closest")
        if not snap:
            return None
        m = WAYBACK.match(snap["url"].replace("http://", "https://"))
    ts, orig = m.groups()
    raw = session.get(f"https://web.archive.org/web/{ts}id_/{orig}", timeout=60)
    time.sleep(DELAY)
    if raw.status_code != 200:
        return None
    t = time.mktime(time.strptime(ts, "%Y%m%d%H%M%S"))
    return raw.content, orig, t


def load(con=None):
    con = con or store.connect()
    s = requests.Session()
    s.headers["User-Agent"] = config.USER_AGENT
    out = []
    for row in csv.DictReader(PAIRS.open()):
        if not row["original_url"]:
            out.append((row["pair_id"], "original_url_unknown"))
            continue
        ids = {}
        for side, kind in (("farm_url", "suspect"), ("original_url", "original")):
            try:
                snap = _snapshot(s, row[side])
            except (requests.RequestException, ValueError) as e:
                snap = None
                print(f"{row['pair_id']} {side}: {e}")
            if not snap:
                break
            raw, orig, captured = snap
            page = extract.extract_page(raw, orig)
            if not page or len(page["text"].split()) < 80:
                break
            host = host_of(orig)
            published, source = page["published"], page["date_source"]
            if not published:
                published, source = captured, "wayback_capture"
            doc_id = store.doc_id_for(orig)
            store.save_doc(con, {"doc_id": doc_id, "url": orig, "host": host, "site": site_of(host), "kind": kind,
                                 "origin": "newsguard", "title": page["title"], "text": page["text"],
                                 "published": published, "date_source": source, "fetched_at": time.time(),
                                 "meta": {"pair_id": row["pair_id"]}})
            ids[side] = doc_id
        out.append((row["pair_id"], "loaded" if len(ids) == 2 else "fetch_failed"))
    con.commit()
    return out
