"""One CC-NEWS WARC file (Common Crawl's news crawl) as extra real-world documents for a scale experiment.

The file is read locally; nothing from it is committed. Records go through the same extraction as the live
crawl, are kept if they look like English articles, and are stored with origin 'ccnews', so the main index
can leave them out and the scale index can put them in.
"""
import time

from warcio.archiveiterator import ArchiveIterator

from .. import config, store, text as tx
from ..crawl import extract
from ..crawl.urlnorm import host_of, normalise, site_of

WARC_URL = "https://data.commoncrawl.org/crawl-data/CC-NEWS/2026/10/CC-NEWS-20261006172634-00164.warc.gz"
WARC = config.DATA / "raw" / "CC-NEWS-20261006172634-00164.warc.gz"


def english(text):
    toks = tx.tokens(text[:3000])
    return len(toks) > 50 and sum(t in tx.STOPWORDS for t in toks) / len(toks) > 0.3


def load(max_docs=20000, con=None):
    con = con or store.connect()
    have = {r[0] for r in con.execute("SELECT url FROM docs")}
    kept = seen = 0
    t0 = time.time()
    with WARC.open("rb") as f:
        for rec in ArchiveIterator(f):
            if rec.rec_type != "response":
                continue
            ctype = rec.http_headers.get_header("Content-Type") or ""
            if "html" not in ctype:
                continue
            seen += 1
            url = normalise(rec.rec_headers.get_header("WARC-Target-URI") or "")
            if not url or url in have:
                continue
            try:
                page = extract.extract_page(rec.content_stream().read(), url)
            except Exception:
                continue
            if not page or len(page["text"].split()) < config.MIN_ARTICLE_WORDS or not english(page["text"]):
                continue
            if not (page["is_article"] or page["published"]):
                continue
            host = host_of(url)
            store.save_doc(con, {
                "doc_id": store.doc_id_for(url), "url": url, "host": host, "site": site_of(host), "kind": "original",
                "origin": "ccnews", "title": page["title"], "text": page["text"], "published": page["published"],
                "date_source": page["date_source"], "fetched_at": None,
            })
            have.add(url)
            kept += 1
            if kept % 1000 == 0:
                con.commit()
                print(f"ccnews kept {kept} of {seen} html records, {time.time() - t0:.0f}s", flush=True)
            if kept >= max_docs:
                break
    con.commit()
    return kept, seen
