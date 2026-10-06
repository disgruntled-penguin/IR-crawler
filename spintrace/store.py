"""SQLite storage shared by every stage. Documents are keyed by a stable doc_id (sha1 of the canonical URL)."""
import hashlib
import json
import sqlite3
import time

from . import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS urls (
    url TEXT PRIMARY KEY,
    host TEXT NOT NULL,
    priority INTEGER NOT NULL,
    status TEXT NOT NULL DEFAULT 'queued',
    source TEXT,
    hint_date REAL,
    discovered_at REAL,
    fetched_at REAL,
    http_status INTEGER,
    note TEXT
);
CREATE INDEX IF NOT EXISTS urls_status ON urls(status);
CREATE TABLE IF NOT EXISTS docs (
    doc_id TEXT PRIMARY KEY,
    url TEXT UNIQUE NOT NULL,
    host TEXT NOT NULL,
    site TEXT NOT NULL,
    kind TEXT NOT NULL,
    origin TEXT NOT NULL,
    title TEXT,
    text TEXT NOT NULL,
    published REAL,
    date_source TEXT,
    fetched_at REAL,
    n_words INTEGER,
    content_hash TEXT,
    meta TEXT
);
CREATE INDEX IF NOT EXISTS docs_site ON docs(site);
CREATE TABLE IF NOT EXISTS fetch_log (
    ts REAL NOT NULL,
    host TEXT NOT NULL,
    url TEXT NOT NULL,
    event TEXT NOT NULL,
    status INTEGER,
    detail TEXT
);
CREATE INDEX IF NOT EXISTS fetch_log_host ON fetch_log(host, ts);
CREATE TABLE IF NOT EXISTS robots (
    host TEXT NOT NULL,
    fetched_at REAL NOT NULL,
    status INTEGER,
    body TEXT
);
CREATE TABLE IF NOT EXISTS hosts (
    host TEXT PRIMARY KEY,
    state TEXT NOT NULL,
    detail TEXT,
    updated REAL
);
"""


def doc_id_for(url):
    return hashlib.sha1(url.encode()).hexdigest()[:16]


def connect(path=None):
    config.ensure_dirs()
    con = sqlite3.connect(str(path or config.DB_PATH), timeout=60)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")
    con.executescript(SCHEMA)
    return con


def add_url(con, url, host, priority, source, hint_date=None):
    cur = con.execute(
        "INSERT OR IGNORE INTO urls(url, host, priority, source, hint_date, discovered_at) VALUES (?,?,?,?,?,?)",
        (url, host, priority, source, hint_date, time.time()),
    )
    return cur.rowcount > 0


def mark_url(con, url, status, http_status=None, note=None):
    con.execute(
        "UPDATE urls SET status=?, fetched_at=?, http_status=?, note=? WHERE url=?",
        (status, time.time(), http_status, note, url),
    )


def log_event(con, host, url, event, status=None, detail=None, ts=None):
    con.execute(
        "INSERT INTO fetch_log(ts, host, url, event, status, detail) VALUES (?,?,?,?,?,?)",
        (ts or time.time(), host, url, event, status, detail),
    )


def save_doc(con, doc):
    con.execute(
        """INSERT OR REPLACE INTO docs(doc_id, url, host, site, kind, origin, title, text, published,
           date_source, fetched_at, n_words, content_hash, meta) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            doc["doc_id"], doc["url"], doc["host"], doc["site"], doc["kind"], doc["origin"], doc.get("title"),
            doc["text"], doc.get("published"), doc.get("date_source"), doc.get("fetched_at"),
            len(doc["text"].split()), hashlib.sha1(doc["text"].encode()).hexdigest(),
            json.dumps(doc.get("meta") or {}),
        ),
    )


def iter_docs(con, where="1=1", params=()):
    for row in con.execute(f"SELECT * FROM docs WHERE {where} ORDER BY doc_id", params):
        yield dict(row)


def set_host_state(con, host, state, detail=None):
    con.execute(
        "INSERT OR REPLACE INTO hosts(host, state, detail, updated) VALUES (?,?,?,?)",
        (host, state, detail, time.time()),
    )
