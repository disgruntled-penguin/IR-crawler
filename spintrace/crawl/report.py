"""Crawler correctness and politeness report, computed from the crawl's own logs."""
import datetime
import json
import urllib.robotparser
from collections import Counter, defaultdict

from .. import config, corpus, dedup, store


def _parsers(con):
    """Every stored robots.txt version per host, oldest first."""
    out = defaultdict(list)
    for host, ts, status, body in con.execute("SELECT host, fetched_at, status, body FROM robots ORDER BY fetched_at"):
        rp = urllib.robotparser.RobotFileParser()
        if body == "__unreachable__" or status in (401, 403):
            rp.disallow_all = True
        elif status is None or status >= 400 or body is None:
            rp.allow_all = True
        else:
            rp.parse(body.splitlines())
        out[host].append((ts, rp))
    return out


def build(con=None):
    con = con or store.connect()
    rep = {}
    fetches = con.execute("SELECT ts, host, url, status FROM fetch_log WHERE event='fetch' ORDER BY ts").fetchall()
    rep["requests_page"] = len(fetches)
    rep["requests_robots"] = con.execute("SELECT COUNT(*) FROM fetch_log WHERE event='robots_fetched'").fetchone()[0]
    rep["docs_saved"] = con.execute("SELECT COUNT(*) FROM docs WHERE origin='crawl'").fetchone()[0]
    if fetches:
        hours = (fetches[-1][0] - fetches[0][0]) / 3600
        rep["crawl_hours"] = round(hours, 2)
        rep["pages_per_minute"] = round(len(fetches) / max(hours * 60, 1e-9), 1)
    rep["status_counts"] = dict(Counter(str(s) for _, _, _, s in fetches).most_common())
    rep["hosts"] = len({h for _, h, _, _ in fetches})

    # Re-check every fetched URL against the robots.txt version in force at the time (or the earliest stored).
    parsers = _parsers(con)
    violations, audited, unaudited = [], 0, 0
    for ts, host, url, _ in fetches:
        versions = parsers.get(host)
        if not versions:
            unaudited += 1
            continue
        rp = versions[0][1]
        for vts, vrp in versions:
            if vts <= ts:
                rp = vrp
        audited += 1
        if not rp.can_fetch(config.ROBOTS_AGENT, url):
            violations.append(url)
    rep["robots_audited_fetches"] = audited
    rep["robots_unaudited_fetches"] = unaudited
    rep["disallowed_urls_fetched"] = len(violations)
    rep["disallowed_examples"] = violations[:5]
    rep["robots_skips"] = con.execute("SELECT COUNT(*) FROM fetch_log WHERE event='robots_skip'").fetchone()[0]

    # Smallest gap between consecutive requests (pages and robots.txt) to one host.
    gaps = {}
    rows = con.execute("SELECT host, ts FROM fetch_log WHERE event IN ('fetch','robots_fetched') ORDER BY host, ts")
    prev = {}
    violations_gap = []
    for host, ts in rows:
        if host in prev:
            g = ts - prev[host]
            gaps[host] = min(gaps.get(host, g), g)
            if g < config.DEFAULT_DELAY - 0.01:
                violations_gap.append((ts, host, round(g, 2)))
        prev[host] = ts
    violations_gap.sort()
    rep["delay_violations"] = len(violations_gap)
    rep["last_delay_violation"] = (
        datetime.datetime.fromtimestamp(violations_gap[-1][0]).isoformat(timespec="seconds")
        if violations_gap else None)
    rep["delay_violation_examples"] = violations_gap[-10:]
    rep["min_gap_seconds_overall"] = round(min(gaps.values()), 2) if gaps else None
    rep["min_gap_seconds_by_host"] = {h: round(g, 2) for h, g in sorted(gaps.items(), key=lambda x: x[1])}
    rep["configured_min_delay"] = config.DEFAULT_DELAY

    events = Counter(r[0] for r in con.execute("SELECT event FROM fetch_log"))
    rep["trap_guards_fired"] = events.get("trap_guard", 0)
    rep["trap_guard_reasons"] = dict(Counter(
        r[0].split()[-1] for r in con.execute("SELECT detail FROM fetch_log WHERE event='trap_guard'")))
    rep["backoffs"] = events.get("backoff", 0)
    rep["hosts_stopped"] = [dict(r) for r in con.execute("SELECT host, detail FROM hosts WHERE state='stopped'")]
    rep["not_article_pages"] = events.get("not_article", 0)

    # Content-seen check over the crawl: exact duplicates and MinHash near-duplicates (Jaccard >= 0.8).
    feats = corpus.load_features() if corpus.FEATURES.exists() else {}
    crawl = [r[0] for r in con.execute("SELECT doc_id FROM docs WHERE origin='crawl'") if r[0] in feats]
    exact = Counter(feats[d]["exact"] for d in crawl)
    rep["content_seen_docs"] = len(crawl)
    rep["exact_duplicate_docs"] = sum(c - 1 for c in exact.values() if c > 1)
    lsh = dedup.LSH()
    for d in crawl:
        lsh.add(d, feats[d]["minhash"])
    near = set()
    for a, b in lsh.pairs():
        if dedup.jaccard(feats[a]["shingles"], feats[b]["shingles"]) >= 0.8:
            near.add(b)
    rep["near_duplicate_docs"] = len(near)
    rep["duplicate_rate"] = round((rep["exact_duplicate_docs"] + len(near)) / max(len(crawl), 1), 4)
    rep["docs_by_site"] = dict(Counter(r[0] for r in con.execute("SELECT site FROM docs WHERE origin='crawl'")).most_common())
    rep["date_sources"] = dict(Counter(str(r[0]) for r in con.execute("SELECT date_source FROM docs WHERE origin='crawl'")))
    return rep


def write(con=None):
    rep = build(con)
    config.RESULTS.mkdir(exist_ok=True)
    (config.RESULTS / "crawl_report.json").write_text(json.dumps(rep, indent=2))
    return rep
