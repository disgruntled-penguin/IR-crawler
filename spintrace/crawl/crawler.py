"""Polite focused crawl loop: feeds and sitemaps -> frontier -> robots check -> fetch -> extract -> store."""
import csv
import logging
import sqlite3
import time
from email.utils import parsedate_to_datetime

import requests

from .. import config, store
from . import extract, frontier as fr
from .robots import RobotsCache
from .urlnorm import host_of, normalise, site_of, skip_reason, trap_reason

log = logging.getLogger("spintrace.crawl")
# Rate limiting, server errors and outright refusals all count towards backoff and the per-host stop.
BLOCKING = {401, 403, 405, 429, 451, 503}


def load_seeds(path):
    """Seed CSV columns: name, kind (original|suspect), type (feed|sitemap|page), url."""
    with open(path) as f:
        return [r for r in csv.DictReader(f) if r.get("url") and not r["name"].startswith("#")]


def retry_after(resp):
    v = resp.headers.get("Retry-After")
    if not v:
        return None
    try:
        return float(v)
    except ValueError:
        try:
            return max(0.0, parsedate_to_datetime(v).timestamp() - time.time())
        except (TypeError, ValueError):
            return None


class Crawler:
    def __init__(self, seeds_path, max_pages=None, url_list=None):
        self.con = store.connect()
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": config.USER_AGENT, "Accept-Language": "en"})
        self.robots = RobotsCache(self.session, self._log_request, self.con)
        self.robots.before_request = lambda host: self.wait_for(host) if host in self.robots.cache else None
        self.frontier = fr.Frontier()
        self.seeds = load_seeds(seeds_path)
        self.max_pages = max_pages
        self.feeds = {}
        self.page_seeds = []
        self.trapped = set()
        # Hosts that seeds, feeds or sitemaps point at; links are followed only within these.
        self.article_hosts = set()
        self.site_kind = {}
        self.last_request = {}
        self.errors = {}
        self.backoff = {}
        self.pages_per_host = {}
        self.last_poll = 0.0
        self.saved = 0
        # Re-crawl mode: fetch exactly the listed article URLs, follow no links, poll no feeds.
        self.url_list = url_list
        if url_list:
            self.seeds = []
            for u, kind in url_list:
                self.site_kind[site_of(host_of(u))] = kind
        for s in self.seeds:
            url = normalise(s["url"])
            self.site_kind[site_of(host_of(url))] = s["kind"]
            if s["type"] in ("feed", "sitemap"):
                self.feeds[url] = s
            else:
                self.page_seeds.append(url)

    def _log_request(self, host, url, event, status=None, detail=None):
        """Every request to a host goes through here so the gap since the previous one is logged."""
        now = time.time()
        gap = now - self.last_request[host] if host in self.last_request else None
        self.last_request[host] = now
        gap_s = f"{gap:.2f}s" if gap is not None else "first"
        store.log_event(self.con, host, url, event, status, f"gap={gap_s} {detail or ''}".strip(), ts=now)
        log.info("%s host=%s status=%s gap=%s %s %s", event.upper(), host, status, gap_s, url, detail or "")

    def wait_for(self, host):
        """Hard politeness guard, independent of the frontier: never contact a host before its delay is over."""
        last = self.last_request.get(host)
        if last is not None:
            entry = self.robots.cache.get(host)
            delay = max(config.DEFAULT_DELAY, float(entry[2] or 0)) if entry else config.DEFAULT_DELAY
            gap = delay - (time.time() - last)
            if gap > 0:
                log.info("GUARD host=%s waiting %.2fs", host, gap)
                time.sleep(gap)

    def _event(self, host, url, event, detail=None):
        store.log_event(self.con, host, url, event, None, detail)
        log.info("%s host=%s %s %s", event.upper(), host, url, detail or "")

    def enqueue(self, url, source, hint_date=None, base=None, kind_site=None):
        url = normalise(url, base)
        if not url:
            return False
        host = host_of(url)
        site = site_of(host)
        if kind_site and site not in self.site_kind:
            # Articles linked from a seed's feed belong to that seed's outlet even on another domain.
            self.site_kind[site] = self.site_kind.get(kind_site, "original")
        if site not in self.site_kind:
            return False
        if source == "link" and host not in self.article_hosts:
            return False
        if source != "link":
            self.article_hosts.add(host)
        if url not in self.feeds:
            if skip_reason(url):
                return False
            reason = trap_reason(url)
            if reason:
                if url not in self.trapped:
                    self.trapped.add(url)
                    self._event(host, url, "trap_guard", reason)
                return False
        prio = fr.priority(url, source)
        if store.add_url(self.con, url, host, prio, source, hint_date):
            self.frontier.add(url, host, prio)
            return True
        return False

    def restore(self):
        for row in self.con.execute("SELECT DISTINCT host FROM urls WHERE source IN ('seed','feed','sitemap')"):
            self.article_hosts.add(row["host"])
        n = 0
        for row in self.con.execute("SELECT url, host, priority FROM urls WHERE status='queued'"):
            if row["host"] in self.article_hosts:
                self.frontier.add(row["url"], row["host"], row["priority"])
                n += 1
        log.info("RESTORE queued=%d hosts=%d", n, len(self.article_hosts))

    def poll_feeds(self):
        """Re-add feeds, sitemaps and seed pages so new articles are discovered while the crawl runs (freshness)."""
        for url in list(self.feeds) + self.page_seeds:
            if not store.add_url(self.con, url, host_of(url), 0, "seed"):
                self.con.execute("UPDATE urls SET status='queued' WHERE url=?", (url,))
            self.frontier.add(url, host_of(url), 0)
        self.con.commit()
        self.last_poll = time.time()

    def host_error(self, host, url, status, resp=None):
        self.errors[host] = self.errors.get(host, 0) + 1
        factor = min(self.backoff.get(host, 1.0) * 2, 64)
        self.backoff[host] = factor
        wait = retry_after(resp) if resp is not None else None
        delay = min(config.MAX_DELAY, max(self.robots.delay(host) * factor, wait or 0))
        self._event(host, url, "backoff", f"status={status} errors={self.errors[host]} next_delay={delay:.0f}s")
        if self.errors[host] >= config.MAX_CONSECUTIVE_ERRORS:
            self.frontier.disable(host)
            store.set_host_state(self.con, host, "stopped", f"{self.errors[host]} consecutive errors, last {status}")
            self._event(host, url, "host_stopped", f"after {self.errors[host]} consecutive errors")
        return delay

    def fetch(self, url, host):
        """One GET without automatic redirects, so a redirect target goes back through robots and politeness."""
        self.wait_for(host)
        try:
            resp = self.session.get(url, timeout=config.REQUEST_TIMEOUT, allow_redirects=False, stream=True)
            body = resp.raw.read(config.MAX_BYTES, decode_content=True)
            resp.close()
        except requests.RequestException as e:
            self._log_request(host, url, "fetch", None, f"error={type(e).__name__}")
            return None, None
        self._log_request(host, url, "fetch", resp.status_code)
        return resp, body

    def handle_feed(self, url, body):
        seed = self.feeds[url]
        entries, children = extract.parse_feed_or_sitemap(body)
        site = site_of(host_of(url))
        added = sum(self.enqueue(link, "feed", date, kind_site=site) for link, date in entries[:300])
        # Sitemap indexes can list years of archives; follow only the newest few children.
        children.sort(key=lambda c: c[1] or 0, reverse=True)
        for child, _ in children[:3]:
            c = normalise(child)
            if c and c not in self.feeds:
                self.feeds[c] = seed
                self.enqueue(c, "sitemap")
        self._event(host_of(url), url, "feed_parsed", f"entries={len(entries)} new={added} children={len(children)}")

    def handle_page(self, url, host, body, hint_date):
        page = extract.extract_page(body, url)
        if page is None:
            return
        site = site_of(host)
        if not self.url_list:
            for link in page["links"]:
                self.enqueue(link, "link", base=url)
        n_words = len(page["text"].split())
        if n_words < config.MIN_ARTICLE_WORDS or not (page["is_article"] or page["published"] or hint_date):
            self._event(host, url, "not_article", f"words={n_words}")
            return
        published, date_source = page["published"], page["date_source"]
        if hint_date and (published is None or date_source == "trafilatura_day"):
            published, date_source = hint_date, "feed"
        doc = {
            "doc_id": store.doc_id_for(url), "url": url, "host": host, "site": site,
            "kind": self.site_kind.get(site, "original"), "origin": "crawl", "title": page["title"],
            "text": page["text"], "published": published, "date_source": date_source, "fetched_at": time.time(),
        }
        store.save_doc(self.con, doc)
        self.saved += 1
        self._event(host, url, "saved", f"doc={doc['doc_id']} words={n_words} date={date_source} title={(page['title'] or '')[:70]!r}")

    def step(self):
        item = self.frontier.next()
        if item is None:
            return False
        ready, host, url = item
        wait = ready - time.time()
        if wait > 0:
            time.sleep(wait)
        cached = host in self.robots.cache
        allowed = self.robots.allowed(url, host)
        delay = self.robots.delay(host) * self.backoff.get(host, 1.0)
        if not cached:
            # The robots.txt request counts as a visit: put the URL back and wait the host delay.
            self.frontier.push_back(host, url)
            self.frontier.done(host, delay)
            return True
        if not allowed:
            store.mark_url(self.con, url, "disallowed")
            self._event(host, url, "robots_skip", "disallowed by robots.txt")
            self.frontier.release(host)
            return True
        if self.pages_per_host.get(host, 0) >= config.MAX_PAGES_PER_HOST and url not in self.feeds:
            store.mark_url(self.con, url, "capped")
            self.frontier.release(host)
            return True
        resp, body = self.fetch(url, host)
        self.pages_per_host[host] = self.pages_per_host.get(host, 0) + 1
        if resp is None:
            delay = self.host_error(host, url, "network")
            store.mark_url(self.con, url, "error", None, "network")
        elif resp.status_code in BLOCKING or resp.status_code >= 500:
            delay = self.host_error(host, url, resp.status_code, resp)
            store.mark_url(self.con, url, "error", resp.status_code)
        else:
            self.errors[host] = 0
            self.backoff[host] = max(1.0, self.backoff.get(host, 1.0) / 2)
            status = resp.status_code
            if status in (301, 302, 303, 307, 308) and resp.headers.get("Location"):
                target = normalise(resp.headers["Location"], url)
                if target in self.feeds or url in self.feeds:
                    if url in self.feeds and target:
                        self.feeds[target] = self.feeds[url]
                    self.enqueue(target, "seed")
                else:
                    hint = self.con.execute("SELECT hint_date, source FROM urls WHERE url=?", (url,)).fetchone()
                    self.enqueue(target, hint["source"] if hint else "link", hint["hint_date"] if hint else None,
                                 kind_site=site_of(host))
                store.mark_url(self.con, url, "redirect", status, target)
            elif status != 200:
                store.mark_url(self.con, url, "http_error", status)
            elif url in self.feeds:
                self.handle_feed(url, body)
                store.mark_url(self.con, url, "done", status)
            elif "html" not in resp.headers.get("Content-Type", "html"):
                store.mark_url(self.con, url, "non_html", status)
            else:
                hint = self.con.execute("SELECT hint_date FROM urls WHERE url=?", (url,)).fetchone()
                try:
                    self.handle_page(url, host, body, hint["hint_date"] if hint else None)
                except Exception as e:
                    log.exception("extract failed %s", url)
                    self._event(host, url, "extract_error", repr(e)[:200])
                store.mark_url(self.con, url, "done", status)
        self.frontier.done(host, delay)
        self.con.commit()
        return True

    def run(self, hours=None):
        self.restore()
        if self.url_list:
            for u, _ in self.url_list:
                self.enqueue(u, "seed")
            self.con.commit()
            while self.step():
                pass
            log.info("RECRAWL done saved=%d", self.saved)
            return
        self.poll_feeds()
        stop_at = time.time() + hours * 3600 if hours else None
        idle = 0
        while True:
            if stop_at and time.time() > stop_at:
                break
            if self.max_pages and self.saved >= self.max_pages:
                break
            if time.time() - self.last_poll > config.FEED_REPOLL:
                self.poll_feeds()
            try:
                progressed = self.step()
            except sqlite3.OperationalError as e:
                # Another process (an evaluation run) holds the database; wait and carry on.
                log.warning("DB_BUSY %s, retrying in 30s", e)
                self.con.rollback()
                time.sleep(30)
                continue
            if not progressed:
                idle += 1
                time.sleep(5)
                if idle % 60 == 0:
                    log.info("IDLE frontier empty, waiting for next feed poll")
            else:
                idle = 0
        log.info("STOP saved=%d", self.saved)
        self.con.commit()


def setup_logging(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(path)
    handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    root = logging.getLogger("spintrace")
    root.setLevel(logging.INFO)
    root.addHandler(handler)
    stream = logging.StreamHandler()
    stream.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    root.addHandler(stream)
