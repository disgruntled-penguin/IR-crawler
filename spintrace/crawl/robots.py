"""robots.txt cache. Parsing uses urllib.robotparser; fetching, caching and delay policy are ours."""
import time
import urllib.robotparser

from .. import config


class RobotsCache:
    def __init__(self, session, log, con=None):
        self.session = session
        self.log = log
        self.con = con
        self.cache = {}
        self.before_request = lambda host: None

    def _load(self, host):
        rp = urllib.robotparser.RobotFileParser()
        url = f"https://{host}/robots.txt"
        body = None
        self.before_request(host)
        try:
            r = self.session.get(url, timeout=config.REQUEST_TIMEOUT)
            status = r.status_code
            body = r.text if status < 400 else None
            if status in (401, 403) or status >= 500:
                # Forbidden or a server error: treat the whole host as disallowed until the next check.
                rp.disallow_all = True
            elif status >= 400:
                rp.allow_all = True
            else:
                rp.parse(r.text.splitlines())
        except Exception as e:
            # Unreachable robots.txt: be conservative and treat the host as disallowed for now.
            status = None
            rp.disallow_all = True
            self.log(host, url, "robots_error", None, str(e)[:200])
        delay = rp.crawl_delay(config.ROBOTS_AGENT)
        self.cache[host] = (rp, time.time(), delay)
        if self.con is not None:
            # Kept so the crawl report can re-check every fetched URL against the rules in force.
            self.con.execute("INSERT INTO robots(host, fetched_at, status, body) VALUES (?,?,?,?)",
                             (host, time.time(), status, body if status is not None else "__unreachable__"))
        self.log(host, url, "robots_fetched", status, f"crawl_delay={delay}")
        return rp, delay

    def get(self, host):
        entry = self.cache.get(host)
        if entry and time.time() - entry[1] < config.ROBOTS_TTL:
            return entry[0], entry[2]
        return self._load(host)

    def allowed(self, url, host):
        rp, _ = self.get(host)
        return rp.can_fetch(config.ROBOTS_AGENT, url)

    def delay(self, host):
        _, d = self.get(host)
        return max(config.DEFAULT_DELAY, float(d or 0))
