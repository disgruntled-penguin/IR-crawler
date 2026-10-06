"""robots.txt cache. Parsing uses urllib.robotparser; fetching, caching and delay policy are ours."""
import time
import urllib.robotparser

from .. import config


class RobotsCache:
    def __init__(self, session, log):
        self.session = session
        self.log = log
        self.cache = {}

    def _load(self, host):
        rp = urllib.robotparser.RobotFileParser()
        url = f"https://{host}/robots.txt"
        try:
            r = self.session.get(url, timeout=config.REQUEST_TIMEOUT)
            status = r.status_code
            if status in (401, 403):
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
