"""URL frontier in the Mercator style, with the front queues partitioned by host.

Each host keeps one queue per priority level (the front queues); a heap orders hosts by the earliest
time they may be contacted again, with at most one heap entry per host (the back-queue selector).
When a host's turn comes, its next URL is drawn from its priority queues with a bias towards high
priority. Because every host is fetched as often as its delay allows, priority decides what each
host's next request is, and no host waits for another host's backlog. The priority of a URL comes from
priority(), the single function to replace for a different crawl policy.
"""
import heapq
import random
import re
import time
from collections import deque

from .. import config

N_PRIORITIES = 3
FRONT_BIAS = [8, 3, 1]
ARTICLE_PATH = re.compile(r"/(19|20)\d\d/|/\d{6,}|[a-z0-9]+(-[a-z0-9]+){3,}", re.I)


def priority(url, source):
    """0 is most urgent. Feed and sitemap entries first, article-shaped links next, everything else last."""
    if source in ("feed", "sitemap", "seed"):
        return 0
    if ARTICLE_PATH.search(url):
        return 1
    return 2


class Frontier:
    def __init__(self):
        self.queues = {}
        self.heap = []
        self.scheduled = set()
        self.in_flight = set()
        self.next_time = {}
        self.disabled = set()
        self.rng = random.Random(358)
        # After a restart, no host is contacted until one full delay has passed.
        self.not_before = time.time() + config.DEFAULT_DELAY

    def __len__(self):
        return sum(len(q) for qs in self.queues.values() for q in qs)

    def _pending(self, host):
        return any(self.queues.get(host, ()))

    def add(self, url, host, prio):
        if host in self.disabled:
            return
        qs = self.queues.setdefault(host, [deque() for _ in range(N_PRIORITIES)])
        qs[min(prio, N_PRIORITIES - 1)].append(url)
        self._schedule(host)

    def _schedule(self, host):
        # A host being fetched is rescheduled only by done()/release(), once its next time is known.
        if host not in self.scheduled and host not in self.in_flight and self._pending(host):
            heapq.heappush(self.heap, (self.next_time.get(host, self.not_before), host))
            self.scheduled.add(host)

    def _pick(self, host):
        qs = self.queues[host]
        live = [i for i in range(N_PRIORITIES) if qs[i]]
        i = self.rng.choices(live, weights=[FRONT_BIAS[j] for j in live])[0]
        return qs[i].popleft()

    def next(self):
        """Pop the URL of the host that may be contacted soonest; returns (ready_time, host, url) or None."""
        while self.heap:
            t, host = heapq.heappop(self.heap)
            self.scheduled.discard(host)
            if host in self.disabled or not self._pending(host):
                continue
            if t < self.next_time.get(host, t):
                self._schedule(host)
                continue
            self.in_flight.add(host)
            return t, host, self._pick(host)
        return None

    def push_back(self, host, url):
        """Return a URL to the head of its host's most urgent queue (used when robots.txt had to be fetched first)."""
        qs = self.queues.setdefault(host, [deque() for _ in range(N_PRIORITIES)])
        qs[0].appendleft(url)

    def done(self, host, delay):
        """A request was sent to host: it may not be contacted again for delay seconds."""
        self.next_time[host] = time.time() + delay
        self.in_flight.discard(host)
        self._schedule(host)

    def release(self, host):
        """No request was sent (URL skipped before fetching): keep the host's slot unchanged."""
        self.in_flight.discard(host)
        self._schedule(host)

    def disable(self, host):
        self.disabled.add(host)
        self.queues.pop(host, None)
