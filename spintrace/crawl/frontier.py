"""Mercator-style URL frontier.

Front queues hold URLs by priority; each host has one back queue; a heap orders hosts by the
earliest time they may be contacted again, with at most one heap entry per host. The priority of
a URL comes from priority(), the single function to replace for a different crawl policy.
"""
import heapq
import random
import re
import time
from collections import deque

N_PRIORITIES = 3
FRONT_BIAS = [8, 3, 1]
BACK_QUEUE_CAP = 50
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
        self.front = [deque() for _ in range(N_PRIORITIES)]
        self.back = {}
        self.heap = []
        self.scheduled = set()
        self.next_time = {}
        self.disabled = set()

    def __len__(self):
        return sum(len(q) for q in self.front) + sum(len(q) for q in self.back.values())

    def add(self, url, host, prio):
        if host not in self.disabled:
            self.front[min(prio, N_PRIORITIES - 1)].append((url, host))

    def _schedule(self, host):
        if host not in self.scheduled and self.back.get(host):
            heapq.heappush(self.heap, (self.next_time.get(host, 0.0), host))
            self.scheduled.add(host)

    def _refill(self, budget=500):
        """Move URLs from front to back queues, picking front queues with a bias towards high priority."""
        deferred = [[] for _ in range(N_PRIORITIES)]
        moved = examined = 0
        while moved < budget and examined < 5 * budget:
            live = [i for i in range(N_PRIORITIES) if self.front[i]]
            if not live:
                break
            i = random.choices(live, weights=[FRONT_BIAS[j] for j in live])[0]
            url, host = self.front[i].popleft()
            examined += 1
            if host in self.disabled:
                continue
            q = self.back.setdefault(host, deque())
            if len(q) >= BACK_QUEUE_CAP:
                deferred[i].append((url, host))
                continue
            q.append(url)
            moved += 1
            self._schedule(host)
        for i in range(N_PRIORITIES):
            self.front[i].extendleft(reversed(deferred[i]))

    def next(self):
        """Pop the URL of the host that may be contacted soonest; returns (ready_time, host, url) or None."""
        if len(self.heap) < 3:
            self._refill()
        while self.heap:
            t, host = heapq.heappop(self.heap)
            self.scheduled.discard(host)
            q = self.back.get(host)
            if host in self.disabled or not q:
                continue
            url = q.popleft()
            if not q:
                self._refill(budget=100)
            return t, host, url
        return None

    def done(self, host, delay):
        """A request was sent to host: it may not be contacted again for delay seconds."""
        self.next_time[host] = time.time() + delay
        self._schedule(host)

    def release(self, host):
        """No request was sent (URL skipped before fetching): keep the host's slot unchanged."""
        self._schedule(host)

    def disable(self, host):
        self.disabled.add(host)
        self.back.pop(host, None)
