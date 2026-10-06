import random

from spintrace.crawl import frontier as fr


def test_no_host_contacted_before_its_delay(monkeypatch):
    clock = [1000.0]
    monkeypatch.setattr(fr.time, "time", lambda: clock[0])
    f = fr.Frontier()
    rng = random.Random(1)
    hosts = [f"h{i}.example" for i in range(7)]
    for i in range(3000):
        h = rng.choice(hosts)
        f.add(f"https://{h}/a{i}", h, rng.randint(0, 2))
    last = {}
    while True:
        item = f.next()
        if item is None:
            break
        ready, host, _ = item
        clock[0] = max(clock[0], ready) + rng.uniform(0, 0.5)
        if host in last:
            assert clock[0] - last[host] >= 5.0 - 1e-9, (host, clock[0] - last[host])
        last[host] = clock[0]
        if rng.random() < 0.05:
            f.release(host)
            last.pop(host)
        else:
            f.done(host, 5.0)
        # Occasionally new URLs arrive mid-crawl, as links and feed entries do.
        if rng.random() < 0.1:
            h = rng.choice(hosts)
            f.add(f"https://{h}/n{clock[0]}", h, 0)
    assert len(f) == 0


def test_no_host_starves_behind_a_prolific_one(monkeypatch):
    clock = [1000.0]
    monkeypatch.setattr(fr.time, "time", lambda: clock[0])
    f = fr.Frontier()
    for i in range(5000):
        f.add(f"https://big.example/a{i}", "big.example", 1)
    quiet = [f"q{i}.example" for i in range(10)]
    # Same priority, queued behind the prolific host's backlog, as when one site's links flood the frontier.
    for h in quiet:
        for i in range(3):
            f.add(f"https://{h}/a{i}", h, 1)
    served = set()
    for _ in range(60):
        ready, host, _ = f.next()
        clock[0] = max(clock[0], ready) + 0.1
        served.add(host)
        f.done(host, 5.0)
    assert set(quiet) <= served


def test_priority_orders_a_hosts_own_requests():
    f = fr.Frontier()
    f.not_before = 0
    for i in range(50):
        f.add(f"https://h.example/low{i}", "h.example", 2)
    f.add("https://h.example/feed-entry", "h.example", 0)
    picks = []
    for _ in range(5):
        _, host, url = f.next()
        picks.append(url)
        f.release(host)
    assert "https://h.example/feed-entry" in picks
