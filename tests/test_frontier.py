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
