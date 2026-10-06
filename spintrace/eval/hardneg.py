"""Hard negatives: independent articles about the same event from different outlets.

A pair qualifies when both come from original outlets, neither is agency copy, they were published
within three days of each other, and they share at least three names, yet the later one does not
reuse the earlier one's text. Same story is not the same as copied; these pairs test exactly that.
Selection uses names, dates and shingles only, never the SpinTrace verification signals.
"""
import csv
import random
import re
from collections import defaultdict

from .. import config, corpus, dedup, store, text as tx

OUT = config.ROOT / "evaldata" / "hard_negatives.csv"
WIRE = re.compile(r"\((AP|AFP|Reuters|PTI|ANI|IANS|UNI)\)|\b(Associated Press|Reuters|Agence France-Presse|"
                  r"Press Trust of India|PTI|ANI|IANS)\b")
MAX_GAP = 3 * 86400
MIN_SHARED = 3
MAX_CONTAINMENT = 0.1
MAX_NAME_DF = 20
GENERIC = {"monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday", "january", "february",
           "march", "april", "june", "july", "august", "september", "october", "november", "december"}


def mine(con=None, limit=600, seed=358):
    con = con or store.connect()
    feats = corpus.load_features()
    docs = [d for d in store.iter_docs(con, "origin='crawl' AND kind='original' AND published IS NOT NULL")
            if d["doc_id"] in feats and not WIRE.search(d["text"])]
    names = {}
    by_name = defaultdict(set)
    for d in docs:
        n, _ = tx.facts(d["text"])
        n = {x for x in n if (" " in x or len(x) > 5) and x not in GENERIC}
        names[d["doc_id"]] = n
        for x in n:
            by_name[x].add(d["doc_id"])
    # Only rare names identify an event; a name in many articles (a country, a weekday) does not.
    for d_id in names:
        names[d_id] = {x for x in names[d_id] if len(by_name[x]) <= MAX_NAME_DF}
    meta = {d["doc_id"]: d for d in docs}
    pairs = []
    for a in docs:
        counts = defaultdict(int)
        for x in names[a["doc_id"]]:
            for b in by_name[x]:
                counts[b] += 1
        best = None
        for b_id, c in counts.items():
            b = meta[b_id]
            if c < MIN_SHARED or b["site"] == a["site"] or not (0 < a["published"] - b["published"] <= MAX_GAP):
                continue
            if not any(" " in x for x in names[a["doc_id"]] & names[b_id]):
                continue
            cont = dedup.containment(feats[a["doc_id"]]["shingles"], feats[b_id]["shingles"])
            if cont > MAX_CONTAINMENT:
                continue
            if best is None or c > best[1]:
                best = (b_id, c, cont)
        if best:
            pairs.append((a["doc_id"], *best))
    random.Random(seed).shuffle(pairs)
    pairs = pairs[:limit]
    OUT.parent.mkdir(exist_ok=True)
    with OUT.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["suspect", "earlier", "suspect_url", "earlier_url", "shared_names", "containment", "label",
                    "judge1", "judge2"])
        for a, b, c, cont in pairs:
            shared = sorted(names[a] & names[b])[:6]
            w.writerow([a, b, meta[a]["url"], meta[b]["url"], "; ".join(shared), f"{cont:.3f}", "independent(auto)",
                        "", ""])
    return len(pairs)


def load():
    """(suspect doc_id, same-event earlier doc_id) pairs; a judge label of 'derived' removes a pair."""
    if not OUT.exists():
        return []
    out = []
    for r in csv.DictReader(OUT.open()):
        judged = {r.get("judge1", "").strip().lower(), r.get("judge2", "").strip().lower()}
        if "derived" in judged:
            continue
        out.append((r["suspect"], r["earlier"]))
    return out
