"""Human judgments: an interactive labelling loop and the metrics computed from the labels.

Sheets: evaldata/live_flagged.csv (pairs SpinTrace flagged on the live crawl), evaldata/hard_negatives.csv (mined
same-event pairs), evaldata/search_judgments.csv (live search results, label relevant / not).
Labels: d = derived, n = not derived (r / x for search relevance).
"""
import csv
import textwrap

from .. import config, store

SHEETS = {
    "live": config.ROOT / "evaldata" / "live_flagged.csv",
    "hardneg": config.ROOT / "evaldata" / "hard_negatives.csv",
    "search": config.ROOT / "evaldata" / "search_judgments.csv",
}
PAIR_KEYS = {"live": ("suspect", "source"), "hardneg": ("suspect", "earlier")}
LABELS = {"d": "derived", "n": "not_derived", "r": "relevant", "x": "not_relevant"}


def _show_pair(con, a, b):
    for tag, d in (("SUSPECT", a), ("EARLIER", b)):
        r = con.execute("SELECT url, title, published, text FROM docs WHERE doc_id=?", (d,)).fetchone()
        if not r:
            print(f"{tag}: {d} not in local store (run recrawl)")
            continue
        print(f"{tag}: {r['title']}\n  {r['url']}\n  " + "\n  ".join(textwrap.wrap(r["text"][:900], 110)))


def label(sheet, judge):
    """Ask for a label on every row this judge has not labelled yet; saves after each answer."""
    path = SHEETS[sheet]
    rows = list(csv.DictReader(path.open()))
    col = f"judge{judge}"
    con = store.connect()
    todo = [r for r in rows if not r.get(col)]
    print(f"{len(todo)} rows to label in {path.name} as {col}. Keys: "
          + ("r relevant, x not relevant" if sheet == "search" else "d derived, n not derived") + ", s skip, q quit")
    for i, r in enumerate(todo, 1):
        print("=" * 110 + f"\n[{i}/{len(todo)}]")
        if sheet == "search":
            print(f"QUERY: {r['query']}")
            _show_pair(con, r["doc_id"], r["doc_id"])
        else:
            a, b = (r[k] for k in PAIR_KEYS[sheet])
            _show_pair(con, a, b)
        ans = input("label> ").strip().lower()
        if ans == "q":
            break
        if ans in LABELS:
            r[col] = LABELS[ans]
            with path.open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)


def kappa(a, b):
    """Cohen's kappa between two judges over the items both labelled."""
    pairs = [(x, y) for x, y in zip(a, b) if x and y]
    if not pairs:
        return None
    labels = sorted({x for p in pairs for x in p})
    po = sum(x == y for x, y in pairs) / len(pairs)
    pe = sum((sum(x == l for x, _ in pairs) / len(pairs)) * (sum(y == l for _, y in pairs) / len(pairs)) for l in labels)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def metrics():
    """Precision of live flags and agreement, from whatever has been judged so far."""
    out = {}
    for sheet, positive in (("live", "derived"), ("hardneg", "derived")):
        path = SHEETS[sheet]
        if not path.exists():
            continue
        rows = list(csv.DictReader(path.open()))
        j1 = [r.get("judge1", "") for r in rows]
        j2 = [r.get("judge2", "") for r in rows]
        both = [(x, y) for x, y in zip(j1, j2) if x and y]
        agreed = [x for x, y in both if x == y]
        out[sheet] = {"rows": len(rows), "judged_by_both": len(both), "agreement": (len(agreed) / len(both)) if both else None,
                      "kappa": kappa(j1, j2), "share_derived_where_agreed":
                      (sum(x == positive for x in agreed) / len(agreed)) if agreed else None}
    if "live" in out:
        out["live"]["precision_of_flags"] = out["live"]["share_derived_where_agreed"]
    return out
