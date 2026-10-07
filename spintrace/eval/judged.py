"""Human judgments: an interactive labelling loop and the metrics computed from the labels.

Sheets: evaldata/live_flagged.csv (pairs SpinTrace flagged on the live crawl), evaldata/hard_negatives.csv (mined
same-event pairs), evaldata/search_judgments.csv (live search results, label relevant / not).
Labels: d = derived, n = not derived (r / x for search relevance).
"""
import csv
import textwrap
import threading

from .. import config, store

SHEETS = {
    "live": config.ROOT / "evaldata" / "live_flagged.csv",
    "hardneg": config.ROOT / "evaldata" / "hard_negatives.csv",
    "search": config.ROOT / "evaldata" / "search_judgments.csv",
}
PAIR_KEYS = {"live": ("suspect", "source"), "hardneg": ("suspect", "earlier")}
_lock = threading.Lock()
POSITIVE = {"live": "derived", "hardneg": "derived", "search": "relevant"}
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
            set_label(sheet, rows.index(r), judge, LABELS[ans])


def read_rows(sheet):
    return list(csv.DictReader(SHEETS[sheet].open()))


def set_label(sheet, i, judge, value):
    """Write one judge's label for row i, re-reading the sheet so the other judge's labels are kept."""
    with _lock:
        rows = read_rows(sheet)
        rows[i][f"judge{judge}"] = value
        with SHEETS[sheet].open("w", newline="") as f:
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


def is_positive(row, sheet, rule="any"):
    """One verdict per row from the two judges, None if nobody judged it.

    Rule "any" is the team's convention: the positive label (derived, relevant) wins if either judge gave it.
    Rule "both" is the strict reading: positive only if both judges gave it.
    """
    pos = POSITIVE[sheet]
    got = [row[c] for c in ("judge1", "judge2") if row.get(c)]
    if not got:
        return None
    return (pos in got) if rule == "any" else (len(got) == 2 and all(g == pos for g in got))


def metrics():
    """Agreement and the share of positives under each way of combining the judges, from the labels so far."""
    out = {}
    for sheet in ("live", "hardneg", "search"):
        path = SHEETS[sheet]
        if not path.exists():
            continue
        positive = POSITIVE[sheet]
        rows = list(csv.DictReader(path.open()))
        j1 = [r.get("judge1", "") for r in rows]
        j2 = [r.get("judge2", "") for r in rows]
        both = [(x, y) for x, y in zip(j1, j2) if x and y]
        agreed = [x for x, y in both if x == y]
        judged_rows = [r for r in rows if is_positive(r, sheet) is not None]
        share = lambda rule: (sum(is_positive(r, sheet, rule) for r in judged_rows) / len(judged_rows)) if judged_rows else None
        out[sheet] = {"rows": len(rows), "judged_by_both": len(both), "agreement": (len(agreed) / len(both)) if both else None,
                      "kappa": kappa(j1, j2), "share_derived_where_agreed":
                      (sum(x == positive for x in agreed) / len(agreed)) if agreed else None,
                      "positive_any": share("any"), "positive_both": share("both"),
                      "judge1_positive": (j1.count(positive) / sum(bool(x) for x in j1)) if any(j1) else None,
                      "judge2_positive": (j2.count(positive) / sum(bool(x) for x in j2)) if any(j2) else None}
    if "live" in out:
        out["live"]["precision_of_flags"] = out["live"]["positive_any"]
    return out
