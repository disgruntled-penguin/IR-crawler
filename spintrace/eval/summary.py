"""Collect the result CSVs into one markdown summary (results/summary.md) for the README and report."""
import csv
import json

from .. import config

R = config.RESULTS


def _rows(name):
    p = R / name
    return list(csv.DictReader(p.open())) if p.exists() else []


def _f(x, d=3):
    try:
        return f"{float(x):.{d}f}"
    except (TypeError, ValueError):
        return "-"


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def build():
    parts = []
    sets = json.loads((R / "eval_sets.json").read_text()) if (R / "eval_sets.json").exists() else {}
    if sets:
        levels = sorted({l for s in sets["counts"].values() for l in s})
        parts.append(f"Index: {sets['index_docs']} documents. Evaluation items (dev / test):\n\n" + table(
            ["set"] + levels, [[s] + [sets["counts"][s].get(l, 0) for l in levels] for s in ("dev", "test")]))

    det = _rows("detection_by_level.csv")
    if det:
        levels = [l for l in ["exact", "light", "synonym", "seo", "summary"] if any(r["level"] == l for r in det)]
        methods = list(dict.fromkeys(r["method"] for r in det))
        rows = []
        for m in methods:
            f = {r["level"]: r for r in det if r["method"] == m}
            rows.append([m] + [_f(f[l]["f1"]) if l in f else "-" for l in levels]
                        + [_f(f.get("fpr_hard_negative", {}).get("f1")), _f(f.get("fpr_facts_only", {}).get("f1"))])
        parts.append("**Is this article derived from another?** Test F1 per rewrite level (negatives pooled), and "
                     "false-positive rate on same-event negatives. Thresholds tuned on dev.\n\n"
                     + table(["method"] + levels + ["FPR hard neg (crawl)", "FPR facts-only"], rows))
        sp = [r for r in det if r["method"] == "spintrace" and r["level"] in levels]
        parts.append("SpinTrace precision / recall / source correct per level:\n\n" + table(
            ["level", "precision", "recall", "F1", "source correct"],
            [[r["level"], _f(r["precision"]), _f(r["recall"]), _f(r["f1"]), _f(r["source_correct"])] for r in sp]))

    rank = _rows("ranking.csv")
    if rank:
        parts.append("**Is its original ranked first?** Test positives, all levels:\n\n" + table(
            ["method", "P@1", "MRR", "Recall@5", "Recall@20"],
            [[r["method"], _f(r["recall@1"]), _f(r["mrr"]), _f(r["recall@5"]), _f(r["recall@20"])]
             for r in rank if r["level"] == "all"]))

    qb = _rows("query_budget.csv")
    if qb:
        parts.append("**Index elimination:** source recall from m query terms chosen by idf (test positives):\n\n" + table(
            ["query", "Recall@1", "Recall@20", "postings touched"],
            [[r["query"], _f(r["recall@1"]), _f(r["recall@20"]), _f(r["postings_touched"], 0)]
             for r in qb if r["level"] == "all"]))

    ret = _rows("retrieval_variants.csv")
    if ret:
        parts.append("Retrieval variants (Recall@1 / Recall@20, all levels):\n\n" + table(
            ["variant", "Recall@1", "Recall@20"],
            [[r["variant"], _f(r["recall@1"]), _f(r["recall@20"])] for r in ret if r["level"] == "all"]))

    abl = _rows("ablations.csv")
    if abl:
        parts.append("**Ablations** (test macro F1 over levels; refit on dev without the signal):\n\n" + table(
            ["variant", "macro F1", "FPR hard neg", "FPR facts-only"],
            [[r["variant"], _f(r["macro_f1"]), _f(r.get("fpr_hard_negative")), _f(r.get("fpr_facts_only"))]
             for r in abl]))

    eff = _rows("efficiency.csv")
    if eff:
        parts.append("**Cost per suspect** (test items):\n\n" + table(
            ["method", "ms per suspect", "docs fully compared", "postings touched", "earlier docs"],
            [[r["method"], _f(r.get("ms_per_suspect"), 1), _f(r.get("docs_compared"), 0),
              _f(r.get("postings_touched"), 0), _f(r.get("earlier_docs"), 0)] for r in eff]))

    und = _rows("undated_provenance.csv")
    if und:
        parts.append("**Provenance without dates** (coverage asymmetry on the known pair, margin 0.2):\n\n" + table(
            ["level", "n", "suspect called derived", "direction reversed", "uncertain"],
            [[r["level"], r["n"], _f(r["suspect_called_derived"]), _f(r["direction_reversed"]), _f(r["uncertain"])]
             for r in und]))

    se = _rows("search_eval.csv")
    if se:
        parts.append("**Do originals outrank copies in search?** Test queries (Wikinews titles), body-only relevance:\n\n"
                     + table(["setting", "lambda", "P@10", "nDCG@10", "original ranked first", "originals among relevant top 10"],
                             [[r["setting"], r["lambda"], _f(r["p@10"]), _f(r["ndcg@10"]), _f(r["original_first"]),
                               _f(r["original_share_top10"])] for r in se]))

    ng = _rows("newsguard_pairs.csv")
    if ng:
        parts.append("**Does it work on real farms?** NewsGuard-documented pairs (Wayback snapshots) in the full index:\n\n"
                     + table(["pair", "status", "rank of original", "p(original)", "flagged", "traced to original"],
                             [[r["pair_id"], r["status"], r.get("original_rank", ""), r.get("prob_original", ""),
                               r.get("flagged", ""), r.get("traced_to_original", "")] for r in ng]))

    from . import judged
    jm = judged.metrics()
    if jm:
        parts.append("**Human judgments** (two team members; filled in `evaldata/*.csv`):\n\n" + table(
            ["sheet", "rows", "judged by both", "agreement", "Cohen's kappa", "share derived (agreed)"],
            [[k, v["rows"], v["judged_by_both"], _f(v["agreement"]), _f(v["kappa"]), _f(v["share_derived_where_agreed"])]
             for k, v in jm.items()]))

    if (R / "crawl_report.json").exists():
        c = json.loads((R / "crawl_report.json").read_text())
        keys = ["requests_page", "requests_robots", "docs_saved", "crawl_hours", "pages_per_minute", "hosts",
                "robots_audited_fetches", "disallowed_urls_fetched", "robots_skips", "min_gap_seconds_overall",
                "delay_violations", "last_delay_violation", "trap_guards_fired", "backoffs", "exact_duplicate_docs",
                "near_duplicate_docs", "duplicate_rate"]
        parts.append("**Is the crawler correct and polite?**\n\n" + table(["measure", "value"], [[k, c.get(k)] for k in keys]))

    text = "# Results\n\nGenerated by `python -m spintrace eval` and friends; every evaluation number below is on the test split.\n\n" \
           + "\n\n".join(parts) + "\n"
    (R / "summary.md").write_text(text)
    update_readme(text)
    return text


START, END = "<!-- results:start -->", "<!-- results:end -->"


def update_readme(text):
    """Keep the README's results block identical to results/summary.md."""
    readme = config.ROOT / "README.md"
    body = readme.read_text()
    if START not in body or END not in body:
        return
    block = text.split("\n", 2)[2] if text.startswith("# Results") else text
    head, rest = body.split(START, 1)
    _, tail = rest.split(END, 1)
    readme.write_text(head + START + "\n" + block.strip() + "\n" + END + tail)
