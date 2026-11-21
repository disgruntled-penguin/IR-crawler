"""Write the final-phase tables as LaTeX from results/*.csv: python report/make_tables.py [results_dir]

Each table lands in report/tables/<name>.tex as a tabularx environment in the report's style; the report
\\input's it inside its own table float, so the numbers always come from the result files.
"""
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results"
OUT = Path(__file__).resolve().parent / "tables"

XL_METHODS = [
    ("spintrace_xl", "SpinTrace (anchors, no translation)"),
    ("bt_spintrace_en", "Translate back, then SpinTrace"),
    ("bt_tfidf_cosine", "Translate back, then tf-idf"),
    ("bt_bm25", "Translate back, then BM25"),
    ("bt_minhash_jaccard", "Translate back, then MinHash"),
    ("bt_exact_hash", "Translate back, then exact hash"),
    ("dense_cosine", "Multilingual dense cosine"),
    ("latin_only", "Latin-script tokens only"),
]
MS_METHODS = [
    ("spintrace", "SpinTrace multi-source"),
    ("containment", "Shingle containment"),
    ("minhash", "MinHash LSH"),
    ("midterm", "Midterm single source"),
    ("oneshot", "One-shot multi"),
]


def rows(name):
    path = RESULTS / name
    if not path.exists():
        return []
    with path.open() as f:
        return list(csv.DictReader(f))


def num(x, d=3):
    try:
        return f"{float(x):.{d}f}"
    except (TypeError, ValueError):
        return "--"


def find(rs, **kw):
    return next((r for r in rs if all(r.get(k) == v for k, v in kw.items())), None)


def write(name, header, body, spec):
    OUT.mkdir(exist_ok=True)
    lines = [f"\\begin{{tabularx}}{{\\textwidth}}{{{spec}}}", "  \\toprule", "  \\hrow " + " & ".join(f"\\hd{{{h}}}" for h in header) + " \\\\",
             "  \\midrule"]
    for i, cells in enumerate(body):
        lines.append("  " + ("\\ours " if i == 0 else "") + " & ".join(cells) + " \\\\")
    lines += ["  \\bottomrule", "\\end{tabularx}"]
    (OUT / f"{name}.tex").write_text("\n".join(lines) + "\n")


def xl_detection():
    det, rank = rows("crosslingual_detection.csv"), rows("crosslingual_ranking.csv")
    body = []
    for key, label in XL_METHODS:
        cells = [label]
        for level in ("hi_translation", "hi_seo", "hi_summary"):
            r = find(det, method=key, level=level)
            cells.append(num(r["f1"]) if r else "--")
        for neg in ("hi_facts_only", "hi_facts_short", "source_absent"):
            r = find(det, method=key, level="fpr_" + neg)
            cells.append(num(r["f1"]) if r else "--")
        r = find(rank, method=key, level="all_hi")
        cells.append(num(r["p@1"]) if r else "--")
        body.append(cells)
    write("xl_detection", ["Method", "transl.", "SEO", "summary", "FPR same event", "FPR same event, 100 words",
                           "FPR source absent", "P@1"], body, "@{}l" + "R" * 7 + "@{}")


def xl_ablation():
    ab = rows("crosslingual_ablations.csv")
    names = {"full": "Full system", "no_names": "No name anchors", "no_numbers": "No number anchors",
             "no_dates": "No date anchors", "key_exact": "Exact phonetic key only", "translit_exact":
             "Exact transliteration only (no fuzzy k-gram)", "no_idf_check": "No source-language df check",
             "window_30d": "30-day window instead of 7", "dense_verify": "Dense retrieval, our verifier",
             "no_alignment_signals": "No alignment signals", "plus_anchor_signals": "Anchor-overlap signals added",
             "no_retrieval_score": "No retrieval score"}
    body = []
    for r in ab:
        body.append([names.get(r["variant"], r["variant"]), num(r["macro_f1"]), num(r["p@1"]),
                     num(r["fpr_hi_facts_only"]), num(r["postings_touched"], 0)])
    body.sort(key=lambda c: c[0] != "Full system")
    write("xl_ablation", ["Variant (refit on dev)", "macro F1", "P@1", "FPR same event", "postings touched"], body, "@{}l" + "R" * 4 + "@{}")


def ms_detection():
    det = rows("multisource_detection.csv")
    groups = [("splice (all)", "all splices"), ("splice splice", "plain"), ("splice splice_light", "light edit"),
              ("splice splice_synonym", "synonyms"), ("splice splice_seo", "SEO rewrite"),
              ("LLM mashups (both sources test)", "LLM mashups")]
    body = []
    for key, label in MS_METHODS:
        cells = [label]
        for g, _ in groups:
            r = find(det, method=key, group=g)
            cells.append(num(r["f1"]) if r else "--")
        body.append(cells)
    write("ms_detection", ["Method"] + [g[1] for g in groups], body, "@{}l" + "R" * len(groups) + "@{}")


def ms_by_rank():
    rk = rows("multisource_by_rank.csv")
    cats = [("2", "1", "2 src: largest"), ("2", "2", "2 src: second"), ("3", "1", "3 src: largest"),
            ("3", "2", "3 src: second"), ("3", "3", "3 src: third")]
    body = []
    for key, label in MS_METHODS[:4]:
        cells = [label]
        for n, k, _ in cats:
            r = find(rk, method=key, n_sources=n, source_rank=k)
            cells.append(num(r["recall"]) if r else "--")
        body.append(cells)
    write("ms_by_rank", ["Method"] + [c[2] for c in cats], body, "@{}l" + "R" * len(cats) + "@{}")


def ms_ablation():
    ab = rows("multisource_ablations.csv")
    names = {"full": "Full (explain-away rounds)", "no_residual": "No residual query (one query)",
             "no_facts_quotes": "Residual without names, numbers, quotes",
             "midterm_verifier_in_rounds": "Midterm whole-document verifier in the rounds"}
    body = [[names.get(r["variant"], r["variant"]), num(r["F1 splice (all)"]), num(r["F1 splice splice_seo"]),
             num(r["recall 2 src rank 2"]), num(r["recall 3 src rank 3"]), num(r["facts_only false_source_rate"])]
            for r in ab]
    body.sort(key=lambda c: c[0] != "Full (explain-away rounds)")
    write("ms_ablation", ["Variant (retuned on dev)", "F1 splices", "F1 SEO splices", "2nd of 2", "3rd of 3",
                          "false source, same story"], body, "@{}l" + "R" * 5 + "@{}")


def ms_controls():
    ct = rows("multisource_controls.csv")
    body = []
    for key, label in MS_METHODS[:4]:
        cells = [label]
        for level in ("light", "synonym", "seo", "summary"):
            r = find(ct, method=key, level=level)
            cells.append(num(r["extra_source_rate"]) if r else "--")
        for level in ("facts_only", "hard_negative"):
            r = find(ct, method=key, level=level)
            cells.append(num(r["false_source_rate"]) if r else "--")
        body.append(cells)
    write("ms_controls", ["Method", "extra: light", "extra: synonym", "extra: SEO", "extra: summary",
                          "false: same story", "false: hard neg."], body, "@{}l" + "R" * 6 + "@{}")


def dataset():
    path = RESULTS / "dataset.json"
    if not path.exists():
        return
    d = json.loads(path.read_text())
    o, f = d["origins"], d["files"]
    n = lambda k: f"{o.get(k, {}).get('n', 0):,}"
    scripts = d["crawl_scripts"]
    indic = sum(v for k, v in scripts.items() if k != "latn")
    rx = f.get("rewrites_xl.jsonl", {}).get("levels", {})
    gen = sum(v for k, v in rx.items() if not k.startswith("bt:") and k != "mash_llm")
    body = [
        ["Live crawl (ours)", f"{n('crawl')} articles, {d['crawl_sites']} sites, {indic:,} in Indian scripts",
         "RSS and Atom feeds and homepages, robots.txt obeyed, Oct to Nov 2026", "real inputs, live detection, hard negatives, search"],
        ["English Wikinews dump", f"{n('wikinews')} originals, {n('wikinews_distractor')} distractors",
         "CC BY 2.5, Wikinews contributors", "originals for every synthetic rewrite"],
        ["English rewrites (ours)", f"{f.get('rewrites.jsonl', {}).get('rows', 0):,}",
         "rule-based and Llama 3.2 3B, from Wikinews", "graded rewrite levels and same-story negatives"],
        ["Indian-language rewrites (ours)", f"{gen:,}, plus {rx.get('mash_llm', 0)} LLM mashups",
         "Gemma 3 4B, from Wikinews", "Hindi levels, Tamil, Telugu, Bengali, mashups"],
        ["Splice mashups and formats (ours)", f"{f.get('mashups.jsonl', {}).get('rows', 0):,} and {f.get('formats.jsonl', {}).get('rows', 0):,}",
         "rule-based, from Wikinews", "exact source shares, listicle and Q\\&A"],
        ["CC-NEWS (Common Crawl)", f"{n('ccnews')} articles", "one WARC file, 6 Oct 2026", "scale check"],
        ["NewsGuard documented pairs", f"{n('newsguard')} pages", "Wayback Machine captures", "real farm cases"],
        ["Hard negatives and judged labels", f"{f.get('hard_negatives.csv', {}).get('rows', 0)} pairs, "
         f"{f.get('live_flagged.csv', {}).get('rows', 0)} live pairs, {f.get('search_judgments.csv', {}).get('rows', 0)} results",
         "mined, then judged by both authors (midterm)", "same-event tests, live precision, search"],
    ]
    write("dataset", ["Data", "Size", "Source and licence", "Used for"], body, "@{}P{3.1cm}P{4.2cm}P{4.6cm}L@{}")


if __name__ == "__main__":
    for fn in (xl_detection, xl_ablation, ms_detection, ms_by_rank, ms_ablation, ms_controls, dataset):
        fn()
    print("tables in", OUT)
