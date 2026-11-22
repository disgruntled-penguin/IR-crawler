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


def xl_other():
    det, rank = rows("crosslingual_detection.csv"), rows("crosslingual_ranking.csv")
    langs = [("ta_translation", "Tamil"), ("te_translation", "Telugu"), ("bn_translation", "Bengali"),
             ("hi_translation", "Hindi, for comparison")]
    spec = [("P@1, SpinTrace anchors", "spintrace_xl", "rank"), ("P@1, multilingual dense", "dense_cosine", "rank"),
            ("P@1, Latin tokens only", "latin_only", "rank"), ("F1, SpinTrace verified as derived", "spintrace_xl", "det")]
    body = []
    for label, key, kind in spec:
        cells = [label]
        for lv, _ in langs:
            r = find(rank, method=key, level=lv) if kind == "rank" else find(det, method=key, level=lv)
            cells.append(num(r["p@1" if kind == "rank" else "f1"]) if r else "--")
        body.append(cells)
    write("xl_other", ["Measure"] + [l for _, l in langs], body, "@{}l" + "R" * len(langs) + "@{}")


def xlm_tables():
    det, ctrl = rows("xlmulti_detection.csv"), rows("xlmulti_controls.csv")
    names = [("passage tracing", "Passage tracing (blocks, merged)"), ("whole text, one source (dev threshold)",
             "Whole text, one source (dev threshold)"), ("whole text, shipped threshold", "Whole text, one source (shipped threshold)")]
    body = []
    for key, label in names:
        cells = [label]
        for st in ("all", "mash_translation", "mash_seo"):
            r = find(det, method=key, set=st)
            cells += [num(r["precision"]), num(r["recall"]), num(r["f1"])] if r else ["--"] * 3
        r = find(det, method=key, set="all")
        cells.append(num(r["exact_set"]) if r else "--")
        body.append(cells)
    write("xlm_detection", ["Method", "P", "R", "F1", "P transl.", "R transl.", "F1 transl.", "P SEO", "R SEO", "F1 SEO",
                            "exact set"], body, "@{}l" + "R" * 10 + "@{}")
    body = []
    for key, label in names:
        cells = [label]
        for lv in ("hi_translation", "hi_seo", "hi_summary"):
            r = find(ctrl, method=key, level=lv)
            cells.append(num(r["extra_source_rate"]) if r else "--")
        for lv in ("hi_summary",):
            r = find(ctrl, method=key, level=lv)
            cells.append(num(r["source_found"]) if r else "--")
        for lv in ("hi_facts_only", "hi_facts_short"):
            r = find(ctrl, method=key, level=lv)
            cells.append(num(r["false_source_rate"]) if r else "--")
        body.append(cells)
    write("xlm_controls", ["Method", "extra: transl.", "extra: SEO", "extra: summary", "summary found",
                           "false: same event", "false: 100 words"], body, "@{}l" + "R" * 6 + "@{}")
    N = {}
    for key, tag in (("passage tracing", "Xlm"), ("whole text, shipped threshold", "XlmWhole")):
        r = find(det, method=key, set="all")
        N[tag + "F"], N[tag + "Recall"], N[tag + "Prec"] = (num(r["f1"]), num(r["recall"]), num(r["precision"])) if r else ("--",) * 3
        r = find(ctrl, method=key, level="hi_facts_only")
        N[tag + "FalseSame"] = num(r["false_source_rate"]) if r else "--"
        r = find(ctrl, method=key, level="hi_facts_short")
        N[tag + "FalseShort"] = num(r["false_source_rate"]) if r else "--"
        r = find(ctrl, method=key, level="hi_summary")
        N[tag + "SummaryFound"] = num(r["source_found"]) if r else "--"
    r = find(det, method="passage tracing", set="all")
    N["XlmExact"], N["XlmN"] = (num(r["exact_set"]), r["n"]) if r else ("--", "--")
    pj = RESULTS / "xlmulti_params.json"
    if pj.exists():
        pr = json.loads(pj.read_text())
        N["XlmBlock"] = str(pr["methods"]["passage tracing"]["block_size"])
        N["XlmMs"] = num(pr["cost_ms"]["ms_blocks"], 0)
    extra = "".join(f"\\newcommand{{\\n{k}}}{{{v}}}\n" for k, v in sorted(N.items()))
    OUT.mkdir(exist_ok=True)
    with (OUT / "numbers.tex").open("a") as f:
        f.write(extra)


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


def numbers():
    """LaTeX macros for every number the report's prose cites, so the prose cannot drift from the results."""
    det, rank = rows("crosslingual_detection.csv"), rows("crosslingual_ranking.csv")
    ab, eff = rows("crosslingual_ablations.csv"), rows("crosslingual_efficiency.csv")
    msd, msr = rows("multisource_detection.csv"), rows("multisource_by_rank.csv")
    msa, msc, cost = rows("multisource_ablations.csv"), rows("multisource_controls.csv"), rows("multisource_cost.csv")
    N = {}

    def d(method, level):
        r = find(det, method=method, level=level)
        return num(r["f1"]) if r else "--"

    def r1(method):
        r = find(rank, method=method, level="all_hi")
        return num(r["p@1"]) if r else "--"

    def a(variant, col="macro_f1"):
        r = find(ab, variant=variant)
        return num(r[col]) if r else "--"

    def g(method, group):
        r = find(msd, method=method, group=group)
        return num(r["f1"]) if r else "--"

    def k(method, n, rk):
        r = find(msr, method=method, n_sources=n, source_rank=rk)
        return num(r["recall"]) if r else "--"

    for key, m in (("XlTrans", "hi_translation"), ("XlSeo", "hi_seo"), ("XlSum", "hi_summary")):
        N[key] = d("spintrace_xl", m)
        N["BtSpin" + key[2:]] = d("bt_spintrace_en", m)
    N["DenseTrans"], N["DenseSeo"], N["DenseSum"] = (d("dense_cosine", m) for m in ("hi_translation", "hi_seo", "hi_summary"))
    N["LatinSum"] = d("latin_only", "hi_summary")
    N["BtTfidfSum"], N["BtBmSum"] = d("bt_tfidf_cosine", "hi_summary"), d("bt_bm25", "hi_summary")
    N["BtTfidfTrans"], N["BtTfidfSeo"] = d("bt_tfidf_cosine", "hi_translation"), d("bt_tfidf_cosine", "hi_seo")
    N["BtBmTrans"], N["BtBmSeo"] = d("bt_bm25", "hi_translation"), d("bt_bm25", "hi_seo")
    N["BtTfidfPone"] = r1("bt_tfidf_cosine")
    align_path = RESULTS / "crosslingual_align_check.json"
    if align_path.exists():
        ac = json.loads(align_path.read_text())
        for key, lv in (("Hi", "hi_translation"), ("Ta", "ta_translation"), ("Te", "te_translation"), ("Bn", "bn_translation")):
            N["Align" + key] = num(ac[lv]["align_mean"], 2)
            N["AlignCov" + key] = num(ac[lv]["align_coverage"], 2)
    length_path = RESULTS / "crosslingual_length_check.json"
    if length_path.exists():
        lc = json.loads(length_path.read_text())
        N["LenSumMed"] = num(lc["median_words"]["test:hi_summary"], 0)
        N["LenLongMed"] = num(lc["median_words"]["test:hi_facts_only"], 0)
        N["LenShortMed"] = num(lc["median_words"]["test:hi_facts_short"], 0)
        N["LenAucSum"] = num(lc["auc"]["test:hi_summary:vs_long"], 2)
        N["LenAucTrans"] = num(lc["auc"]["test:hi_translation:vs_long"], 2)
        N["LenAucSeo"] = num(1 - lc["auc"]["test:hi_seo:vs_long"], 2)
        N["LenAucSumShort"] = num(lc["auc"]["test:hi_summary:vs_short"], 2)
    crawl_path = RESULTS / "crawl_report.json"
    if crawl_path.exists():
        cr = json.loads(crawl_path.read_text())
        fc = cr.get("final_crawl")
        if fc:
            for key, v in (("Req", "requests_page"), ("Robots", "requests_robots"), ("Hosts", "hosts"), ("Skips", "robots_skips"),
                           ("Back", "backoffs"), ("Trap", "trap_guards_fired"), ("Docs", "docs_saved"),
                           ("Disallowed", "disallowed_urls_fetched"), ("Under", "requests_under_floor")):
                N["Crawl" + key] = f"{fc[v]:,}".replace(",", "{,}")
            N["CrawlGap"] = num(fc["min_gap_seconds"], 1)
        N["StoreCrawl"] = f"{cr['docs_saved']:,}".replace(",", "{,}")
        N["CrawlAudited"] = f"{cr['robots_audited_fetches']:,}".replace(",", "{,}")
        N["CrawlAllReq"] = f"{cr['requests_page']:,}".replace(",", "{,}")
    N["XlPone"], N["DensePone"], N["BtSpinPone"] = r1("spintrace_xl"), r1("dense_cosine"), r1("bt_spintrace_en")
    for key, lvl in (("Same", "fpr_hi_facts_only"), ("Short", "fpr_hi_facts_short"), ("Absent", "fpr_source_absent")):
        N["XlFpr" + key] = d("spintrace_xl", lvl)
        N["DenseFpr" + key] = d("dense_cosine", lvl)
        N["BtSpinFpr" + key] = d("bt_spintrace_en", lvl)
    e = find(eff, method="spintrace_xl")
    b = find(eff, method="bt_spintrace_en")
    N["XlMs"] = num(e["ms_per_suspect"], 0) if e else "--"
    N["XlRetMs"] = num(e["ms_retrieve"], 0) if e else "--"
    N["XlPostings"] = num(e["postings_touched"], 0) if e else "--"
    N["BtLlmSec"] = num(b["llm_seconds"], 1) if b else "--"
    N["DenseMs"] = num(find(eff, method="dense_cosine")["ms_per_suspect"], 0) if find(eff, method="dense_cosine") else "--"
    N["XlTestN"] = str(find(det, method="spintrace_xl", level="hi_translation")["n_pos"]) if find(det, method="spintrace_xl", level="hi_translation") else "--"
    cold_path = RESULTS / "crosslingual_cold_cost.json"
    cold = json.loads(cold_path.read_text()) if cold_path.exists() else None
    N["XlColdMs"] = num(cold["ms_per_suspect_cold"], 0) if cold else "--"
    try:
        N["Speedup"] = num(float(b["ms_per_suspect"]) / float(cold["ms_per_suspect_cold"]), 0)
    except (TypeError, ValueError, KeyError):
        N["Speedup"] = "--"
    for lang, lv in (("Ta", "ta_translation"), ("Te", "te_translation"), ("Bn", "bn_translation")):
        N[lang + "F"] = d("spintrace_xl", lv)
        r = find(rank, method="spintrace_xl", level=lv)
        N[lang + "Pone"] = num(r["p@1"]) if r else "--"
        r = find(rank, method="dense_cosine", level=lv)
        N["Dense" + lang + "Pone"] = num(r["p@1"]) if r else "--"
    for key, v in (("Full", "full"), ("NoNames", "no_names"), ("NoNumbers", "no_numbers"), ("NoDates", "no_dates"),
                   ("TranslitExact", "translit_exact"), ("NoDf", "no_idf_check"), ("DenseVerify", "dense_verify"),
                   ("NoAlign", "no_alignment_signals")):
        N["Abl" + key] = a(v)
    N["MsAll"], N["MsContAll"], N["MsMinhashAll"], N["MsMidAll"] = (g(m, "splice (all)") for m in ("spintrace", "containment", "minhash", "midterm"))
    N["MsPlain"], N["MsPlainCont"] = g("spintrace", "splice splice"), g("containment", "splice splice")
    N["MsSyn"], N["MsSynCont"] = g("spintrace", "splice splice_synonym"), g("containment", "splice splice_synonym")
    N["MsSeo"], N["MsSeoCont"] = g("spintrace", "splice splice_seo"), g("containment", "splice splice_seo")
    N["MsLlm"], N["MsLlmCont"] = g("spintrace", "LLM mashups (both sources test)"), g("containment", "LLM mashups (both sources test)")
    r = find(msd, method="spintrace", group="LLM mashups (both sources test)")
    N["MsLlmPrec"], N["MsLlmRecall"] = (num(r["precision"]), num(r["recall"])) if r else ("--", "--")
    N["MsSecondSpin"], N["MsSecondMid"] = k("spintrace", "2", "2"), k("midterm", "2", "2")
    N["MsThirdSpin"], N["MsThirdMid"] = k("spintrace", "3", "3"), k("midterm", "3", "3")
    for key, v in (("Full", "full"), ("NoResid", "no_residual")):
        r = find(msa, variant=v)
        N["MsAbl" + key] = num(r["F1 splice (all)"]) if r else "--"
    r = find(msc, method="spintrace", level="facts_only")
    N["MsFalseSame"] = num(r["false_source_rate"]) if r else "--"
    r = find(msc, method="midterm_shipped", level="facts_only")
    N["MsFalseSameMid"] = num(r["false_source_rate"]) if r else "--"
    N["MsMs"] = num(cost[0]["ms_per_suspect"], 0) if cost else "--"
    live = RESULTS / "crosslingual_live.json"
    if live.exists():
        lj = json.loads(live.read_text())
        N["XlLiveSuspects"], N["XlLiveFlagged"] = f"{lj.get('suspects', 0):,}".replace(",", "{,}"), str(lj.get("flagged", "--"))
        N["XlLiveDeva"] = f"{lj.get('by_script', {}).get('deva', {}).get('suspects', 0):,}".replace(",", "{,}")
        N["XlLiveLag"] = num(lj.get("median_lag_hours"), 1)
        N["XlLiveCredit"] = str(lj.get("flagged_crediting_source", "--"))
        N["XlLiveAgency"] = str(lj.get("flagged_naming_agency", "--"))
    ss = RESULTS / "crosslingual_live_same_site.json"
    if ss.exists():
        sj = json.loads(ss.read_text())
        N["SameSiteSuspects"], N["SameSiteFlagged"] = str(sj.get("suspects", "--")), str(sj.get("flagged", "--"))
    ds = RESULTS / "dataset.json"
    if ds.exists():
        dj = json.loads(ds.read_text())
        N["DocsCrawl"] = f"{dj['origins'].get('crawl', {}).get('n', 0):,}".replace(",", "{,}")
        N["Sites"] = str(dj["crawl_sites"])
        N["Indic"] = f"{sum(v for k_, v in dj['crawl_scripts'].items() if k_ != 'latn'):,}".replace(",", "{,}")
    OUT.mkdir(exist_ok=True)
    (OUT / "numbers.tex").write_text("".join(f"\\newcommand{{\\n{k_}}}{{{v}}}\n" for k_, v in sorted(N.items())))


if __name__ == "__main__":
    for fn in (numbers, xl_detection, xl_other, xl_ablation, ms_detection, ms_by_rank, ms_ablation, ms_controls, dataset, xlm_tables):
        fn()
    print("tables in", OUT)
