# Requirements checklist (assignment PDF and CLAUDE.md)

Status as of the overnight run. "Team" marks the items that are human tasks.

## Assignment PDF

| Requirement | Status | Where |
| --- | --- | --- |
| Working IR system on one track (T4: crawling, freshness, web integrity) | Done | `spintrace/` |
| IR principles used correctly, choices explained | Done | README "How IR is used", "Design choices" |
| Libraries explained in IR terms | Done | README "Libraries, in IR terms" |
| Out-of-syllabus methods (BM25, dense retrieval) | Done, as baselines | `index.py`, `eval/run.py` |
| Runs live on real inputs; nothing faked or hard-coded | Done | live crawl, `demo` draws seeded random cases |
| README: setup, how to run, data sources, works vs planned | Done | README |
| README reproduces from a clean clone | Done (tested: clone, tests, recrawl, build, sampled eval, inspect, crawl-report) | PROGRESS.md |
| Evaluation: judged queries, P/R/P@k, baseline comparison | Synthetic and real parts done; live judgments pending | `results/`, `evaldata/*.csv` (Team: judge) |
| Graphs and tables | Done | `results/*.png`, `results/summary.md` |
| Data ethics: robots.txt, delays, no personal data, credits | Done | README "Ethics", "Data and credits"; `crawl-report` |
| Pipeline diagram for the report | Done | `results/pipeline.png` |
| Demo video 5-8 min showing intermediate output and a limitation | Team | `demo`, `suspect`, `pair`, `postings`, `search`, `crawl-report` |
| Report PDF <= 8 pages with the given structure | Team | README sections map onto it |
| Work division | Team | |
| AI-use declaration | Done (log) | README "AI-use log" |

## CLAUDE.md non-negotiables

| Item | Status |
| --- | --- |
| Real inputs; synthetic rewrites only as labels | Done |
| Own IR core: positional zoned index, tf-idf, BM25, shingling, MinHash, LSH, frontier with politeness | Done |
| Inspectable: postings, shingle overlap and Jaccard, candidate scores, sentence alignment, verdict | Done (`postings`, `pair`, `suspect`, `demo`) |
| robots.txt with Crawl-delay; delay per host; identifying UA with contact; backoff on 429/503; stop after errors; skip comments and profiles; no crawled texts committed | Done; 9 early delay violations before the 23:45 fix are disclosed |
| One README, clean clone works | Done |
| Code style: short comments, no decorative separators | Done (pyflakes clean) |
| AI-use log | Done |
| First commit is the plan; small commits with real timestamps; timeline in README | Done |

## CLAUDE.md evaluation requirements

| Question | Status |
| --- | --- |
| Detection P/R/F1 per level, one line per method, PR curve | Done |
| P@1, MRR, Recall@k; candidates scored and time vs all-pairs | Done |
| Real farms: rank of the original per NewsGuard pair | Done for 2 pairs; others need original URLs (Team) |
| Precision of flagged live pairs with judge agreement | Sheet and tool ready; Team to judge |
| Search: P@10, nDCG@10 with and without g(d); share of originals | Done on synthetic queries; live sheet for Team |
| Crawler: pages, rate, disallowed fetched (0), smallest gap per host, duplicate rate, trap guards | Done (`results/crawl_report.json`) |
| Ablations, dev-only tuning, where SpinTrace loses, no hand-picked demos | Done |
| Hard negatives (same event, other outlet) | Done: crawl-mined and facts-only synthetic |
| Stemming comparison, dense baseline, per-domain integrity report | Done |
| Only-if-ahead: CC-NEWS scale, round-trip translation | Both done |
