# Requirements checklist (assignment PDF and CLAUDE.md)

Status as of 7 Oct evening, after judging. "Team" marks the items that are human tasks.

## Assignment PDF

| Requirement | Status | Where |
| --- | --- | --- |
| Working IR system on one track (T4: crawling, freshness, web integrity) | Done | `spintrace/` |
| IR principles used correctly, choices explained | Done | README "How IR is used", "Design choices" |
| Libraries explained in IR terms | Done | README "Libraries, in IR terms" |
| Out-of-syllabus methods (BM25, dense retrieval) | Done, as baselines | `index.py`, `eval/run.py` |
| Runs live on real inputs; nothing faked or hard-coded | Done | live crawl, `demo` draws seeded random cases |
| README: setup, how to run, data sources, works vs planned | Done | README |
| README reproduces from a clean clone | Done (fresh clone and venv: install, 17 tests, build, postings, eval, demo; recrawl and the NewsGuard fetch need the live web and were not rerun) | PROGRESS.md |
| Evaluation: judged queries, P/R/P@k, baseline comparison | Done: synthetic, real pairs, and both judges' labels on 174 live flags and 154 search results | `results/`, `evaldata/*.csv` |
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
| robots.txt with Crawl-delay; delay per host; identifying UA with contact; backoff on 429/503; stop after errors; skip comments and profiles; no crawled texts committed | Done; 39 early delay violations are disclosed (9 from the 23:45 race, 30 from overlapping rehearsal crawls on 7 Oct), and a lock now refuses a second crawler; the final crawl has none |
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
| Precision of flagged live pairs with judge agreement | Done: 0.845 either judge, 0.437 both, 0.738 agreed pairs; kappa 0.12 (README states all three) |
| Search: P@10, nDCG@10 with and without g(d); share of originals | Done on synthetic queries and on 15 judged live queries (g(d) leaves P@10 and nDCG@10 unchanged there; README says so) |
| Crawler: pages, rate, disallowed fetched (0), smallest gap per host, duplicate rate, trap guards | Done (`results/crawl_report.json`) |
| Ablations, dev-only tuning, where SpinTrace loses, no hand-picked demos | Done |
| Hard negatives (same event, other outlet) | Done: crawl-mined and facts-only synthetic |
| Stemming comparison, dense baseline, per-domain integrity report | Done |
| Only-if-ahead: CC-NEWS scale, round-trip translation | Both done |

## Extras beyond the brief

| Item | Where |
| --- | --- |
| Blind judging web page, judge pack and label merge for a remote judge | `spintrace/ui/judge_*`, `eval/judgepack.py` |
| Demo web app: search with g(d), trace view, site integrity | `python -m spintrace app` |

## Final phase (21 to 22 Nov 2026)

Statuses as of 22 Nov 05:30 IST. The final brief itself is not in the repository, so these are checked against the
mid-term assignment, CLAUDE.md and the team's stated needs (dataset description, project report, presentation on 25 Nov).

| Item | Status | Where |
| --- | --- | --- |
| Translations traced to their English source (Hindi, verified end to end) | Done, on synthetic Hindi; live results unjudged | `text_indic.py`, `translit.py`, `kgram.py`, `detect/crosslingual.py`, README "Cross-language tracing" |
| Tamil, Telugu, Bengali | Partly: anchors retrieve the source, verification does not transfer (sentence model coverage); reported as a negative result | `results/crosslingual_*`, report Section 6.1 |
| Mashups traced to all their sources with shares | Done | `detect/multisource.py`, README "Multi-source mashups" |
| Format changes (listicle, Q&A) | Done as a negative result: the fact signals did not help | `results/format_*` |
| Baselines with their best dev threshold, ablations, confidence intervals | Done for both features | `results/crosslingual_*`, `results/multisource_*` |
| Hard negatives, and a check that length does not separate the classes | Done: length artefact found and length-matched negatives added | `notes/critique_final.md` |
| Inspectable commands | Done | `suspect-xl`, `kgram`, `sources` |
| Critique rounds | Done: per feature and cross-cutting | `notes/critique_xl.md`, `critique_multisource.md`, `critique_final.md` |
| Dataset described | Done | `results/dataset.md`, README "Data and credits", report Section 6.1 |
| Project report | Draft written; numbers come from the result files; page limit unknown | `report/SpinTrace_final_report.tex` |
| Presentation on 25 Nov | Team: format not yet confirmed | |
| Judged labels for the final's live results | Not done (no judging time); reported as unjudged counts | README, report Section 7 |
| Tests | 35 pass | `tests/` |
| Crawler politeness in the final crawl | Done at 09:05 on 22 Nov, after the crawler stopped, with `crawl-report --since 2026-11-21` (31,917 page requests to 45 hosts, 0 disallowed fetches, 0 requests under the 5 s floor, closest pair 20.0 s, 43 backoffs, 366 trap guards) | `results/crawl_report.json`, report Section 6.13 |
| README commands from a clean clone | Partly checked at 05:25: fresh clone, 35 tests pass, CLI help runs, the report builds with tectonic (20 pages). Not rerun from a clean clone: the crawl, index build and evaluation (they need the 1 GB data directory and hours) | PROGRESS.md |
| Report in the professor's research-paper format (title, abstract, introduction with contributions, related work with 5-7 cited papers, objective, proposed model with a diagram, methodology with formulas, experiments and results with a comparison with published methods, conclusions and limitations, references in her format) | Done on 22 Nov (26 pages including the appendix) | `report/SpinTrace_final_report.pdf` |
| Submission ZIP: report PDF, GitHub link, source code, dataset; named CSD358_Member1_Member2_Member3 | Built by `scripts/make_submission.sh`; GitHub link needs the push (not yet approved) and the member names need confirming | |
