# Progress

Updated 2026-10-07 evening IST. The crawl is stopped; judging is complete.

## State

Everything in CLAUDE.md priorities 1 and 2 is built and evaluated, plus both "only if ahead" items (one CC-NEWS
file for scale, a round-trip translation level). Three critique rounds are in `notes/` with an addendum for the
final rerun; `notes/requirements_checklist.md` maps every requirement to its status. Final numbers are in
`results/summary.md` and the README's results block; charts in `results/`.

| test F1 | exact | light | synonym | SEO | summary | translation | FPR crawl hard neg | FPR facts-only |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SpinTrace | 0.960 | 0.960 | 0.960 | 0.782 | 0.902 | 0.816 | 0.007 | 0.156 |
| MinHash Jaccard | 0.992 | 0.990 | 0.322 | 0.021 | 0.000 | 0.000 | 0.000 | 0.033 |
| Shingle containment (all pairs) | 0.850 | 0.850 | 0.850 | 0.573 | 0.681 | 0.555 | 0.090 | 0.433 |
| tf-idf cosine | 0.922 | 0.922 | 0.920 | 0.723 | 0.659 | 0.695 | 0.063 | 0.144 |
| BM25 | 0.927 | 0.927 | 0.922 | 0.636 | 0.852 | 0.739 | 0.043 | 0.189 |
| Dense cosine | 0.671 | 0.671 | 0.671 | 0.436 | 0.332 | 0.314 | 0.502 | 0.311 |

- Index elimination: 10 highest-idf terms find the source at Recall@1 0.970 touching ~280 postings; 10 lowest-idf terms 0.713 touching ~48,000. The full candidate query touches ~1.1% of a full-document query's postings.
- With same-story distractors, dense retrieval ranks the true source first for 0.833 of SEO/summary rewrites; SpinTrace 0.989.
- Search: with g(d) the original ranks first for 52.6% of test queries (10% without); nDCG@10 0.837 -> 0.886.
- NewsGuard pairs: NYT -> GlobalVillageSpace retrieved first of 20.9k, verified, traced; Wired -> TopGolf.kr retrieved first, not verified.
- Scale: +3,637 CC-NEWS articles (24.5k docs), nothing refit: F1 within 0.01 of the main run.
- Crawl (final, stopped about 12:00 on 7 Oct): 32,513 articles from 43,683 page requests over 13.5 h; 0 disallowed URLs fetched; 39 delay violations (9 before the 23:45 fix, 30 from rehearsal crawls that overlapped on 7 Oct 23:13-23:16; a lock now refuses a second crawler); median freshness 70 min. Evaluations used the snapshot at the 04:30 run (~16k crawled articles).
- Human judgments: both judges labelled all 174 live flags and 154 search results. Agreement 0.59 / 0.64, kappa 0.12 / 0.16. Live precision 0.845 (either judge), 0.437 (both), 0.738 (agreed pairs). Team convention: positive if either judge said so; the README reports all three. Hard negatives were not hand-checked.

## Overnight bugs found and fixed

1. 23:45 frontier rescheduling race (9 requests under 5 s, min 0.29 s). Fixed plus a hard per-host guard; regression test.
2. 01:12 frontier starvation (most hosts idle behind three prolific ones). Front queues now partitioned by host; regression test.
3. Hard-negative labels: the first miner let wire copies in as "independent"; now requires no reuse of any earlier text.
4. Wikinews category tags leaked into originals; cleaned, rule-based rewrites regenerated.
5. 02:21-02:50 the crawler died with "database is locked" while the CC-NEWS loader held long write transactions. The crawler now waits and retries; bulk writers commit every 20-50 rows.

## For the team (from 10:00)

1. Judging is done for the live flags and search (hard negatives were not hand-checked). `python -m spintrace judge live --metrics` prints agreement; `python -m spintrace eval --report-only` folds the labels into every table and the README.
2. Add farm-site seeds to `seeds/seeds.csv` and any further NewsGuard pairs to `evaldata/newsguard_pairs.csv` (then `python -m spintrace newsguard`).
3. Demo: `python -m spintrace demo --seed 7` (seeded random cases including a miss and a false alarm), `suspect`, `pair`, `postings teheran`, `search "..."` with and without `--no-g`, `crawl-report` (prints politeness log lines).
4. Report: README sections map onto the report structure; `results/pipeline.png` is the diagram; `notes/` holds the critique history.
5. Before recording, stop the crawler with `kill $(cat data/crawl.pid)` if the machine is needed; restarting it is polite (one-delay guard on start).

## Blockers

- None open. The Bored Panda and People originals in NewsGuard's report are not named, so those two pairs stay unloaded.

## Web interfaces (added 2026-10-07 afternoon)

- `python -m spintrace judge-ui` (port 8765): blind judging page over the three label sheets.
- `python -m spintrace app` (port 8766): Search with g(d), Trace and Site integrity tabs over the same index and detection code.

## Clean-clone check (7 Oct evening)

Fresh clone and virtual environment: `pip install -r requirements.txt`, 17 tests, `build --stem`, `postings`, `eval` and `demo --seed 7` all ran. `recrawl` (hours, live web) and the NewsGuard Wayback fetch were not rerun, so the clean clone's numbers differ from the README's, which come from the full crawl.

## Final phase (21 to 22 Nov 2026)

Plan: two problems the mid-term cannot solve, because both break its assumption that a copy shares words with one original.

1. Cross-language derivatives: English originals rewritten in Hindi (Tamil, Telugu and Bengali as a generality check), traced through numbers, dates and transliterated names (own transliteration, phonetic key and k-gram index, no translation step).
2. Multi-source mashups: one article stitched from two or three originals, recovered by explain-away rounds with per-source shares.
3. Prototype: Hindi mashups traced by passage (the two features composed). Stretch: format changes (negative result).

State at 22 Nov 05:25 IST:
- Both helper-agent branches merged into `main` (cross-language and multi-source); 35 tests pass; the report builds from a clean clone (20 pages).
- Generation finished at 04:56 (820 rows, 100 Hindi originals per level). `scripts/final_run.sh` reran everything incrementally (nine steps, all exit 0, 05:17), then the length check and the cold-cost benchmark.
- Headline test numbers (49-50 Hindi test originals per level): SpinTrace-XL F1 0.860 / 0.852 / 0.818 (translation / SEO / summary) against 0.906 / 0.771 / 0.753 for translate-back-then-SpinTrace, 0.897 / 0.800 / 0.427 for tf-idf on the back-translation and about 0.4 for multilingual dense cosine; all intervals overlap, so the claim is comparable accuracy at 65 ms (483 ms cold) against 17.4 s of LLM time per suspect. Same-event false-positive rate 10% (long) and 22% (130 words). Mashups: English splices F1 0.870 against 0.742 (shingle containment) and 0.117 (midterm); Hindi passage tracing 0.765 against 0.528 whole-text, with a 30% false-source rate (prototype).
- Live (unjudged): see the post-crawl block below for the final run (4,426 pages, 21 flagged).
- Found and fixed along the way: the crawler died at 17:47 on an unhandled connection reset (regression test added); word count alone separated Hindi summaries from the same-event negatives, so length-matched negatives were added (their false-positive rate is higher, which is reported); the Hindi mashup pairing reshuffled on every new rewrite; hand-typed length and cost figures moved into scripts that write result files.
- Final report: `report/SpinTrace_final_report.tex` (numbers generated from `results/`; 20 pages including the appendix, body about 16; page limit unknown). Dataset: `results/dataset.md`, README "Data and credits", report Section 5.
- Crawler: running since 22 Nov 02:00 for 7 h (log `data/logs/crawl.log`) after a restart following the 17:47 crash. `crawl-report --since 2026-11-21`: 0 disallowed fetches, 0 requests under the 5 s floor, closest pair 20 s.
- No human judging this round: the final's live results are unjudged counts with seeded-random examples.

Post-crawl refresh (22 Nov 09:00-09:30 IST), done:
- The crawler stopped on its own at 09:00:12 (`STOP saved=19027` for this run; 57,353 crawled articles in the store, 4,167 Devanagari). `crawl-report --since 2026-11-21`: 31,917 page requests to 45 hosts over 17.4 h, 0 disallowed fetches, 0 requests under the 5 s floor, closest pair 20.0 s, 43 backoffs, 366 trap guards. Over both crawls 77,088 fetches were re-audited against the stored robots.txt, 0 disallowed.
- Live pass rerun on the final crawl (`live-xl --rebuild`, 18 min): 4,426 Indian-language pages (4,175 Devanagari), 21 flagged (0.5%), median lag 3.6 h, 2 credit their source (BBC Hindi), 2 name PTI; by outlet Amar Ujala 7/611, Aaj Tak 4/786, TV9 Hindi 4/620, India TV 2/733, ABP Live 2/634, BBC Hindi 2/216, Dainik Bhaskar 0/687, Hindustan Times 0/125. `--same-site`: 341 pages, 2 flagged. This replaces the 05:15 pass (2,375 pages, 15 flagged). Still unjudged.
- All evaluations except the live passes ran on index snapshots built before the crawl ended (the English evaluation index holds 20,860 documents); the report and README now say so.
- `dataset_stats`, `export` (57,357 URLs, 27,220 MinHash signatures), tables, figures and the PDF regenerated; the PDF is 20 pages and builds.
- Bibliography: Potthast 2011 and Hagen 2015 entries checked against web sources and correct.

Still open: the user's answers on the report page limit and the 25 Nov format.

Blocked or open: the final report's length limit and the presentation format on 25 Nov are unknown (asked, no answer yet). Blocked feeds: Navbharat Times, Jagran, Zee News Hindi and Patrika (404), News18 Hindi (403), Live Hindustan (503); Jansatta, DW Hindi and Prabhasakshi returned empty feeds.
