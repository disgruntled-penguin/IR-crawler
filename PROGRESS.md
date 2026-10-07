# Progress

Updated 2026-10-07 09:25 IST. The live crawl keeps running in the background (watched by a monitor).

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
- Crawl at 09:20: 27,176 articles from 36,418 page requests over 10.8 h; 0 disallowed URLs fetched; 9 delay violations, all before the 23:45 fix; median freshness 73 min. Evaluations used the snapshot at the 04:30 run (~16k crawled articles).

## Overnight bugs found and fixed

1. 23:45 frontier rescheduling race (9 requests under 5 s, min 0.29 s). Fixed plus a hard per-host guard; regression test.
2. 01:12 frontier starvation (most hosts idle behind three prolific ones). Front queues now partitioned by host; regression test.
3. Hard-negative labels: the first miner let wire copies in as "independent"; now requires no reuse of any earlier text.
4. Wikinews category tags leaked into originals; cleaned, rule-based rewrites regenerated.
5. 02:21-02:50 the crawler died with "database is locked" while the CC-NEWS loader held long write transactions. The crawler now waits and retries; bulk writers commit every 20-50 rows.

## For the team (from 10:00)

1. Judge: `python -m spintrace judge live --judge 1` and `--judge 2` (174 flagged live pairs), then `judge hardneg` and `judge search`. `python -m spintrace judge live --metrics` prints precision and kappa; `python -m spintrace eval --report-only` folds the labels into every table and the README.
2. Add farm-site seeds to `seeds/seeds.csv` and any further NewsGuard pairs to `evaldata/newsguard_pairs.csv` (then `python -m spintrace newsguard`).
3. Demo: `python -m spintrace demo --seed 7` (seeded random cases including a miss and a false alarm), `suspect`, `pair`, `postings teheran`, `search "..."` with and without `--no-g`, `crawl-report` (prints politeness log lines).
4. Report: README sections map onto the report structure; `results/pipeline.png` is the diagram; `notes/` holds the critique history.
5. Before recording, stop the crawler with `kill $(cat data/crawl.pid)` if the machine is needed; restarting it is polite (one-delay guard on start).

## Blockers

- None open. The Bored Panda and People originals in NewsGuard's report are not named, so those two pairs stay unloaded.
