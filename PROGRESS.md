# Progress

Updated 2026-10-07 02:55 IST. Unattended run until 16:00; the live crawl keeps running in the background.

## State

Everything in CLAUDE.md priorities 1 and 2 is built and evaluated; three critique rounds are done
(`notes/critique_round1.md` to `round3.md`). Final numbers are in `results/summary.md` and the README's
results block; charts in `results/`.

| test F1 | exact | light | synonym | SEO | summary | FPR crawl hard neg | FPR facts-only |
| --- | --- | --- | --- | --- | --- | --- | --- |
| SpinTrace | 0.962 | 0.962 | 0.962 | 0.793 | 0.906 | 0.004 | 0.156 |
| MinHash Jaccard | 0.992 | 0.990 | 0.322 | 0.021 | 0.000 | 0.000 | 0.033 |
| Shingle containment (all pairs) | 0.860 | 0.860 | 0.860 | 0.586 | 0.695 | 0.094 | 0.433 |
| tf-idf cosine | 0.920 | 0.920 | 0.917 | 0.764 | 0.705 | 0.077 | 0.167 |
| BM25 | 0.929 | 0.929 | 0.924 | 0.648 | 0.856 | 0.051 | 0.189 |
| Dense cosine | 0.720 | 0.720 | 0.720 | 0.489 | 0.375 | 0.485 | 0.311 |

- Index elimination: 10 highest-idf terms find the source at Recall@1 0.968 touching ~200 postings; 10 lowest-idf terms 0.677 touching ~33,500.
- With same-story distractors, dense retrieval ranks the true source first for 0.833 of SEO/summary rewrites; SpinTrace 0.989.
- Search: with g(d) the original ranks first for 54% of test queries (14% without); nDCG@10 0.847 -> 0.894.
- NewsGuard pairs: NYT -> GlobalVillageSpace retrieved first of 14.6k, verified, traced; Wired -> TopGolf.kr retrieved first, not verified.
- Crawl: ~8.9k articles, 12.3k page requests, 0 disallowed URLs fetched; 9 delay violations, all before the 23:45 fix.

## After the final run

- Scale: one CC-NEWS file (3,645 English articles from 6 Oct) grows the index to 19.4k docs; with nothing refit,
  F1 stays within 0.004 of the main results and the query still touches ~1.2% of a full-document query's postings
  (`results/scale.csv`).
- Round-trip translation (English -> German -> English, llama3.2:3b) is generating for 100 originals as a sixth
  rewrite level; the evaluation will be rerun with it.

## Overnight bugs found and fixed

1. 23:45 frontier rescheduling race (9 requests under 5 s, min 0.29 s). Fixed plus a hard per-host guard; regression test.
2. 01:12 frontier starvation (most hosts idle behind three prolific ones). Front queues now partitioned by host; regression test.
3. Hard-negative labels: the first miner let wire copies in as "independent"; now requires no reuse of any earlier text.
4. Wikinews category tags leaked into originals; cleaned, rule-based rewrites regenerated.
5. 02:21-02:50 the crawler died with "database is locked" while the CC-NEWS loader held long write transactions.
   The crawler now waits and retries; bulk writers commit every 20-50 rows; a monitor watches for a dead or stalled crawl.

## For the team (from 10:00)

1. Judge: `python -m spintrace judge live --judge 1` and `--judge 2` (83 flagged live pairs), then `judge hardneg` and `judge search`. `python -m spintrace judge live --metrics` shows precision and kappa; `python -m spintrace eval --report-only` refreshes every table and the README.
2. Add farm-site seeds to `seeds/seeds.csv` and any further NewsGuard pairs to `evaldata/newsguard_pairs.csv` (then `python -m spintrace newsguard`).
3. Demo material: `python -m spintrace demo --seed 7`, `suspect`, `pair`, `postings`, `search`, `crawl-report` (prints example politeness log lines). Every case in `demo` is a seeded random draw, including a miss and a false alarm.
4. Report: README sections map onto the report structure; `results/pipeline.png` is the pipeline diagram.

## Blockers

- None open. Wayback CDX was intermittently offline overnight; the Bored Panda and People originals in NewsGuard's report are not named, so those two pairs stay unloaded.
