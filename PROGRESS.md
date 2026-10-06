# Progress

Updated 2026-10-07 01:50 IST. Unattended run until 16:00.

## Done

- Network check (session start): bbc.com/robots.txt, en.wikinews.org, web.archive.org and PyPI reachable.
- Live crawl running since 22:29 (`data/logs/crawl.log`), ~3.5k articles from 35 seeds. `python -m spintrace crawl-report` prints the politeness audit.
- Politeness bug found and fixed at 23:45: a frontier rescheduling race let 9 requests (7 hosts) go out 0.3-4.9 s after the previous one to the same host. Fixed in the frontier, plus a hard per-host guard before every request, plus a regression test that fails on the old code. No delay violations since; robots.txt audit: 0 disallowed URLs fetched.
- Wikinews: 400 originals + 4,000 same-site distractors from the official dump (the API is disallowed by robots.txt).
- Rewrites: exact, light, synonym for 400 originals; LLM SEO (done for 200) and summaries (generating, llama3.2:3b via Ollama).
- Full pipeline: index, content-seen, candidate retrieval (fact-boosted rare terms + quote phrases + earlier-only filter), verification, provenance, copy graph, search with g(d), inspect commands.
- Evaluation harness with 6 baselines, dev-tuned thresholds, ablations, retrieval variants, query budget, undated provenance, search eval, crawl report. Critique round 1 done (`notes/critique_round1.md`), fixes F1-F7 applied.

## Interim numbers (round 1 rerun, SEO partial, no summaries yet)

| test F1 | exact | light | synonym | SEO | FPR hard neg |
| --- | --- | --- | --- | --- | --- |
| SpinTrace | 0.997 | 0.997 | 0.997 | 0.980 | 0.007 |
| MinHash Jaccard | 1.000 | 0.995 | 0.397 | 0.000 | 0.000 |
| Shingle containment (all pairs) | 0.977 | 0.977 | 0.977 | 0.731 | 0.049 |
| tf-idf cosine | 0.977 | 0.977 | 0.977 | 0.859 | 0.063 |
| Dense cosine | 0.922 | 0.922 | 0.920 | 0.521 | 0.217 |

Verifier with IR signals only (no embeddings): macro F1 0.988 vs 0.993 full. Candidate query touches ~3.6k postings vs ~321k for a full-document query.

## Next

- Final run after LLM summaries finish (~04:30): rerun `build --stem`, `eval`, `scan`, `search-eval`, `crawl-report`; critique round 2; README results.

## Blockers and notes for the team

- Laptop was on battery (24%, discharging) at 23:30; if the run stopped overnight, that is why.
- Team inputs not yet in the repo: farm-site seeds, NewsGuard pairs. I added the pairs named in NewsGuard's Aug 2023 report (`evaldata/newsguard_pairs.csv`); only Wired -> TopGolf.kr loads. The NYT original is not archived; the Bored Panda and People originals are not named; please add their URLs.
- Judging needed (columns judge1, judge2): `evaldata/live_flagged.csv`, `evaldata/hard_negatives.csv`, `evaldata/search_judgments.csv`.
- Wayback CDX was temporarily offline around 00:10.
- NPR and NDTV disallow article pages in robots.txt; their feeds are kept so the logs show the skips.
