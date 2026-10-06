# Progress

Updated 2026-10-06 23:35 IST.

## Done

- Network check (session start): bbc.com/robots.txt, en.wikinews.org, web.archive.org and PyPI all reachable.
- Polite crawler running since 22:29 (`data/logs/crawl.log`): robots.txt and Crawl-delay obeyed, 5 s minimum per host, manual redirects, backoff on 429/503, host stop after 5 errors, trap guards, feed re-polling every 45 min.
- Wikinews originals: 400 published articles from the official dump (Wikinews robots.txt disallows the API, so it was not used).
- Rewrites: exact, light, synonym for all 400; LLM SEO rewrite and summary (llama3.2:3b via Ollama) generating in the background for the first 200.
- Positional zoned index, tf-idf and BM25, shingles, MinHash, LSH, candidate retrieval, verification signals, provenance, copy-graph scan, search with g(d), inspect commands (`postings`, `pair`, `suspect`).

## In progress

- Evaluation harness: detection F1 per level against baselines, ranking metrics, hard negatives, ablations.

## Blockers and notes for the team

- Laptop was on battery (24%, discharging) at 23:30; if the run stopped overnight, that is why.
- Team inputs not yet in the repo: farm-site seeds, NewsGuard pairs with Wayback links, judged labels. The crawl uses a robots-checked seed list in `seeds/seeds.csv`; add farm sites there.
- NPR and NDTV disallow article pages in robots.txt; their feeds are kept so the logs show the skips.
