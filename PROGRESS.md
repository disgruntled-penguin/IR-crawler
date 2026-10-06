# Progress

Updated 2026-10-07 01:20 IST. Unattended run until 16:00.

## Done

- Network check (session start): bbc.com/robots.txt, en.wikinews.org, web.archive.org and PyPI reachable.
- Live crawl running since 22:29 (`data/logs/crawl.log`), ~6.2k articles from 36 seeds. `python -m spintrace crawl-report` prints the politeness audit.
- Politeness bug found and fixed at 23:45: a frontier rescheduling race let 9 requests (7 hosts) go out 0.3-4.9 s after the previous one to the same host. Fixed in the frontier, plus a hard per-host guard before every request, plus a regression test that fails on the old code. No delay violations since; robots.txt audit: 0 disallowed URLs fetched. Later: refusals (401/403/405/451) now count towards backoff and the host stop; a 5xx robots.txt means "disallowed for now".
- Wikinews: 400 originals + 4,000 same-site distractors from the official dump (the API is disallowed by robots.txt); category tag lines cleaned.
- Rewrites: exact, light, synonym for 400; LLM SEO and summary for 200 (llama3.2:3b via Ollama); facts-only independent-coverage negatives generating.
- Real farm pairs from NewsGuard's Aug 2023 report via the Wayback Machine: NYT -> GlobalVillageSpace (ranked 1st of 11.6k docs, p = 0.999, traced) and Wired -> TopGolf.kr (ranked 1st, but verification fails on the list-format page).
- Full pipeline, evaluation harness, judging tool, demo command, pipeline diagram, README with design choices and an auto-synced results block. Critique rounds 1 and 2 in `notes/`.

## Latest numbers (round 2, before facts-only)

| test F1 | exact | light | synonym | SEO | summary | FPR hard neg |
| --- | --- | --- | --- | --- | --- | --- |
| SpinTrace | 0.995 | 0.995 | 0.995 | 0.989 | 0.978 | 0.011 |
| MinHash Jaccard | 1.000 | 0.997 | 0.326 | 0.022 | 0.000 | 0.000 |
| Shingle containment (all pairs) | 0.948 | 0.948 | 0.948 | 0.800 | 0.844 | 0.109 |
| tf-idf cosine | 0.943 | 0.943 | 0.943 | 0.847 | 0.835 | 0.126 |
| BM25 | 0.974 | 0.974 | 0.974 | 0.844 | 0.942 | 0.052 |
| Dense cosine | 0.814 | 0.814 | 0.814 | 0.623 | 0.561 | 0.460 |

Index elimination: 10 highest-idf terms find the source at Recall@1 0.987 touching ~160 postings; 10 lowest-idf terms 0.691 at ~26,600.

## Next

- When facts-only generation finishes: `scripts/run_all.sh`, critique round 3, final README, clean-clone test.

## For the team (from 10:00)

- Judge with `python -m spintrace judge live --judge 1` (and `--judge 2`), then `judge hardneg` and `judge search`. `python -m spintrace judge live --metrics` prints precision and agreement; `python -m spintrace eval --report-only` refreshes every table and the README.
- A spot check of 8 live flags (round-1 model) found about 3 real derivatives; the rest were independent same-event coverage or shared wire copy. The facts-only negatives are meant to fix this; your labels decide.
- Still missing from the team inputs: farm-site seeds and further NewsGuard pairs (the Bored Panda and People originals are not named in the report).
- Power: the laptop was on battery until about 01:00; it is on AC now.
