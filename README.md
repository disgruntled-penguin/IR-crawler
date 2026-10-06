# SpinTrace

A polite, focused news crawler that catches articles rewritten from other outlets, including AI
paraphrases that standard duplicate checks miss, and traces each one to its original.

CSD358 IR hackathon, Track 4 (web crawling, freshness and web integrity).

**Claim.** SpinTrace extends the crawler's "content seen?" check from copy detection to
paraphrase-aware derivative detection: it retrieves a rewrite's likely source using the facts a
rewrite rarely changes, then traces provenance and demotes copies in ranking.

**Why.** NewsGuard fed 43 AI-rewritten farm articles to Grammarly's plagiarism checker; it could not
find the source of 34 of them (79%) ([NewsGuard, Aug 2023](https://www.newsguardtech.com/misinformation-monitor/august-2023)).
NewsGuard now tracks 3,749 AI content-farm sites ([AI Tracking Center](https://www.newsguardtech.com/special-reports/ai-tracking-center)).
Shingle-based near-duplicate detection, the textbook "content seen?" check, fails for the same reason:
a paraphrase shares almost no word 5-grams with its source.

**Idea.** A rewrite changes wording but keeps names, numbers, dates and direct quotes. Those are
high-idf terms and exact phrases. So the suspect's rarest terms become a query against an inverted
index (index elimination), its quotes become positional phrase queries, and a parametric filter keeps
only earlier articles. Classical IR finds the candidate sources; sentence embeddings only verify the
few candidates IR returns.

## Pipeline

```
 seeds (RSS, sitemaps)                                           Wikinews dump (CC BY)
        |                                                                 |
 [1 crawl] Mercator frontier -> robots.txt + delay -> fetch -> extract   [rewrites] exact, light,
        |   (spintrace/crawl/)                                            synonym, LLM SEO, LLM summary
        v                                                                 |
 SQLite store keyed by doc_id  <------------------------------------------+
        |
 [2 index] tokenise/normalise -> positional zoned index (title, body, quote), tf-idf, BM25
        |   content-seen: exact hash, 5-gram shingles, MinHash(128), LSH(32x4)
        v
 [3 detect] suspect -> 30 rarest terms (lnc.ltc cosine, term-at-a-time, heap top-20)
        |             + quote windows as phrase queries on the positional index
        |             + parametric filter: other sites, published earlier
        |          -> verify top 5: sentence alignment coverage and order, quotes, names/numbers,
        |             shingle containment -> logistic score (fit on dev)
        |          -> provenance: earlier date, or containment asymmetry when undated
        v
 [4 use] copy graph -> originality g(d) -> search: net = relevance + lambda * g(d)
         evaluation against exact hash, MinHash Jaccard, tf-idf cosine, BM25, dense cosine
```

## Setup

Python 3.11 or newer (developed on 3.14, macOS).

```
pip install -r requirements.txt
```

The sentence-embedding model (`sentence-transformers/all-MiniLM-L6-v2`, about 90 MB) downloads on
first use. Regenerating the LLM rewrites needs [Ollama](https://ollama.com) with `ollama pull llama3.2:3b`;
the generated rewrites are committed, so this is optional.

## Run

From a clean clone, in order:

```
python -m spintrace recrawl                 # re-fetch the committed article URLs politely (slow: >= 5 s per host)
# or run a fresh live crawl instead:  python -m spintrace crawl --hours 2
python -m spintrace build --stem            # ingest Wikinews + rewrites, shingles/MinHash, both indexes
python -m spintrace eval                    # all experiments; tables and charts in results/
python -m spintrace scan                    # detection over the crawl, stores the copy graph
python -m spintrace crawl-report            # politeness and correctness statistics
```

Inspecting intermediate output (for the demo):

```
python -m spintrace postings teheran                 # df, idf and positional postings per zone
python -m spintrace pair <doc_id|url> <doc_id|url>   # shingle overlap, Jaccard, containment, MinHash, LSH bands
python -m spintrace suspect <doc_id|url>             # query terms, candidates, signals, sentence alignment, verdict
python -m spintrace search "nobel prize physics"     # ranked results with relevance, g(d) and net score
python -m spintrace search "nobel prize physics" --no-g
```

`pytest` runs the unit tests (frontier politeness, text pipeline, index, MinHash).

## Data and credits

| Source | Use | Licence and notes |
| --- | --- | --- |
| Live crawl of 35 news sites (`seeds/seeds.csv`) | the index, live detection, hard negatives, search | Only URLs and features are committed (`evaldata/crawl_urls.csv`, `crawl_minhash.npz`); texts stay local. Each seed's robots.txt was checked before use |
| [English Wikinews](https://en.wikinews.org) dump, Oct 2026 | 400 originals for synthetic rewrites | CC BY 2.5, Wikinews contributors. Read from the official Wikimedia dump because Wikinews robots.txt disallows `/w/` (its API) |
| Synthetic rewrites (`evaldata/rewrites.jsonl`) | labelled evaluation set | Generated from the Wikinews originals; LLM levels by Meta Llama 3.2 3B via Ollama |
| NewsGuard August 2023 report | documented real farm rewrites (`evaldata/newsguard_pairs.csv`) | Farm copies and originals fetched from the Internet Archive Wayback Machine |
| all-MiniLM-L6-v2 (sentence-transformers) | sentence embeddings for verification and the dense baseline | Apache 2.0 |

## How IR is used, and where

| Concept | Where | Why SpinTrace needs it |
| --- | --- | --- |
| Crawl loop, URL frontier (Mercator front and back queues), politeness, robots.txt | `crawl/frontier.py`, `crawl/crawler.py`, `crawl/robots.py` | Collecting news and farm pages responsibly. Front queues by priority (feed entries > article-shaped links > other); one back queue per host; a heap of next-allowed times; `priority()` is the one replaceable policy function |
| URL normalisation, spider-trap guards | `crawl/urlnorm.py` | Lowercased host, no fragment, tracking parameters dropped, sorted query; guards on length, depth, repeated segments, parameter count and calendar loops |
| Content-seen check: exact hash, shingles, Jaccard, MinHash, LSH | `dedup.py` | The textbook duplicate check SpinTrace extends, and the main baseline |
| Tokenisation and normalisation | `text.py` | One pipeline for index, shingles and queries: NFKC, case and accent folding, numbers kept as values (`1,394` = `1394`), stopwords kept in postings so phrase queries work |
| Positional inverted index with zones, phrase queries | `index.py` | Quotes survive rewriting, so phrase matching on the positional index is a key signal; postings intersected in increasing df order |
| idf and index elimination | `detect/candidates.py` | Rare terms survive rewriting; only the suspect's 30 highest-weight terms with df >= 2 form the query |
| tf-idf (lnc.ltc), cosine, length normalisation, heap top-K | `index.py` | Candidate scoring; cosine normalisation matters because summaries are short |
| Parametric filter on publish date | `detect/candidates.py` | Provenance only compares a suspect with earlier articles |
| Static quality g(d) and net score; zone weighting | `search.py`, `detect/scan.py` | Originals outrank their copies in search |
| BM25, dense retrieval (out of syllabus) | `index.py`, `eval/run.py` | Baselines answering when sparse retrieval wins |
| Precision, recall, F1, P@k, MRR, Recall@k | `eval/report.py` | Every evaluation |

Libraries, in IR terms: `requests` is the HTTP fetcher; `trafilatura` performs boilerplate removal
(deciding what the document is); `lxml` parses links, RSS, Atom and sitemaps; `urllib.robotparser`
parses robots.txt rules (fetching, caching and delay policy are ours); `nltk` provides the Porter
stemmer and WordNet (used only to make synonym-spun test rewrites); `sentence-transformers` maps
sentences to vectors so paraphrased sentences are close, used for verification and the dense
baseline; `scikit-learn` fits the logistic combination of signals and supplies the stopword list;
`numpy` holds postings arrays; `matplotlib` draws the charts.

Written ourselves: the frontier and politeness logic, URL normalisation, tokenisation, the
positional zoned index, tf-idf and BM25, phrase queries, shingling, MinHash, LSH, candidate
retrieval, the signals, provenance, the copy graph, search and the evaluation.

## Design choices and why

| Choice | Value | Reason |
| --- | --- | --- |
| Politeness delay | max(Crawl-delay, 5 s) per host | The brief asks for "a few seconds"; 5 s keeps ~30 hosts busy at ~1 page/s overall from one process |
| Backoff | delay x2 per consecutive error (cap 64x, 10 min), stop host after 5 | Standard exponential backoff; 429/503 mean the host wants us to slow down |
| Frontier | 3 front queues (feed/sitemap entries, article-shaped links, other), bias 8:3:1; one back queue per host, capped at 50 | Mercator design; the bias keeps fresh feed entries first without starving discovery |
| Trap guards | URL <= 300 chars, depth <= 8, a segment repeated <= 2 times, <= 3 query params, 1,500 pages per host | Calendar and faceted-navigation loops show up as long, deep or parameter-heavy URLs |
| What a document is | trafilatura main text, >= 120 words, plus article metadata, a publish date or a feed date | Drops section fronts, galleries and stubs; keeps short wire items |
| Tokens | NFKC, case and accent folding, numbers as values, stopwords kept in postings | Rewrites keep names and numbers but not formatting; phrase queries need every position |
| Shingles | word 5-grams | Long enough to be specific, short enough to survive light edits (textbook range 4-9) |
| MinHash / LSH | 128 permutations; 32 bands x 4 rows | Estimate error about 1/sqrt(128) = 0.09; the LSH knee (1/32)^(1/4) = 0.42 Jaccard catches light edits |
| Candidate query | 30 highest tf-idf body terms with df >= 2; names and numbers boosted x2 | df = 1 terms can only match the suspect itself; the query-budget experiment shows how recall depends on m and on idf |
| Quote phrases | 6-token windows of each quote (step 3) as phrase queries | A rewrite may trim a quote; any intact window is evidence |
| Weighting | lnc.ltc cosine for candidates, BM25 (k1 = 1.2, b = 0.75) as a baseline | lnc.ltc puts idf on the query side only, so the rare-term query decides; cosine normalisation stops long documents winning |
| Candidates | top 20 retrieved, top 5 verified | Recall@20 is the retrieval target; verification is the expensive step |
| Alignment | sentence pair matches if cosine >= 0.62 (MiniLM) | Paraphrased sentences sit around 0.7-0.9, unrelated ones below 0.5 |
| Verifier | logistic regression over the signals, class-balanced, fit on dev pairs | Few parameters, inspectable weights (`results/verifier.json`); the threshold maximises dev F1 |
| Provenance | earlier publish date; if a date is missing, the side covered by the other (margin 0.2); otherwise uncertain | A copy must come after its source; a summary covers less of its source than the source covers of it |
| g(d) | 1 - p(strongest source edge) | Originals keep g = 1; confirmed copies drop towards 0 |
| Net score | relevance + lambda g(d), lambda from {0, 0.1, ..., 0.8} chosen on dev queries | Lets relevance dominate and g break near-ties between an original and its copies |
| Splits | by original story (sha1 parity), dev about 50% / test about 50% | A story and all its rewrites fall on one side, so nothing leaks between dev and test |
| Evaluation sizes | 400 Wikinews originals, 4,000 distractors, 200 LLM-rewritten; ~150 crawl hard negatives per split | Stable metrics at hackathon cost: a 1-point F1 change is a handful of items |

## Ethics

Enforced in code (`crawl/`):

- robots.txt fetched per host and obeyed, including Crawl-delay; an unreachable robots.txt means the host is not crawled.
- At least 5 s between requests to one host (the robots.txt request counts), enforced twice: by the frontier's per-host heap and by a hard guard before every request.
- User agent `SpinTraceBot/0.1 (CSD358 IR course project; contact: ...)`.
- Exponential backoff on 429, 503 and other 5xx (Retry-After honoured); a host is stopped after 5 consecutive errors.
- Redirects are not followed automatically, so a redirect target goes back through robots.txt and the delay.
- Comments, author and profile pages, accounts, search and media are never fetched; trafilatura drops comment sections from article text; author names are not stored.
- Crawled full texts are never committed: only URLs, dates, hashes and MinHash signatures, plus `recrawl`.

## Results

Charts: `results/detection_f1_by_level.png`, `pr_curve.png`, `p_at_1_by_level.png`, `index_elimination.png`,
`ablations.png`, `pipeline.png`. Every table below is regenerated from `results/*.csv` by the commands above.

<!-- results:start -->
Generated by `python -m spintrace eval` and friends; every number below is on the test split.

Index: 11607 documents. Evaluation items (dev / test):

| set | exact | facts_only | hard_negative | light | original | seo | summary | synonym |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dev | 210 | 0 | 210 | 210 | 210 | 110 | 110 | 210 |
| test | 190 | 3 | 174 | 190 | 190 | 90 | 90 | 190 |

**Is this article derived from another?** Test F1 per rewrite level (negatives pooled), and false-positive rate on same-event negatives. Thresholds tuned on dev.

| method | exact | light | synonym | seo | summary | FPR hard neg (crawl) | FPR facts-only |
| --- | --- | --- | --- | --- | --- | --- | --- |
| spintrace | 0.995 | 0.995 | 0.995 | 0.989 | 0.978 | 0.011 | 0.000 |
| exact_hash | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| minhash_jaccard | 1.000 | 0.997 | 0.326 | 0.022 | 0.000 | 0.000 | 0.000 |
| shingle_containment | 0.948 | 0.948 | 0.948 | 0.800 | 0.844 | 0.109 | 0.000 |
| tfidf_cosine | 0.943 | 0.943 | 0.943 | 0.847 | 0.835 | 0.126 | 0.000 |
| bm25 | 0.974 | 0.974 | 0.974 | 0.844 | 0.942 | 0.052 | 0.333 |
| dense_cosine | 0.814 | 0.814 | 0.814 | 0.623 | 0.561 | 0.460 | 0.000 |

SpinTrace precision / recall / source correct per level:

| level | precision | recall | F1 | source correct |
| --- | --- | --- | --- | --- |
| exact | 0.990 | 1.000 | 0.995 | 0.995 |
| light | 0.990 | 1.000 | 0.995 | 0.995 |
| synonym | 0.990 | 1.000 | 0.995 | 1.000 |
| seo | 0.978 | 1.000 | 0.989 | 1.000 |
| summary | 0.978 | 0.978 | 0.978 | 0.978 |

**Is its original ranked first?** Test positives, all levels:

| method | P@1 | MRR | Recall@5 | Recall@20 |
| --- | --- | --- | --- | --- |
| spintrace | 0.999 | 0.999 | 1.000 | 1.000 |
| exact_hash | 0.253 | 0.253 | 0.253 | 0.253 |
| minhash_jaccard | 0.557 | 0.557 | 0.557 | 0.557 |
| shingle_containment | 0.992 | 0.994 | 0.997 | 0.999 |
| tfidf_cosine | 0.999 | 0.999 | 1.000 | 1.000 |
| bm25 | 0.999 | 0.999 | 0.999 | 0.999 |
| dense_cosine | 0.995 | 0.997 | 1.000 | 1.000 |

**Index elimination:** source recall from m query terms chosen by idf (test positives):

| query | Recall@1 | Recall@20 | postings touched |
| --- | --- | --- | --- |
| rare3 | 0.832 | 0.867 | 19 |
| rare5 | 0.904 | 0.955 | 44 |
| rare10 | 0.987 | 1.000 | 159 |
| rare30 | 0.999 | 1.000 | 1987 |
| common10 | 0.691 | 0.919 | 26649 |
| random10 | 0.969 | 0.996 | 6404 |

Retrieval variants (Recall@1 / Recall@20, all levels):

| variant | Recall@1 | Recall@20 |
| --- | --- | --- |
| rare terms + quote phrases | 0.999 | 1.000 |
| rare terms only (no quotes) | 0.999 | 1.000 |
| no fact boost | 0.997 | 1.000 |
| rare terms, Porter stemming | 0.999 | 1.000 |
| dense retrieval | 0.995 | 1.000 |
| full-document tf-idf | 0.999 | 1.000 |

**Ablations** (test macro F1 over levels; refit on dev without the signal):

| variant | macro F1 | FPR hard neg | FPR facts-only |
| --- | --- | --- | --- |
| full | 0.990 | 0.011 | 0.000 |
| minus rare_cos | 0.989 | 0.006 | 0.000 |
| minus shingle_containment | 0.993 | 0.006 | 0.000 |
| minus align_coverage | 0.994 | 0.006 | 0.000 |
| minus align_mean | 0.991 | 0.006 | 0.000 |
| minus align_order | 0.993 | 0.006 | 0.000 |
| minus back_coverage | 0.989 | 0.011 | 0.000 |
| minus ordered_coverage | 0.991 | 0.006 | 0.000 |
| minus quote_overlap | 0.990 | 0.011 | 0.000 |
| minus fact_overlap | 0.972 | 0.035 | 0.000 |
| no embedding signals (IR signals only) | 0.977 | 0.017 | 0.333 |
| dense candidates instead of IR | 0.989 | 0.006 | 0.000 |
| IR retrieval only, no verification | 0.856 | 0.207 | 0.333 |

**Cost per suspect** (test items):

| method | ms per suspect | docs fully compared | postings touched | earlier docs |
| --- | --- | --- | --- | --- |
| spintrace | 24.0 | 20 | 4564 | 5301 |
| full-document query (tf-idf/BM25) | - | - | 377512 | 5301 |
| exact_hash | 0.1 | - | - | 5301 |
| minhash_jaccard | 0.4 | - | - | 5301 |
| shingle_containment | 60.7 | 5301 | - | 5301 |
| tfidf_cosine | 16.4 | 5301 | - | 5301 |
| bm25 | 13.7 | 5301 | - | 5301 |
| dense_cosine | 1.6 | 5301 | - | 5301 |

**Provenance without dates** (coverage asymmetry on the known pair, margin 0.2):

| level | n | suspect called derived | direction reversed | uncertain |
| --- | --- | --- | --- | --- |
| exact | 190 | 0.000 | 0.000 | 1.000 |
| light | 190 | 0.079 | 0.000 | 0.921 |
| synonym | 190 | 0.000 | 0.000 | 1.000 |
| seo | 90 | 0.167 | 0.033 | 0.800 |
| summary | 90 | 0.956 | 0.000 | 0.044 |
| hard_negative | 174 | 0.052 | 0.029 | 0.919 |
| facts_only | 3 | 0.000 | 0.000 | 1.000 |

**Do originals outrank copies in search?** Test queries (Wikinews titles), body-only relevance:

| setting | lambda | P@10 | nDCG@10 | original ranked first | originals among relevant top 10 |
| --- | --- | --- | --- | --- | --- |
| relevance only (no g) | 0.0000 | 0.426 | 0.868 | 0.205 | 0.228 |
| net score with g(d) | 0.1000 | 0.424 | 0.939 | 0.874 | 0.230 |

**Does it work on real farms?** NewsGuard-documented pairs (Wayback snapshots) in the full index:

| pair | status | rank of original | p(original) | flagged | traced to original |
| --- | --- | --- | --- | --- | --- |
| gvs_waller | loaded | 1 | 0.999 | True | True |
| topgolf_wired_deals | loaded | 1 | 0.001 | False | False |
| walli_boredpanda_cat | not loaded |  |  |  |  |
| topgolf_people_cohen | not loaded |  |  |  |  |

**Human judgments** (two team members; filled in `evaldata/*.csv`):

| sheet | rows | judged by both | agreement | Cohen's kappa | share derived (agreed) |
| --- | --- | --- | --- | --- | --- |
| live | 79 | 0 | - | - | - |
| hardneg | 384 | 0 | - | - | - |

**Is the crawler correct and polite?**

| measure | value |
| --- | --- |
| requests_page | 6661 |
| requests_robots | 281 |
| docs_saved | 4838 |
| crawl_hours | 2.02 |
| pages_per_minute | 55.1 |
| hosts | 79 |
| robots_audited_fetches | 6515 |
| disallowed_urls_fetched | 0 |
| robots_skips | 110 |
| min_gap_seconds_overall | 0.29 |
| delay_violations | 9 |
| last_delay_violation | 2026-10-06T23:45:28 |
| trap_guards_fired | 374 |
| backoffs | 13 |
| exact_duplicate_docs | 34 |
| near_duplicate_docs | 84 |
| duplicate_rate | 0.033 |
<!-- results:end -->

## What works and what is planned

Works, end to end on live data:

- Polite focused crawl from 35 seeds (RSS, Atom, news sitemaps, two homepages), re-polling feeds every 45 minutes, with a robots.txt audit and per-host gap report (`crawl-report`).
- Content-seen check over the crawl: exact hash, shingles, MinHash, LSH.
- Positional zoned index (title, body, quote), tf-idf and BM25, phrase queries, optional Porter stemming.
- Candidate retrieval from rare (fact-boosted) terms and quote phrases with an earlier-only filter; verification by sentence alignment, order, quotes, names/numbers and shingle containment; provenance by date or, when undated, by coverage asymmetry; copy graph and originality g(d); search with net score.
- Evaluation: synthetic graded rewrites, same-site distractors, mined hard negatives, six baselines with dev-tuned thresholds, ablations, stemming and query variants, undated provenance, search with and without g(d), crawl statistics.
- `suspect`, `pair` and `postings` print every intermediate step for the demo.

Partial:

- Real farm pairs: one NewsGuard-documented pair (Wired -> TopGolf.kr) loads from the Wayback Machine; the NYT original is not archived and the Bored Panda and People originals are not named in the report. The team's pair list is still to be added.
- Hard negatives and flagged live pairs are mined automatically; the judge columns in `evaldata/hard_negatives.csv`, `live_flagged.csv` and `search_judgments.csv` are for the two team members to fill.
- No live farm sites were found among the seeds yet; suspects come from aggregators that republish agency copy (latestly, devdiscourse, socialnews, newsx).

Planned (course project):

1. Adaptive recrawl scheduling from estimated change rates; full Mercator queues with several fetcher threads; distributed crawling.
2. Farm-aware focused crawling: raise the frontier priority of domains with high derivative rates (the `priority()` hook).
3. Cross-lingual derivatives, such as English originals rewritten into Hindi.
4. Learned weights for more signals, trained on judged live pairs instead of synthetic rewrites.
5. A measurement study of how fast farms copy and how long copies outrank originals.
6. A larger index (one CC-NEWS file) and external benchmarks (PAN, METER).

## Timeline

Tracks announced 12:00 on 6 Oct 2026; planning 12:00 to 21:00; build from 21:00.

TIMELINE_PLACEHOLDER

## AI-use log

The team used Claude Code (Anthropic, model Claude Opus 5.5) as a coding agent throughout the build.

- Generated with Claude Code from the team's plan (PLAN.md, CLAUDE.md): the package code under `spintrace/`, the tests, the CLI, the evaluation harness and charts, this README, PROGRESS.md and the critique notes in `notes/`.
- The team wrote the plan, the brief and the seed strategy, chose the track and claim, reviewed the code and results, and judges the live pairs, hard negatives and search results.
- Synthetic SEO rewrites and summaries were generated by Meta Llama 3.2 3B run locally through Ollama (`spintrace/eval/rewrites.py`); they are evaluation data only.
- all-MiniLM-L6-v2 (a pretrained sentence-embedding model) is used for verification signals and the dense baseline.
- No AI system labelled evaluation data: synthetic labels come from construction, hard negatives from a rule-based miner, and live pairs are for human judges.
