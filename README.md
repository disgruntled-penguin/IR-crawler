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
 [1 crawl] Mercator frontier -> robots.txt + delay -> fetch -> extract   [rewrites] exact, light, synonym,
        |   (spintrace/crawl/)                                            LLM SEO, summary, round-trip translation
        v                                                                 |
 SQLite store keyed by doc_id  <------------------------------------------+
        |
 [2 index] tokenise/normalise -> positional zoned index (title, body, quote), tf-idf, BM25
        |   content-seen: exact hash, 5-gram shingles, MinHash(128), LSH(32x4)
        v
 [3 detect] suspect -> 30 rarest terms (lnc.ltc cosine, term-at-a-time, heap top-20)
        |             + quote windows as phrase queries on the positional index
        |             + parametric filter: other sites, published earlier
        |          -> verify top 5: sentence alignment coverage, order and word overlap, back coverage,
        |             quotes, names/numbers, shingle containment -> logistic score (fit on dev)
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
python -m spintrace newsguard               # NewsGuard-documented farm pairs from the Wayback Machine
python -m spintrace build --stem            # ingest Wikinews + rewrites, shingles/MinHash, both indexes
python -m spintrace eval                    # all experiments; tables and charts in results/
python -m spintrace scan --where "origin IN ('crawl','synthetic','newsguard')"   # copy graph and g(d)
python -m spintrace search-eval             # originals versus copies in search, with and without g(d)
python -m spintrace crawl-report            # politeness and correctness statistics, with example log lines
```

`scripts/run_all.sh` runs build, eval, scan, search-eval, crawl-report and export in that order.

Optional scale experiment (downloads one 1 GB CC-NEWS file): `python -m spintrace ccnews`, then
`python -m spintrace build`, then `python -m spintrace scale`. The main index leaves CC-NEWS out unless
`build --with-ccnews` is given, so the main results do not depend on it.
`python -m spintrace judge live --judge 1` (then `--judge 2`, and the sheets `hardneg` and `search`) is the
labelling loop for the two judges; `python -m spintrace eval --report-only` folds their labels into the results.
`python -m spintrace judge-ui` serves the same sheets as a web page on http://127.0.0.1:8765: both texts side by side,
word runs of eight or more shared by the two underlined, keyboard labels, and no SpinTrace score, verdict or other
judge's label on screen, so the judgments stay blind.

A judge without the crawl database (the other team member, on another machine) uses a pack of just the texts the sheets
refer to: `python -m spintrace judge-pack` writes `evaldata/judge_pack.json` (about 3 MB; it holds crawled full
texts, so it is git-ignored and sent privately, never committed). With the repository and that file in `evaldata/`,
`python -m spintrace judge-ui` on their machine shows the same pages and records labels in their copy of the sheets.
They send those CSVs back and `python -m spintrace judge-merge live evaldata/their_live_flagged.csv` (also `search`,
`hardneg`) copies their `judge2` column in without overwriting any existing label.

`python -m spintrace app` serves the demo web app on http://127.0.0.1:8766 (the first trace loads the models and takes
about 20 seconds). It has three tabs over the same modules as the command line: Search (ranked results with relevance,
originality g(d) and a "copy" badge naming the original; untick g(d) to see the ranking without it), Trace (the
`suspect` steps as a page: rare-term query, candidate sources, verification signals, sentence alignment, verdict;
also reachable as `/?trace=<doc id or URL>`) and Site integrity (the per-site derived share from `scan`). It is a thin
layer: `spintrace/ui/service.py` returns plain dicts and holds no detection logic.

Inspecting intermediate output (for the demo):

```
python -m spintrace postings teheran                 # df, idf and positional postings per zone
python -m spintrace pair <doc_id|url> <doc_id|url>   # shingle overlap, Jaccard, containment, MinHash, LSH bands
python -m spintrace suspect <doc_id|url>             # query terms, candidates, signals, sentence alignment, verdict
python -m spintrace search "nobel prize physics"     # ranked results with relevance, g(d) and net score
python -m spintrace search "nobel prize physics" --no-g
python -m spintrace demo --seed 7                    # seeded random test cases, including a miss and a false alarm
```

`pytest` runs the unit tests (frontier politeness and starvation, text pipeline, index, MinHash, candidate retrieval).

## Data and credits

| Source | Use | Licence and notes |
| --- | --- | --- |
| Live crawl from 36 seeds (`seeds/seeds.csv`) | the index, live detection, hard negatives, search | Only URLs and features are committed (`evaldata/crawl_urls.csv`, `crawl_minhash.npz`); texts stay local. Each seed's robots.txt was checked before use |
| [English Wikinews](https://en.wikinews.org) dump, Oct 2026 | 400 originals for synthetic rewrites | CC BY 2.5, Wikinews contributors. Read from the official Wikimedia dump because Wikinews robots.txt disallows `/w/` (its API) |
| Synthetic rewrites (`evaldata/rewrites.jsonl`) | labelled evaluation set | Generated from the Wikinews originals; LLM levels (SEO rewrite, summary, English-German-English round trip, facts-only article) by Meta Llama 3.2 3B via Ollama |
| NewsGuard August 2023 report | documented real farm rewrites (`evaldata/newsguard_pairs.csv`) | Farm copies and originals fetched from the Internet Archive Wayback Machine |
| all-MiniLM-L6-v2 (sentence-transformers) | sentence embeddings for verification and the dense baseline | Apache 2.0 |
| [Common Crawl CC-NEWS](https://commoncrawl.org/blog/news-dataset-available), one WARC file of 6 Oct 2026 | 3,645 English articles for the scale experiment only (`python -m spintrace ccnews`, `scale`) | Common Crawl terms of use; not committed |

## How IR is used, and where

| Concept | Where | Why SpinTrace needs it |
| --- | --- | --- |
| Crawl loop, URL frontier (Mercator front and back queues), politeness, robots.txt | `crawl/frontier.py`, `crawl/crawler.py`, `crawl/robots.py` | Collecting news and farm pages responsibly. Front queues by priority (feed entries > article-shaped links > other), partitioned by host; a heap of next-allowed times picks the host; `priority()` is the one replaceable policy function |
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
`numpy` holds postings arrays; `matplotlib` draws the charts; `mwparserfromhell` strips wikitext markup from the
Wikinews dump (deciding what the document is for originals); `warcio` reads records from the CC-NEWS WARC file.

Written ourselves: the frontier and politeness logic, URL normalisation, tokenisation, the
positional zoned index, tf-idf and BM25, phrase queries, shingling, MinHash, LSH, candidate
retrieval, the signals, provenance, the copy graph, search and the evaluation.

## Design choices and why

| Choice | Value | Reason |
| --- | --- | --- |
| Politeness delay | max(Crawl-delay, 5 s) per host | The brief asks for "a few seconds"; 5 s keeps ~30 hosts busy at ~1 page/s overall from one process |
| Backoff | delay x2 per consecutive error (cap 64x, 10 min), stop host after 5; errors are 5xx, 429 and refusals (401, 403, 405, 451) | 429/503 ask us to slow down; a host refusing every request should be left alone |
| Frontier | 3 priority levels (feed/sitemap entries, article-shaped links, other) kept per host, drawn with bias 8:3:1; a heap orders hosts by next allowed time | Mercator's front and back queues, with the front queues partitioned by host so no host waits behind another's backlog (a starvation bug in the first version); links are followed only on hosts the seeds and feeds point at |
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
| Evaluation sizes | 400 Wikinews originals, 4,000 distractors, 200 SEO-rewritten, 200 summarised, 100 round-trip translated, 200 facts-only; ~300 crawl hard negatives per split | Stable metrics at hackathon cost: a 1-point F1 change is a handful of items |

## Ethics

Enforced in code (`crawl/`):

- robots.txt fetched per host and obeyed, including Crawl-delay; an unreachable, forbidden or failing (5xx) robots.txt means the host is not crawled until the next check. Every robots.txt fetched is stored, and `crawl-report` re-checks every fetched URL against it.
- At least 5 s between requests to one host (the robots.txt request counts), enforced twice: by the frontier's per-host heap and by a hard guard before every request.
- User agent `SpinTraceBot/0.1 (CSD358 IR course project; contact: ...)`.
- Exponential backoff on 429, 5xx and refusals (401, 403, 405, 451), with Retry-After honoured; a host is stopped after 5 consecutive errors.
- Redirects are not followed automatically, so a redirect target goes back through robots.txt and the delay.
- Comments, author and profile pages, accounts, search and media are never fetched; trafilatura drops comment sections from article text; author names are not stored.
- Crawled full texts are never committed: only URLs, dates, hashes and MinHash signatures, plus `recrawl`.

## Results

Charts: `results/detection_f1_by_level.png`, `pr_curve.png`, `p_at_1_by_level.png`, `index_elimination.png`,
`ablations.png`, `pipeline.png`. Every table below is regenerated from `results/*.csv` by the commands above.

<!-- results:start -->
Generated by `python -m spintrace eval` and friends; every evaluation number below is on the test split.

Index: 20860 documents. Evaluation items (dev / test):

| set | exact | facts_only | hard_negative | light | original | seo | summary | synonym | translation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dev | 210 | 110 | 299 | 210 | 210 | 110 | 110 | 210 | 55 |
| test | 190 | 90 | 301 | 190 | 190 | 90 | 90 | 190 | 45 |

**Is this article derived from another?** Test F1 per rewrite level (negatives pooled), and false-positive rate on same-event negatives. Thresholds tuned on dev.

| method | exact | light | synonym | seo | summary | translation | FPR hard neg (crawl) | FPR facts-only |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| spintrace | 0.960 | 0.960 | 0.960 | 0.782 | 0.902 | 0.816 | 0.007 | 0.156 |
| exact_hash | 0.995 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.022 |
| minhash_jaccard | 0.992 | 0.990 | 0.322 | 0.021 | 0.000 | 0.000 | 0.000 | 0.033 |
| shingle_containment | 0.850 | 0.850 | 0.850 | 0.573 | 0.681 | 0.555 | 0.090 | 0.433 |
| tfidf_cosine | 0.922 | 0.922 | 0.920 | 0.723 | 0.659 | 0.695 | 0.063 | 0.144 |
| bm25 | 0.927 | 0.927 | 0.922 | 0.636 | 0.852 | 0.740 | 0.043 | 0.189 |
| dense_cosine | 0.671 | 0.671 | 0.671 | 0.436 | 0.332 | 0.314 | 0.502 | 0.311 |

SpinTrace precision / recall / source correct per level:

| level | precision | recall | F1 | source correct |
| --- | --- | --- | --- | --- |
| exact | 0.922 | 1.000 | 0.960 | 0.995 |
| light | 0.922 | 1.000 | 0.960 | 0.995 |
| synonym | 0.922 | 1.000 | 0.960 | 1.000 |
| seo | 0.809 | 0.756 | 0.782 | 0.756 |
| summary | 0.845 | 0.967 | 0.902 | 0.967 |
| translation | 0.724 | 0.933 | 0.816 | 0.933 |

**Is its original ranked first?** Test positives, all levels:

| method | P@1 | MRR | Recall@5 | Recall@20 |
| --- | --- | --- | --- | --- |
| spintrace | 0.998 | 0.998 | 0.999 | 1.000 |
| exact_hash | 0.239 | 0.239 | 0.239 | 0.239 |
| minhash_jaccard | 0.526 | 0.526 | 0.526 | 0.526 |
| shingle_containment | 0.987 | 0.991 | 0.995 | 0.998 |
| tfidf_cosine | 0.993 | 0.996 | 0.999 | 1.000 |
| bm25 | 0.996 | 0.998 | 0.999 | 0.999 |
| dense_cosine | 0.961 | 0.979 | 0.999 | 1.000 |

**Index elimination:** source recall from m query terms chosen by idf (test positives):

| query | Recall@1 | Recall@20 | postings touched |
| --- | --- | --- | --- |
| rare3 | 0.805 | 0.894 | 28 |
| rare5 | 0.884 | 0.964 | 70 |
| rare10 | 0.970 | 0.998 | 282 |
| rare30 | 0.994 | 0.999 | 3891 |
| common10 | 0.713 | 0.935 | 48375 |
| random10 | 0.937 | 0.998 | 12365 |

Retrieval variants (Recall@1 / Recall@20, all levels):

| variant | Recall@1 | Recall@20 |
| --- | --- | --- |
| rare terms + quote phrases | 0.993 | 1.000 |
| rare terms only (no quotes) | 0.994 | 1.000 |
| no fact boost | 0.991 | 0.999 |
| no date filter (dates untrusted) | 0.989 | 1.000 |
| rare terms, Porter stemming | 0.996 | 1.000 |
| dense retrieval | 0.961 | 1.000 |
| full-document tf-idf | 0.993 | 1.000 |

**Ablations** (test macro F1 over levels; refit on dev without the signal):

| variant | macro F1 | FPR hard neg | FPR facts-only |
| --- | --- | --- | --- |
| full | 0.896 | 0.007 | 0.156 |
| minus rare_cos | 0.882 | 0.010 | 0.211 |
| minus shingle_containment | 0.892 | 0.007 | 0.167 |
| minus align_coverage | 0.884 | 0.007 | 0.211 |
| minus align_mean | 0.896 | 0.010 | 0.144 |
| minus align_order | 0.885 | 0.010 | 0.178 |
| minus back_coverage | 0.887 | 0.007 | 0.189 |
| minus ordered_coverage | 0.885 | 0.010 | 0.178 |
| minus aligned_word_overlap | 0.900 | 0.007 | 0.144 |
| minus quote_overlap | 0.896 | 0.007 | 0.156 |
| minus fact_overlap | 0.884 | 0.007 | 0.222 |
| no embedding signals (IR signals only) | 0.802 | 0.007 | 0.444 |
| dense candidates instead of IR | 0.894 | 0.007 | 0.156 |
| IR retrieval only, no verification | 0.595 | 0.242 | 0.767 |

**Cost per suspect** (test items):

| method | ms per suspect | docs fully compared | postings touched | earlier docs |
| --- | --- | --- | --- | --- |
| spintrace | 23.2 | 20 | 8293 | 8165 |
| full-document query (tf-idf/BM25) | - | - | 745985 | 8165 |
| exact_hash | 0.0 | - | - | 8165 |
| minhash_jaccard | 0.1 | - | - | 8165 |
| shingle_containment | 58.9 | 8165 | - | 8165 |
| tfidf_cosine | 12.0 | 8165 | - | 8165 |
| bm25 | 10.8 | 8165 | - | 8165 |
| dense_cosine | 1.4 | 8165 | - | 8165 |

**Provenance without dates** (coverage asymmetry on the known pair, margin 0.2):

| level | n | suspect called derived | direction reversed | uncertain |
| --- | --- | --- | --- | --- |
| exact | 190 | 0.000 | 0.000 | 1.000 |
| light | 190 | 0.079 | 0.000 | 0.921 |
| synonym | 190 | 0.000 | 0.000 | 1.000 |
| seo | 90 | 0.167 | 0.033 | 0.800 |
| summary | 90 | 0.956 | 0.000 | 0.044 |
| translation | 45 | 0.178 | 0.022 | 0.800 |
| hard_negative | 301 | 0.053 | 0.030 | 0.917 |
| facts_only | 90 | 0.267 | 0.000 | 0.733 |

**Do originals outrank copies in search?** Test queries (Wikinews titles), body-only relevance:

| setting | lambda | P@10 | nDCG@10 | original ranked first | originals among relevant top 10 |
| --- | --- | --- | --- | --- | --- |
| relevance only (no g) | 0.0000 | 0.533 | 0.837 | 0.100 | 0.195 |
| net score with g(d) | 0.1000 | 0.524 | 0.886 | 0.526 | 0.203 |

**Scale** (same test items, index grown to 24501 documents with 3637 CC-NEWS articles from the same days; verifier and threshold not refit):

| level | Recall@1 | Recall@20 | full-doc tf-idf Recall@1 | F1 / FPR | postings touched | postings, full-doc query |
| --- | --- | --- | --- | --- | --- | --- |
| exact | 1.000 | 1.000 | 1.000 | 0.960 | 9698 | 811153 |
| light | 1.000 | 1.000 | 1.000 | 0.960 | 10694 | 784633 |
| synonym | 1.000 | 1.000 | 1.000 | 0.960 | 6747 | 817589 |
| seo | 0.989 | 1.000 | 0.978 | 0.789 | 9948 | 747305 |
| summary | 0.956 | 1.000 | 0.978 | 0.902 | 21650 | 399229 |
| translation | 0.956 | 1.000 | 0.956 | 0.816 | 12022 | 683412 |
| fpr_original | - | - | - | 0.000 | - | - |
| fpr_hard_negative | - | - | - | 0.007 | - | - |
| fpr_facts_only | - | - | - | 0.156 | - | - |

**Does it work on real farms?** NewsGuard-documented pairs (Wayback snapshots) in the full index:

| pair | status | rank of original | p(original) | flagged | traced to original |
| --- | --- | --- | --- | --- | --- |
| gvs_waller | loaded | 1 | 0.999 | True | True |
| topgolf_wired_deals | loaded | 1 | 0.0 | False | False |
| walli_boredpanda_cat | not loaded |  |  |  |  |
| topgolf_people_cohen | not loaded |  |  |  |  |

**Per-domain integrity** (live crawl; share of a site's articles flagged as derived, top 10):

| site | articles | flagged derived | share | main sources |
| --- | --- | --- | --- | --- |
| devdiscourse.com | 692 | 20 | 0.029 | indiatimes.com (11); independent.co.uk (4); socialnews.xyz (2) |
| theprint.in | 400 | 11 | 0.028 | indiatimes.com (4); hindustantimes.com (3); devdiscourse.com (1) |
| independent.co.uk | 608 | 15 | 0.025 | abcnews.com (9); hindustantimes.com (2); france24.com (1) |
| thehindu.com | 449 | 11 | 0.025 | socialnews.xyz (3); theprint.in (3); abcnews.com (2) |
| indiatimes.com | 2324 | 55 | 0.024 | devdiscourse.com (17); hindustantimes.com (13); indianexpress.com (10) |
| latestly.com | 632 | 11 | 0.017 | dw.com (5); socialnews.xyz (3); abc.net.au (1) |
| newsx.com | 316 | 5 | 0.016 | devdiscourse.com (4); theprint.in (1) |
| scroll.in | 585 | 9 | 0.015 | theconversation.com (9) |
| indianexpress.com | 548 | 8 | 0.015 | hindustantimes.com (2); indiatimes.com (2); devdiscourse.com (1) |
| cbsnews.com | 339 | 4 | 0.012 | france24.com (3); abcnews.com (1) |

**Human judgments** (two team members; labels in `evaldata/*.csv`). Positive means derived on the live flags and relevant on search. The team's convention counts a pair positive if either judge said so; the strict column needs both. Agreement and kappa are measured before any combining:

| sheet | rows | judged by both | agreement | Cohen's kappa | judge 1 positive | judge 2 positive | positive (either judge) | positive (both judges) | positive (agreed pairs only) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| live | 174 | 174 | 0.592 | 0.123 | 0.592 | 0.690 | 0.845 | 0.437 | 0.738 |
| hardneg | 600 | 0 | - | - | - | - | - | - | - |
| search | 154 | 154 | 0.636 | 0.161 | 0.929 | 0.578 | 0.935 | 0.571 | 0.898 |

**Search on the live queries, judged** (15 queries, top 10 with and without g(d); pooled judgments):

| judges | setting | queries | P@10 | nDCG@10 | original ranked first | originals among relevant top 10 |
| --- | --- | --- | --- | --- | --- | --- |
| either says relevant | relevance only | 15 | 0.933 | 0.973 | 0.867 | 0.927 |
| either says relevant | with g(d) | 15 | 0.933 | 0.973 | 0.867 | 0.953 |
| both say relevant | relevance only | 15 | 0.573 | 0.811 | 0.600 | 0.924 |
| both say relevant | with g(d) | 15 | 0.560 | 0.803 | 0.600 | 0.963 |

**Is the crawler correct and polite?**

| measure | value |
| --- | --- |
| requests_page | 43683 |
| requests_robots | 620 |
| docs_saved | 32513 |
| crawl_hours | 13.49 |
| pages_per_minute | 54.0 |
| hosts | 118 |
| fresh_articles | 968 |
| freshness_median_minutes | 69.7 |
| freshness_p90_minutes | 439.4 |
| robots_audited_fetches | 43674 |
| disallowed_urls_fetched | 0 |
| robots_skips | 512 |
| min_gap_seconds_overall | 0.29 |
| delay_violations | 9 |
| last_delay_violation | 2026-10-06T23:45:28 |
| trap_guards_fired | 2569 |
| backoffs | 77 |
| exact_duplicate_docs | 792 |
| near_duplicate_docs | 1099 |
| duplicate_rate | 0.0695 |
<!-- results:end -->

## Evaluation protocol

- **Items.** Positives are synthetic rewrites of Wikinews originals at six intensities (exact copy, light edit,
  WordNet synonym spin, LLM "SEO-friendly" rewrite, LLM summary, LLM round-trip translation through German). Negatives are the originals themselves,
  crawl hard negatives (same event, other outlet, published within 3 days, at least 3 shared rare names, no
  shingle containment above 0.1 against any earlier document: median gap 10.5 h, 77% within 24 h), and
  facts-only articles an LLM wrote from an original's topic, names, numbers and quotes without seeing its text.
- **Index.** Every item is scored against the full index (crawl, originals, 4,000 Wikinews distractors, all
  rewrites), except that a rewrite cannot see the other rewrites of its own original.
- **Splits.** By original story; the verifier and every method's threshold are fit on dev, all numbers are test.
- **Baselines.** Exact hash; MinHash Jaccard over LSH candidates; shingle containment against every earlier
  document (the best case for shingles); full-document tf-idf cosine; full-document BM25 normalised by the
  suspect's self-score; dense cosine of mean MiniLM sentence vectors. Each takes its best dev threshold.
- **Real cases.** NewsGuard-documented pairs (Wayback Machine), flagged live pairs and live search results for
  the two judges (`python -m spintrace judge ...`); agreement is reported as Cohen's kappa.

## Known weaknesses

- **Where SpinTrace loses.** On exact copies and light edits, exact hashing and MinHash are better (F1 0.99-1.00
  against 0.960) and far cheaper: SpinTrace's threshold is tuned for paraphrases, so the same-story negatives it
  still flags cost it precision on every level. In a crawler the two should run together: the textbook check first,
  SpinTrace for what it lets through. BM25 comes within 0.05 F1 on summaries, and full-document tf-idf flags
  slightly fewer facts-only articles (14.4% against 15.6%) while missing far more rewrites.
- **Same story versus copied is the hard boundary.** Articles written from an original's facts and quotes alone
  are still flagged 15.6% of the time (test), and SEO-rewrite recall is 0.76 at the dev-tuned threshold. Before
  the facts-only negatives were added, a spot check of 8 live flags found about 3 real derivatives.
- **Live precision is a range, because the judges disagreed.** Both team members labelled all 174 flagged live pairs.
  A pair counts as derived if either judge said so (the team's convention): precision 0.845. If both must say so, it
  is 0.437; on the 103 pairs they agreed on, 0.738. They agreed on only 59% of pairs (Cohen's kappa 0.12; Judge 1
  called 103 derived, Judge 2 called 120), so the number depends on how the labels are combined and we report all
  three. The either-judge rule is the lenient one and raises precision by construction. Judge 1's not-derived pairs
  have much less exact text overlap (median shingle containment 0.38 against 0.85 for derived), which fits the
  sentence-embedding alignment matching independent coverage of one event; the flags cluster on a few site pairs
  (30 of the 55 flagged indiatimes.com articles).
- **Shared quotes.** The judges' guide does not count a shared direct quote alone as copying, because independent
  outlets quote the same speaker. In the flagged pairs quoted sentences are a small share (median 5% of sentences)
  and dropping them leaves every flag in place (an exploratory check, not tuned), but the mined hard negatives share
  no text, so a same-event pair that shares quotes is not tested yet.
- **Judged search does not separate the rankings.** On the 15 live queries, 93.5% of judged results were marked
  relevant when either judge said so (57% when both), so P@10 (0.933) and nDCG@10 (0.973) are the same with and
  without g(d); g(d) only raises the share of originals among the relevant top 10, from 0.927 to 0.953. With both
  judges required, nDCG@10 is 0.811 without g(d) and 0.803 with it. The clear ranking gain is on the synthetic
  queries (original ranked first for 52.6% of test queries against 10%).
- **Shared wire copy is attributed to an outlet.** When two outlets print the same AP or PTI story, SpinTrace
  names the earlier outlet as the source; the agency itself is not crawled.
- **Verification is trained on prose.** The NewsGuard TopGolf.kr page (AI title suggestions plus a reshaped
  deals list) has its Wired source ranked first by retrieval but is not verified.
- **Undated provenance mostly abstains** for same-length rewrites; it works for summaries.
- **Dates are trusted.** A farm that backdates its pages escapes the earlier-only filter; the "no date filter"
  retrieval variant measures what dropping the filter costs.
- **Ranking is easy for sparse methods on this corpus.** Whole-document tf-idf and BM25 also rank the original
  first (P@1 about 0.99); the IR evidence is in cost (about 1% of the postings a full-document query touches) and in
  the index-elimination experiment. Dense retrieval is the one that slips: with same-story distractors in the
  index it ranks the true source first for only 0.83 of SEO and summary rewrites.
- **Synthetic rewrites come from one 3B model.** Stronger LLMs paraphrase more and would be harder.
- **Scale.** One crawler process, 32.5k crawled articles over 13.5 hours (evaluations used the first ~16k). Growing the index by 3,637 CC-NEWS articles
  from the same days (to 24.5k documents) left retrieval and detection within a point of the main results, but
  that is a small step, not a web-scale test.

## What works and what is planned

Works, end to end on live data:

- Polite focused crawl from 36 seeds (RSS and Atom feeds and three homepages), re-polling feeds every 45 minutes (articles published during the crawl reached the store a median 70 minutes after publication), with a robots.txt audit and per-host gap report (`crawl-report`).
- Content-seen check over the crawl: exact hash, shingles, MinHash, LSH.
- Positional zoned index (title, body, quote), tf-idf and BM25, phrase queries, optional Porter stemming.
- Candidate retrieval from rare (fact-boosted) terms and quote phrases with an earlier-only filter; verification by sentence alignment, order, quotes, names/numbers and shingle containment; provenance by date or, when undated, by coverage asymmetry; copy graph and originality g(d); search with net score.
- Evaluation: synthetic graded rewrites, same-site distractors, mined hard negatives, six baselines with dev-tuned thresholds, ablations, stemming and query variants, undated provenance, search with and without g(d), crawl statistics.
- `suspect`, `pair` and `postings` print every intermediate step for the demo; the web app shows the same steps for any indexed article.

Partial:

- Real farm pairs: two NewsGuard-documented pairs load from the Wayback Machine. NYT -> GlobalVillageSpace is retrieved first among 20.9k documents, verified (p = 0.999) and traced; Wired -> TopGolf.kr is retrieved first but not verified. The Bored Panda and People originals are not named in the report.
- Hard negatives and flagged live pairs are mined automatically. Both team members have judged the flagged live pairs (`evaldata/live_flagged.csv`) and the search results (`search_judgments.csv`); the hard-negative sheet (`hard_negatives.csv`) is not judged yet.
- No live farm sites were found among the seeds yet; suspects come from aggregators that republish agency copy (latestly, devdiscourse, socialnews, newsx).

Planned (course project):

1. Adaptive recrawl scheduling from estimated change rates; full Mercator queues with several fetcher threads; distributed crawling.
2. Farm-aware focused crawling: raise the frontier priority of domains with high derivative rates (the `priority()` hook).
3. Cross-lingual derivatives, such as English originals rewritten into Hindi.
4. Learned weights for more signals, trained on judged live pairs instead of synthetic rewrites.
5. A measurement study of how fast farms copy and how long copies outrank originals.
6. A much larger index (many CC-NEWS files, with streaming index construction) and external benchmarks (PAN, METER).

## Timeline

Tracks announced 12:00 on 6 Oct 2026; planning 12:00 to 21:00; build from 21:00. Times are IST, from the commit log.

| Time | Milestone |
| --- | --- |
| 6 Oct 15:12-22:13 | Plan, brief and permissions committed |
| 22:20 | Network check passed; package skeleton |
| 22:29 | Live crawl started (runs in the background from here on) |
| 22:38 | Wikinews dump loader and graded rewrites (LLM levels generating locally) |
| 22:47 | Positional index, MinHash/LSH, candidate retrieval, verification, provenance |
| 23:30 | Copy graph, search with g(d), inspect commands |
| 23:46 | Evaluation harness with baselines; politeness race found by the crawl report and fixed |
| 7 Oct 00:05 | Critique round 1: hard-negative labels, distractors, all-pairs shingle baseline, IR-only ablation |
| 00:30 | Facts-only independent-coverage negatives; structural signals |
| 00:36 | NewsGuard pairs from the Wayback Machine |
| 01:04 | Critique round 2: dates-untrusted variant, refusal backoff |
| 01:12 | Frontier starvation found and fixed |
| 02:00-02:30 | Final run; critique round 3 (dev cross-validation adds aligned word overlap) |
| 02:20-02:50 | CC-NEWS scale experiment; the crawler died on a database lock held by the loader, fixed and restarted |
| 02:55-04:30 | Round-trip translation level generated; full rerun of every experiment with it |

## AI-use log

The team used Claude Code (Anthropic, model Claude Opus 5.5) as a coding agent throughout the build.

- Generated with Claude Code from the team's plan (PLAN.md, CLAUDE.md): the package code under `spintrace/`, the tests, the CLI, the evaluation harness and charts, this README, PROGRESS.md and the critique notes in `notes/`.
- The team wrote the plan, the brief and the seed strategy, chose the track and claim, reviewed the code and results, and judges the live pairs, hard negatives and search results.
- Synthetic SEO rewrites, summaries, round-trip translations and facts-only articles were generated by Meta Llama 3.2 3B run locally through Ollama (`spintrace/eval/rewrites.py`); they are evaluation data only.
- all-MiniLM-L6-v2 (a pretrained sentence-embedding model) is used for verification signals and the dense baseline.
- No AI system labelled evaluation data: synthetic labels come from construction, hard negatives from a rule-based miner, and live pairs are for human judges.
