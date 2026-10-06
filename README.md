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

RESULTS_PLACEHOLDER

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
