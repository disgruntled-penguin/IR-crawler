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

**Final project: what is different.** The mid-term system, like every copy detector we know of, assumes
that a derivative shares wording with one original. Two kinds of derivative break that assumption, and
both are what a content farm has most reason to produce:

1. **Translations.** A Hindi (or Tamil, Telugu, Bengali) article translated from an English one shares
   no words with it, so shingles, Jaccard, tf-idf and quote phrases all fail. SpinTrace traces it
   through what survives translation: numbers, dates and names, with its own Indian-script
   transliteration, a phonetic key and a k-gram name index, over the same inverted index. No translation
   step and no embedding of the corpus; a multilingual sentence model only verifies the few candidates.
2. **Mashups.** One article stitched from two or three originals has no single source, and a detector
   that returns its best match finds one and stops. SpinTrace queries again with only the sentences
   no source has explained yet, and reports every source with the share of the article it explains.

What we claim is narrower than "new": cross-language plagiarism detection and finding all the sources
of a suspicious document are established research areas (CL-ESA, cross-language n-grams, PAN source
retrieval). The claim is that the premise "facts survive rewriting", run on a crawler's own positional
inverted index for news, extends to both, and where it beats the obvious alternatives and where it does
not is measured below (see the Cross-language and Multi-source sections and Known weaknesses). The
report is `report/SpinTrace_final_report.tex`; the dataset is described in `results/dataset.md`.

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
python -m spintrace eval-multi              # multi-source mashups: rounds, baselines, ablations (results/multisource_*)
python -m spintrace eval-multi --live       # multi-source pass over crawled articles (unjudged counts, evaldata/live_multisource.csv)
python -m spintrace format-eval             # format changes (listicle, Q&A): midterm verifier against refits (results/format_*)
```

`eval-multi` regenerates `evaldata/mashups.jsonl` if it is missing, reads whatever LLM mashups `evaldata/rewrites_xl.jsonl`
holds, opens the database read-only, and caches each pass in `data/eval/ms_*.pkl` (`--fresh` recomputes them).

`scripts/run_all.sh` runs build, eval, scan, search-eval, crawl-report and export in that order.

Final-phase data, evaluation and report (the order that was used):

```
python -c "from spintrace.eval import xl_rewrites; xl_rewrites.build()"   # Hindi, Tamil, Telugu, Bengali rewrites,
                                            # back-translations and LLM mashups (needs Ollama with gemma3:4b; resumable,
                                            # about 30 s per item, cached in evaldata/rewrites_xl.jsonl)
bash scripts/final_run.sh                   # eval-multi, eval-multi --live, eval-xl, live-xl, dataset summary, report tables and figures
python -m spintrace.eval.dataset_stats      # results/dataset.md and dataset.json: what is in the store and the label files
python report/make_tables.py                # LaTeX tables and number macros from results/ (report/tables/)
python report/make_figures.py               # report charts (report/figures/)
tectonic report/SpinTrace_final_report.tex  # the report PDF
```

The report build uses macOS system fonts (Charter, Avenir Next, Menlo, Devanagari Sangam MN), so it builds as is on a
Mac and needs those fonts replaced elsewhere. `scripts/final_run.sh` took 16 minutes on an 8 GB M1; each step caches
its work per item, so a rerun after more rewrites are generated only adds the new items.

Cross-language tracing (Indian-language rewrites of English news; see the section below):

```
python -m spintrace build-xl --dense         # name k-gram lexicon (2 min) and dense heads for the baseline (15 min)
python -m spintrace eval-xl --fresh          # cross-language evaluation, 2-4 s per item; tables and charts in results/crosslingual_*
python -m spintrace live-xl --rebuild        # every crawled Indian-language page against the English crawl, about 2 s per page
```

`eval-xl` reads whatever rows `evaldata/rewrites_xl.jsonl` holds (`SPINTRACE_XL_ROWS` points elsewhere) and caches
each scored item in `data/eval/xl_raw.pkl`, so a rerun only scores new rows unless `--fresh` is given;
`--report-only` refits and rewrites the tables from the cache in seconds. Times are for an 8 GB Apple M1 on CPU
with the generation queue running.

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
python -m spintrace sources mash:bea3d8200d0e:splice_light   # multi-source rounds: residual query with idf, candidates,
python -m spintrace sources mash:b3fc93ab4516:splice_seo     #   verification, covered sentences, chosen source, final shares
python -m spintrace sources <doc_id|url>             # the same for an indexed article (a crawled page)
python -m spintrace kgram "नरेंद्र मोदी"               # transliteration, phonetic key, k-gram postings, matches
python -m spintrace suspect-xl <rid|doc_id|url>      # numbers, names, query, candidates, alignment, verdict
python -m spintrace suspect-xl <rid> --absent        # the same suspect with its true source masked out
```

The two mashup ids above were drawn at random from the test split (seed 358), not chosen for how they come out.

`pytest` runs the unit tests (frontier politeness and starvation, text pipeline, index, MinHash, candidate retrieval).

## Data and credits

| Source | Use | Licence and notes |
| --- | --- | --- |
| Live crawl from 36 seeds (`seeds/seeds.csv`) | the index, live detection, hard negatives, search | Only URLs and features are committed (`evaldata/crawl_urls.csv`, `crawl_minhash.npz`); texts stay local. Each seed's robots.txt was checked before use |
| [English Wikinews](https://en.wikinews.org) dump, Oct 2026 | 400 originals for synthetic rewrites | CC BY 2.5, Wikinews contributors. Read from the official Wikimedia dump because Wikinews robots.txt disallows `/w/` (its API) |
| Synthetic rewrites (`evaldata/rewrites.jsonl`) | labelled evaluation set | Generated from the Wikinews originals; LLM levels (SEO rewrite, summary, English-German-English round trip, facts-only article) by Meta Llama 3.2 3B via Ollama |
| NewsGuard August 2023 report | documented real farm rewrites (`evaldata/newsguard_pairs.csv`) | Farm copies and originals fetched from the Internet Archive Wayback Machine |
| Rule-based mashups (`evaldata/mashups.jsonl`) and format rewrites (`evaldata/formats.jsonl`) | multi-source and format-change evaluation | Spliced and reformatted from the Wikinews originals and distractors (CC BY 2.5) and the committed SEO rewrites; deterministic, regenerated by `spintrace/eval/mashups.py` and `formats.py` |
| all-MiniLM-L6-v2 (sentence-transformers) | sentence embeddings for verification and the dense baseline | Apache 2.0 |
| Cross-language rewrites (`evaldata/rewrites_xl.jsonl`) | Hindi translation, SEO rewrite, summary and facts-only article; Tamil, Telugu and Bengali translations; LLM back-translations of the test-split Hindi items | Generated from the Wikinews originals by Google Gemma 3 4B via Ollama (`spintrace/eval/xl_rewrites.py`); evaluation labels only |
| paraphrase-multilingual-MiniLM-L12-v2 (sentence-transformers) | cross-language sentence alignment (verification) and the multilingual dense baseline | Apache 2.0 |
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
| Residual (explain-away) queries | `detect/multisource.py` | A mashup's minor source is outweighed in one query for the whole suspect; a query built only from the sentences no found source explains makes that source's rare terms and quotes the query |
| Static quality g(d) and net score; zone weighting | `search.py`, `detect/scan.py` | Originals outrank their copies in search |
| BM25, dense retrieval (out of syllabus) | `index.py`, `eval/run.py` | Baselines answering when sparse retrieval wins |
| Precision, recall, F1, P@k, MRR, Recall@k | `eval/report.py` | Every evaluation |
| Tokenisation for Indic scripts; number and date normalisation | `text_indic.py` | A translation keeps numbers and dates but writes them differently (`३.५ करोड़` = 35 million, `26 जुलाई`), so both sides are canonicalised before they meet in the index |
| Tolerant retrieval: k-gram index, edit distance, phonetic keys | `kgram.py`, `translit.py` | A transliterated name never spells the English name exactly; the k-gram index over phonetic keys narrows the candidates through postings and Levenshtein distance confirms (IIR ch. 3) |
| Cross-language anchor query | `detect/crosslingual.py` | Names, numbers and dates are the high-idf terms a translation keeps; the same lnc.ltc cosine and phrase queries find the English source |

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
retrieval, the signals, provenance, the copy graph, search and the evaluation. For the
cross-language extension also the Indic tokeniser, digit and lakh/crore number bridge, the
transliteration table and schwa deletion, the phonetic key, the k-gram index and Levenshtein
distance, and the anchor query; no transliteration or Indic NLP library is used. The multilingual
MiniLM model (sentence-transformers) maps a sentence and its translation to nearby vectors; it only
verifies the five candidates the index returns, plus one head vector per document for the dense baseline.

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
| Cross-language query | numbers, dates, Latin tokens and names; weight = anchor type (1, 0.5, 1, 1) x match confidence x idf; top 40 terms | A dev-only sweep: MRR plateaus at 40 terms (0.97 against 0.53 at 10); other type weights were within noise, so the defaults stayed |
| Phonetic key | aspirates, ph/f, w/v, z/s, sh/s, 'ch' and English c folded, doubles collapsed, vowels and glides dropped after the first letter; voicing also folded for Tamil | `नरेंद्र मोदी` and `Narendra Modi` both give `nrndr md`; Tamil does not write voicing, but folding it everywhere would merge Biden with Putin |
| k-gram lookup | k = 3 with `$` boundaries, Jaccard >= 0.3, then edit distance <= 2 (<= 1 for keys of 3 letters or fewer), 3 matches per word | Jaccard is cheap through postings; edit distance confirms; short keys need the tighter bound |
| Name df check | drop a match when the Indic word is in >= 3 crawled pages of its script and the English term is rarer (higher idf) than the Indic word | A name keeps roughly its document frequency across languages; रात (night) is not Rohit. The gap was swept on dev only (MRR 1.000 at 0, 0.962 at 0.5, 0.912 at 1). The cost: a name far more common in Hindi news than in our English crawl (मोदी) stops being an anchor |
| Cross-language alignment | sentence pair matches if multilingual cosine >= 0.6 | Translated sentences sit around 0.7-0.9 with this model, unrelated ones below 0.5 |
| Time window | 30 days for live pages; none in the evaluation | Rewrites follow within days; in the evaluation the window would leave about 300 earlier documents (the crawl is mostly Sep-Oct 2026), so every method searches all ~9,000 earlier documents instead |
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
| requests_page | 45116 |
| requests_robots | 1014 |
| docs_saved | 33402 |
| crawl_hours | 119.52 |
| pages_per_minute | 6.3 |
| hosts | 118 |
| fresh_articles | 1382 |
| freshness_median_minutes | 96.9 |
| freshness_p90_minutes | 495.9 |
| robots_audited_fetches | 45107 |
| disallowed_urls_fetched | 0 |
| robots_skips | 618 |
| min_gap_seconds_overall | 0.29 |
| delay_violations | 39 |
| last_delay_violation | 2026-10-07T23:16:00 |
| trap_guards_fired | 2616 |
| backoffs | 105 |
| exact_duplicate_docs | 792 |
| near_duplicate_docs | 1099 |
| duplicate_rate | 0.0695 |
<!-- results:end -->

## Cross-language tracing: Indian-language rewrites of English news

A content farm can translate an English article into Hindi, Tamil, Telugu or Bengali. The translation
shares no word with its source, so exact hashing, shingles, Jaccard, tf-idf and quote phrases have
nothing to match. What survives translation is what a paraphrase keeps: numbers, dates and names.
SpinTrace-XL turns those into a query against the same English inverted index, and uses a multilingual
sentence model only to verify the five candidates the index returns.

```
 Hindi / Tamil / Telugu / Bengali suspect
        |  text_indic.py: NFC (vowel signs kept), Indic digits -> ASCII, danda-aware sentences
        v
 anchors  numbers   '3.5 करोड़' -> 35000000, '२२' -> 22, 'चार मिलियन' -> 4000000 ('35 million' on the English side)
          dates     '26 जुलाई' -> (7, 26) -> month term + phrase queries 'july 26' / '26 july'
          latin     tokens left in Latin script (UAE, ADF, Wedgetail)
          names     translit.py: one ISCII-offset table for nine scripts, schwa deletion, nukta, anusvara
                    -> phonetic key ('अल्बानेज़' -> albaanez -> albns = key('Albanese'))
                    -> kgram.py: 3-gram index over the keys of every English name word,
                       Jaccard >= 0.3, then Levenshtein <= 2 -> English index terms
                    -> source-language df check: a word common in crawled Hindi pages is not a rare name
        v
 query    weight = anchor type x match confidence x idf, top 40 terms (index elimination), lnc.ltc cosine
          over earlier documents from other sites; the shortlist is reranked by date and name-bigram phrase hits
        v
 verify   top 5: number, name, date and Latin overlap in both directions; multilingual sentence alignment
          (paraphrase-multilingual-MiniLM-L12-v2) coverage, order and back coverage -> logistic, fit on Hindi dev
        v
 provenance  pipeline.provenance unchanged (earlier date, or coverage asymmetry when undated)
```

`pipeline.analyse` routes every document whose script is not Latin (the crawler now tags `meta.script`) to
this path, so `scan` and the web app's Trace tab use it too; the English path is unchanged.

**Evaluation.** Gemma 3 4B wrote, for Wikinews originals alternating dev and test, a Hindi translation, an
"SEO-friendly" Hindi rewrite, a Hindi summary and a Hindi facts-only article (an independent article from the
original's fact sheet: same event, same names and numbers, not derived), and translations into Tamil, Telugu and
Bengali for a few originals. Each Hindi derivative is also scored a second time with its source (and the source's
English rewrites) masked out of the index: a source-absent negative that must not be flagged. Every method searches
all earlier English documents from other sites (about 9,000), splits are by original, the verifier and every
threshold come from Hindi dev, and every number below is test, with 95% bootstrap intervals over items.
Translate-then-detect runs the English methods on the LLM's back-translation of each test item; no dev item was
back-translated, so those methods keep the thresholds they were given on the English dev split, and an oracle row
(threshold chosen on test) bounds what tuning could add.

Charts: `results/crosslingual_f1_by_level.png`, `results/crosslingual_p_at_1_by_level.png`. The tables below are
rewritten by `eval-xl` from `results/crosslingual_*.csv`.

<!-- xl-results:start -->
Generated by `python -m spintrace eval-xl`; every number is on the test split. Items (dev / test):

| set | bn_translation | hi_facts_only | hi_facts_short | hi_seo | hi_summary | hi_translation | source_absent | ta_translation | te_translation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dev | 12 | 22 | 0 | 22 | 22 | 22 | 66 | 12 | 12 |
| test | 12 | 22 | 22 | 22 | 22 | 22 | 66 | 12 | 12 |

Verifier features chosen by 5-fold grouped cross-validation on dev: `no_anchor_signals` (threshold 0.875). all: dev CV macro F1 0.804; no_anchor_signals: dev CV macro F1 0.836; no_retrieval_score: dev CV macro F1 0.804.

**Is this Hindi article derived from an English one?** F1 per level [95% bootstrap CI] with the negatives pooled; false-positive rate on each negative set [95% Wilson CI].

| method | hi_translation | hi_seo | hi_summary | FPR facts-only | FPR facts-only (100 words) | FPR source absent |
| --- | --- | --- | --- | --- | --- | --- |
| spintrace_xl | 0.957 [0.88, 1.00] | 0.933 [0.84, 1.00] | 0.884 [0.76, 0.97] | 0.000 [0.00, 0.15] | 0.091 [0.03, 0.28] | 0.000 [0.00, 0.06] |
| dense_cosine | 0.422 [0.29, 0.55] | 0.404 [0.27, 0.53] | 0.404 [0.27, 0.53] | 0.773 [0.57, 0.90] | 0.727 [0.52, 0.87] | 0.242 [0.16, 0.36] |
| latin_only | 0.356 [0.23, 0.48] | 0.391 [0.26, 0.51] | 0.103 [0.02, 0.21] | 0.727 [0.52, 0.87] | 0.773 [0.57, 0.90] | 0.288 [0.19, 0.41] |
| bt_spintrace_en | 0.936 [0.84, 1.00] | 0.864 [0.73, 0.96] | 0.780 [0.61, 0.90] | 0.045 [0.01, 0.22] | 0.091 [0.03, 0.28] | 0.000 [0.00, 0.06] |
| bt_spintrace_en@oracle | 0.898 [0.79, 0.98] | 0.851 [0.72, 0.95] | 0.826 [0.70, 0.94] | 0.091 [0.03, 0.28] | 0.136 [0.05, 0.33] | 0.000 [0.00, 0.06] |
| bt_exact_hash | 0.000 [0.00, 0.00] | 0.000 [0.00, 0.00] | 0.000 [0.00, 0.00] | 0.000 [0.00, 0.15] | 0.000 [0.00, 0.15] | 0.000 [0.00, 0.06] |
| bt_exact_hash@oracle | 0.000 [0.00, 0.00] | 0.000 [0.00, 0.00] | 0.000 [0.00, 0.00] | 0.000 [0.00, 0.15] | 0.000 [0.00, 0.15] | 0.000 [0.00, 0.06] |
| bt_minhash_jaccard | 0.000 [0.00, 0.00] | 0.000 [0.00, 0.00] | 0.000 [0.00, 0.00] | 0.000 [0.00, 0.15] | 0.000 [0.00, 0.15] | 0.000 [0.00, 0.06] |
| bt_minhash_jaccard@oracle | 0.000 [0.00, 0.00] | 0.000 [0.00, 0.00] | 0.000 [0.00, 0.00] | 0.000 [0.00, 0.15] | 0.000 [0.00, 0.15] | 0.000 [0.00, 0.06] |
| bt_tfidf_cosine | 0.898 [0.79, 0.97] | 0.826 [0.69, 0.93] | 0.457 [0.22, 0.65] | 0.091 [0.03, 0.28] | 0.136 [0.05, 0.33] | 0.000 [0.00, 0.06] |
| bt_tfidf_cosine@oracle | 0.815 [0.68, 0.92] | 0.769 [0.63, 0.88] | 0.667 [0.49, 0.82] | 0.227 [0.10, 0.43] | 0.227 [0.10, 0.43] | 0.000 [0.00, 0.06] |
| bt_bm25 | 0.710 [0.56, 0.83] | 0.644 [0.48, 0.78] | 0.689 [0.54, 0.81] | 0.227 [0.10, 0.43] | 0.591 [0.39, 0.77] | 0.000 [0.00, 0.06] |
| bt_bm25@oracle | 0.800 [0.67, 0.91] | 0.706 [0.55, 0.85] | 0.731 [0.57, 0.86] | 0.136 [0.05, 0.33] | 0.364 [0.20, 0.57] | 0.000 [0.00, 0.06] |

SpinTrace-XL per level [95% Wilson CI]:

| level | positives | precision | recall | source correct |
| --- | --- | --- | --- | --- |
| hi_translation | 22 | 0.917 [0.74, 0.98] | 1.000 [0.85, 1.00] | 1.000 |
| hi_seo | 22 | 0.913 [0.73, 0.98] | 0.955 [0.78, 0.99] | 0.955 |
| hi_summary | 22 | 0.905 [0.71, 0.97] | 0.864 [0.67, 0.95] | 0.864 |
| ta_translation | 12 | 0.000 [0.00, 0.66] | 0.000 [0.00, 0.24] | 0.000 |
| te_translation | 12 | 0.000 [0.00, 0.66] | 0.000 [0.00, 0.24] | 0.000 |
| bn_translation | 12 | 0.333 [0.06, 0.79] | 0.083 [0.01, 0.35] | 0.083 |

**Is the English original ranked first?** Test positives:

| method | level | n | P@1 [95% Wilson CI] | MRR | Recall@5 | Recall@20 |
| --- | --- | --- | --- | --- | --- | --- |
| spintrace_xl | ta_translation | 12 | 0.417 [0.19, 0.68] | 0.426 | 0.417 | 0.500 |
| spintrace_xl | te_translation | 12 | 0.583 [0.32, 0.81] | 0.688 | 0.833 | 0.833 |
| spintrace_xl | bn_translation | 12 | 0.917 [0.65, 0.99] | 0.938 | 1.000 | 1.000 |
| spintrace_xl | all_hi | 66 | 1.000 [0.94, 1.00] | 1.000 | 1.000 | 1.000 |
| dense_cosine | ta_translation | 12 | 0.083 [0.01, 0.35] | 0.167 | 0.250 | 0.250 |
| dense_cosine | te_translation | 12 | 0.083 [0.01, 0.35] | 0.125 | 0.167 | 0.167 |
| dense_cosine | bn_translation | 12 | 0.083 [0.01, 0.35] | 0.125 | 0.167 | 0.167 |
| dense_cosine | all_hi | 66 | 0.924 [0.83, 0.97] | 0.956 | 0.985 | 1.000 |
| latin_only | ta_translation | 12 | 0.417 [0.19, 0.68] | 0.427 | 0.417 | 0.500 |
| latin_only | te_translation | 12 | 0.417 [0.19, 0.68] | 0.431 | 0.417 | 0.583 |
| latin_only | bn_translation | 12 | 0.750 [0.47, 0.91] | 0.798 | 0.833 | 0.917 |
| latin_only | all_hi | 66 | 0.500 [0.38, 0.62] | 0.511 | 0.515 | 0.576 |
| bt_spintrace_en | all_hi | 66 | 0.985 [0.92, 1.00] | 0.985 | 0.985 | 0.985 |
| bt_exact_hash | all_hi | 66 | 0.000 [0.00, 0.06] | 0.000 | 0.000 | 0.000 |
| bt_minhash_jaccard | all_hi | 66 | 0.000 [0.00, 0.06] | 0.000 | 0.000 | 0.000 |
| bt_tfidf_cosine | all_hi | 66 | 1.000 [0.94, 1.00] | 1.000 | 1.000 | 1.000 |
| bt_bm25 | all_hi | 66 | 1.000 [0.94, 1.00] | 1.000 | 1.000 | 1.000 |

**Ablations**, one change at a time (test, Hindi levels; every variant refit on dev):

| variant | macro F1 | P@1 | Recall@5 | FPR facts-only | FPR source absent | postings touched |
| --- | --- | --- | --- | --- | --- | --- |
| full | 0.925 | 1.000 | 1.000 | 0.000 | 0.000 | 81488 |
| no_names | 0.873 | 0.864 | 0.864 | 0.000 | 0.000 | 15203 |
| no_numbers | 0.898 | 0.955 | 0.955 | 0.000 | 0.000 | 81675 |
| no_dates | 0.925 | 1.000 | 1.000 | 0.000 | 0.000 | 77226 |
| key_exact | 0.925 | 1.000 | 1.000 | 0.000 | 0.000 | 42520 |
| translit_exact | 0.891 | 0.909 | 0.909 | 0.000 | 0.000 | 15477 |
| no_idf_check | 0.799 | 0.788 | 0.788 | 0.000 | 0.000 | 133246 |
| window_30d | 0.882 | 1.000 | 1.000 | 0.045 | 0.000 | 81488 |
| dense_verify | 0.896 | 0.985 | 0.985 | 0.000 | 0.000 | - |
| no_alignment_signals | 0.437 | 0.955 | 1.000 | 0.955 | 0.182 | 81488 |
| plus_anchor_signals | 0.793 | 1.000 | 1.000 | 0.091 | 0.000 | 81488 |
| no_retrieval_score | 0.925 | 1.000 | 1.000 | 0.000 | 0.000 | 81488 |

**Cost per suspect** (this machine, CPU; LLM time is the back-translation's own generation time):

| method | ms per suspect | detection only (ms) | LLM seconds | postings touched | docs compared |
| --- | --- | --- | --- | --- | --- |
| spintrace_xl | 129 | 19.4 | 0.0 | 81488 | - |
| latin_only | 1 | 0.6 | 0.0 | 2034 | - |
| dense_cosine | 27 | - | 0.0 | - | 8845 |
| bt_exact_hash | 20906 | 0.1 | 20.9 | - | - |
| bt_minhash_jaccard | 20907 | 0.3 | 20.9 | - | - |
| bt_tfidf_cosine | 20912 | 5.9 | 20.9 | - | - |
| bt_bm25 | 20912 | 5.3 | 20.9 | - | - |
| bt_spintrace_en | 21001 | 95.1 | 20.9 | - | - |
<!-- xl-results:end -->

Inspecting the cross-language steps (for the demo):

```
python -m spintrace kgram "अल्बानेज़"                 # transliteration, phonetic key, its 3-grams and their postings, matches
python -m spintrace suspect-xl <rid>                 # a rewrite from evaldata/rewrites_xl.jsonl, searched as in eval-xl
python -m spintrace suspect-xl <rid> --absent        # the same rewrite with its source masked out: should not flag
python -m spintrace suspect-xl <doc_id|url> --live   # a crawled Hindi page against the recent English crawl
python -m spintrace live-xl --same-site              # diagnostic: bilingual outlets matched against their own English pages
```

`suspect-xl` prints the canonical numbers, dates and Latin tokens; every name with its transliteration, phonetic
key, source-language df and k-gram matches (key, Jaccard, edit distance, English terms); the query terms with idf
and anchor type; the candidates with cosine and phrase hits; the verification signals and provenance; the sentence
alignment (Hindi sentence, English sentence, cosine); and the verdict.

**Live pages (unjudged).** `live-xl` runs every crawled Indian-language page (the restarted crawl adds Hindi,
Bengali, Telugu, Tamil and Gujarati feeds and tags each page's script) against the English crawl: pages from the
restarted crawl against an index of the English pages crawled in the same three days, older pages against the
main index's crawled documents. Output: `evaldata/live_xl.csv` (URLs, scores, anchor counts, lag and whether the
suspect names the outlet or a wire agency; no article text) and `results/crosslingual_live.json` (counts and five
seeded-random examples). Nobody judged these pairs, so they are counts, not precision. In the final run of 22 Nov
(1,387 Indian-language pages: 1,166 Devanagari, 86 Bengali, 70 Telugu, 25 Tamil, 25 Gurmukhi, 15 Gujarati; against
5,729 English pages from the same days and the main crawl) 7 pages were flagged, all Devanagari, a median 3.5 h
after the English page: two BBC Hindi articles traced to BBC News articles on another BBC domain (source credited;
a bilingual outlet translating its own reporting), two Amar Ujala pages traced to Times of India, two India TV pages
traced to The Independent and to The Hindu (one names PTI), and one Aaj Tak page traced to ThePrint 15.8 h earlier.
Two flagged pages credit their source and one names a wire agency.

Why so few: the Hindi outlets mostly cover Indian regional, political and entertainment news written in Hindi or
taken from the Hindi wires, whose English source is not in our crawl, so there is usually nothing to find; the
earlier-other-site filter excludes an outlet translating itself where both pages share a site (HT Bangla from
Hindustan Times), and most pairs that share anchors are independent coverage of one event, which the verifier is
built to reject. The 30-day window is not the limit (flagged lags were 0.4-16 h). The `--same-site` diagnostic lets
the 282 pages of the bilingual outlets match their own English pages and flags 2, both BBC Hindi articles that
credit their source. BBC's language services adapt rather than translate and HT Bangla and Telugu mostly run their
own stories, so this is a small, honest check of recall on real translations, not a measurement.

**Where SpinTrace-XL loses or is unproven.**
- Translate-then-detect is a strong baseline: with the LLM's back-translation, plain tf-idf or BM25 also rank the
  source first (P@1 1.000), though at their best dev thresholds their F1 is lower, most on summaries. It costs
  about 21 s of LLM time per suspect on this machine against well under a second for the anchor query and
  verification, and it needs a generative model in the loop for every page crawled. The cost of SpinTrace-XL
  excludes sentence embedding when those embeddings are cached; the cold cost is measured separately
  (`results/crosslingual_cold_cost.json`, once measured).
- The multilingual dense baseline finds the source almost as often (P@1 0.91) without any IR; its problem is
  deciding, not finding (it flags most facts-only articles). Dense retrieval followed by our verifier is close to
  the full system; the anchors win on cost (no corpus-wide embedding, which took 15 minutes for 20,860 document
  heads here and must be redone for every new page) and on being inspectable, not on recall at this scale.
- Sample sizes are small (22 test positives per Hindi level, 12 per other language in this run); a zero false-positive
  rate on 22 items still allows a rate up to 0.15, and the intervals in the tables say so.
- The length-matched negatives (about 100 words, same event) are flagged 0.091 of the time against 0.000 for the
  long ones, a small rise with overlapping intervals; word count alone separates the Hindi summaries from the long
  negatives perfectly (AUC 1.00), which is why the short negatives exist (`notes/critique_final.md`).
- Tamil, Telugu and Bengali did not generalise end to end: anchors retrieve the source (P@1 0.42, 0.58, 0.92
  against 0.08 for dense retrieval) but the sentence-embedding verifier does not transfer (F1 about 0 to 0.13), most
  likely because the multilingual sentence model was not trained on these languages (not checked against its
  model card).
- Name anchors are noisy: common Hindi words sound like rare English names. The source-language df check removes
  much of it, and the ablation shows the system depends on it (P@1 falls to 0.79 without it).
- All synthetic rewrites come from one 4B model, from Wikinews English; real farms may summarise more loosely.
- Shared wire copy is attributed to the earlier outlet, as in the English pipeline.

### Hindi mashups (prototype)

The two final features compose: a Hindi article stitched from two English originals. This is a prototype built from
parts that already existed, not a new verifier. `detect/crosslingual_multi.py` cuts the Hindi suspect into blocks of a
few sentences (about the size of the Hindi summaries the cross-language verifier was built for), traces each block to
an English source through the anchor index, and merges blocks that name the same source into per-source shares. The
explain-away rounds of the English path need a verifier fit on partial coverage of English sentences, which does not
exist for Hindi.

```
python -m spintrace.eval.xl_multi_eval --show 3   # evaluation, and the block-by-block trace of 3 seeded random mashups
```

Items are built by rule from the Hindi translations and SEO rewrites already in `evaldata/rewrites_xl.jsonl`: the
first k sentences of one derivative followed by the first 10 - k sentences of another derivative of a different
original in the same split (k = 5, 6, 7), so the sources and shares are exact. Originals are paired by their fixed
generation order, so growing the set adds pairs and never reshuffles old ones (the first version paired by sorted id,
which reshuffled every pair whenever a rewrite was added; the two pairings give F1 0.633 and 0.705 on the same
originals, which is a fair measure of how noisy a set this small is). Block size and probability threshold are tuned on
dev under a 10% false-source bar; the whole-text baseline can name at most one source. Numbers and controls:
`results/xlmulti_summary.md`, `xlmulti_detection.csv`, `xlmulti_controls.csv`.

What it shows, and where it fails: passage tracing recovers roughly half of the sources at precision 1.00, against
under a third for tracing the whole text as one article (a whole-text verifier does not pass an article that is only
half copied, as in the English case). It is not reliable: the exact source set is right for about a quarter of the
mashups; it names a false source for 28% of the long same-event articles and 12% of the 100-word ones (the whole-text
method: 0% and 8%), although its dev rate was under the bar, so a threshold tuned at the edge of the bar did not carry
over; and it finds the source of fewer Hindi summaries (0.68 against 0.88) because a three-sentence block of a summary
has too few anchors. It is reported as evidence that the two features compose, not as a result to build on.

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

<!-- multisource:start -->
## Multi-source mashups (final project)

**Problem.** AI content farms often stitch one article from two or three originals. The midterm pipeline keeps the
strongest verified source and stops, so the second source hides behind the first: its sentences are a minority of the
suspect, its rare terms are outweighed in the one query, and its whole-document verification score is low because it
explains only part of the suspect. **Claim:** explain-away retrieval recovers every source of a mashup and its share of
the text, while still giving exactly one source for a single-source rewrite and none for an independent article.

**Method** (`spintrace/detect/multisource.py`). Each round builds the candidate query only from the suspect's sentences
that no found source explains yet: their 30 highest tf-idf terms (names and numbers boosted, df >= 2), their quotes as
positional phrase queries, the earlier-only date filter, and the sources already found excluded. The top 5 candidates
are verified against those unexplained sentences, the best one is kept (the earliest of near-ties, as in the midterm),
and the suspect sentences it aligns to (MiniLM cosine >= the tuned cover value) are marked explained. Rounds stop when
the best candidate's probability is under the tuned threshold, when it explains fewer than 2 new sentences, when the
unexplained share falls under the tuned stop share, or after 4 sources. Each sentence is finally assigned to the found
source it aligns to best; the rest is the unexplained share (boilerplate, the farm's own filler).

**A per-round verifier.** The midterm verifier was fit on whole-document pairs, where a true source covers most of the
suspect; a 30% source of a mashup cannot pass it. The round verifier is a logistic regression over signals measured
against the *unexplained* sentences only: coverage, in-order chain (LIS) and longest contiguous in-order run of aligned
sentences, mean similarity, content-word overlap of aligned pairs, idf-free fact overlap, quote overlap, shingle
containment, back coverage, and the residual query's rare-term cosine. After the 70% source is explained, a 30% source
covers nearly all of what is left. Partial coverage alone cannot tell a mashup's source from the original of an article
written independently from the same facts, so two signals look at the rest of the suspect: the mean similarity of the
unexplained sentences the candidate does *not* align (`res_rest_sim`) and the share of them that are related but not
aligned (`res_related`). In an independent same-story article those sentences are still about the candidate's story
(median similarity 0.50 on dev); in a mashup they belong to another story (0.22-0.30). The verifier is fit on dev rounds
(a candidate is positive if it is a true source not yet found), with dev mashups and dev controls together. The
threshold, stop share and cover value are tuned on dev for micro source-set F1 over all dev items, under the
false-source bar described below. Because an early-stopping run is a prefix of a run with threshold 0, each dev pass is
run once and truncated for every grid point.

**Evaluation data** (`spintrace/eval/mashups.py`, `evaldata/mashups.jsonl`, deterministic, Wikinews text under CC BY):
each mashup splices one contiguous block of sentences from each of 2 or 3 originals in known proportions (50/50, 70/30,
40/30/30, 50/25/25), blocks in random order. Easy mashups combine random originals; hard ones combine originals
published within 7 days that share at least 3 moderately rare terms (in at most 200 of the 4,400 Wikinews articles),
drawn from the 400 originals and the Wikinews distractors. Wikinews rarely covers one event twice, so "hard" means
same-week, related-topic sources, not same-event sources. Levels: the splice as is; a light edit (the midterm's word
swaps and sentence drops, plus its boilerplate opening and closing line); a WordNet synonym spin; and `splice_seo`,
which splices blocks of the committed LLM SEO rewrites of the originals (a paraphrase the shingle checks miss; the
sources are the originals). Every source of a mashup is in the same split (by the first source's story). Controls come
from the index: single-source rewrites (light, synonym, SEO, summary; answer: exactly one source), facts-only articles
and crawl hard negatives (answer: no source). LLM mashups (gemma3:4b, two originals, `evaldata/rewrites_xl.jsonl`) are
read at run time; their source set is the label and their shares are unknown. Synthetic rewrites of a mashup's sources
are hidden from retrieval so a copy cannot stand in for the original.

**Baselines**, each with its best dev threshold for the same objective and bar: the midterm single-source pipeline (at most one
source; also shown at its shipped threshold), one-shot multi (every round-1 candidate the midterm verifier accepts, no
residual), MinHash LSH (every LSH candidate above a Jaccard estimate), and shingle containment against every earlier
document (every document holding more than a share of the suspect's 5-gram shingles).

**Results** (test split; every threshold tuned on dev). The operating point matters, so it is fixed in advance: every
method takes its best dev source-set F1 among the thresholds that name a source for at most 10.6% of the dev
facts-only articles and hard negatives, which is the shipped midterm verifier's own rate there. Without that bar,
source-set F1 alone picks thresholds that give a false source to about half of the facts-only articles (last table).
All tables are in `results/multisource_summary.md` and `results/multisource_*.csv`; charts:
`results/multisource_recall_by_n.png` (source recall by number of sources mixed, one line per method),
`multisource_recall_by_rank.png`, `multisource_share_scatter.png` and `multisource_tradeoff.png` (sources found
against false sources, sweeping each method's threshold).

**Source-set F1 on test mashups** (micro over true sources; 95% bootstrap interval over mashups):

| method | all splices | splice | splice_light | splice_synonym | splice_seo | hard | 3 sources |
| --- | --- | --- | --- | --- | --- | --- | --- |
| SpinTrace multi-source | 0.870 [0.84, 0.90] | 1.000 [1.00, 1.00] | 0.992 [0.98, 1.00] | 0.938 [0.90, 0.97] | 0.329 [0.25, 0.41] | 0.863 [0.81, 0.90] | 0.839 [0.78, 0.88] |
| Midterm single-source | 0.116 [0.08, 0.15] | 0.209 [0.15, 0.27] | 0.182 [0.12, 0.24] | 0.057 [0.02, 0.10] | 0.000 [0.00, 0.00] | 0.085 [0.04, 0.13] | 0.024 [0.01, 0.05] |
| One-shot multi | 0.116 [0.08, 0.15] | 0.209 [0.15, 0.27] | 0.182 [0.12, 0.24] | 0.057 [0.02, 0.10] | 0.000 [0.00, 0.00] | 0.085 [0.04, 0.13] | 0.024 [0.01, 0.05] |
| MinHash LSH | 0.188 [0.15, 0.22] | 0.384 [0.31, 0.45] | 0.292 [0.23, 0.35] | 0.008 [0.00, 0.03] | 0.000 [0.00, 0.00] | 0.184 [0.13, 0.23] | 0.100 [0.07, 0.13] |
| Shingle containment | 0.742 [0.71, 0.77] | 0.981 [0.97, 0.99] | 0.970 [0.96, 0.99] | 0.582 [0.52, 0.64] | 0.050 [0.02, 0.09] | 0.746 [0.69, 0.79] | 0.694 [0.64, 0.74] |
| Midterm, shipped threshold | 0.147 [0.11, 0.19] | 0.242 [0.18, 0.30] | 0.215 [0.15, 0.28] | 0.088 [0.05, 0.14] | 0.025 [0.00, 0.06] | 0.130 [0.08, 0.19] | 0.028 [0.01, 0.05] |

**LLM mashups** (gemma3:4b asked to combine two originals into one article; the source set is the label, shares unknown; never used for tuning; small n, so the intervals are wide):

| method | set | mashups | precision | recall | F1 [95% CI] | exact set | sources per item |
| --- | --- | --- | --- | --- | --- | --- | --- |
| SpinTrace multi-source | (all) | 48 | 1.000 | 0.281 | 0.439 [0.29, 0.57] | 0.188 | 0.56 |
| Midterm single-source | (all) | 48 | 0.000 | 0.000 | 0.000 [0.00, 0.00] | 0.000 | 0.00 |
| One-shot multi | (all) | 48 | 0.000 | 0.000 | 0.000 [0.00, 0.00] | 0.000 | 0.00 |
| MinHash LSH | (all) | 48 | 0.000 | 0.000 | 0.000 [0.00, 0.00] | 0.000 | 0.00 |
| Shingle containment | (all) | 48 | 1.000 | 0.021 | 0.041 [0.00, 0.10] | 0.000 | 0.04 |
| Midterm, shipped threshold | (all) | 48 | 0.000 | 0.000 | 0.000 [0.00, 0.00] | 0.000 | 0.00 |
| SpinTrace multi-source | (both sources test) | 28 | 1.000 | 0.339 | 0.507 [0.33, 0.67] | 0.214 | 0.68 |
| Midterm single-source | (both sources test) | 28 | 0.000 | 0.000 | 0.000 [0.00, 0.00] | 0.000 | 0.00 |
| One-shot multi | (both sources test) | 28 | 0.000 | 0.000 | 0.000 [0.00, 0.00] | 0.000 | 0.00 |
| MinHash LSH | (both sources test) | 28 | 0.000 | 0.000 | 0.000 [0.00, 0.00] | 0.000 | 0.00 |
| Shingle containment | (both sources test) | 28 | 1.000 | 0.018 | 0.035 [0.00, 0.10] | 0.000 | 0.04 |
| Midterm, shipped threshold | (both sources test) | 28 | 0.000 | 0.000 | 0.000 [0.00, 0.00] | 0.000 | 0.00 |
| SpinTrace multi-source | (all), max-F1 thresholds without the bar | 48 | 1.000 | 0.948 | 0.973 [0.94, 0.99] | 0.917 | 1.90 |
| Midterm single-source | (all), max-F1 thresholds without the bar | 48 | 0.900 | 0.375 | 0.529 [0.45, 0.59] | 0.000 | 0.83 |
| One-shot multi | (all), max-F1 thresholds without the bar | 48 | 0.989 | 0.927 | 0.957 [0.92, 0.98] | 0.833 | 1.88 |
| MinHash LSH | (all), max-F1 thresholds without the bar | 48 | 0.000 | 0.000 | 0.000 [0.00, 0.00] | 0.000 | 0.00 |
| Shingle containment | (all), max-F1 thresholds without the bar | 48 | 1.000 | 0.562 | 0.720 [0.64, 0.78] | 0.271 | 1.12 |

**Exact source set and recall by share rank** (test splices):

| method | exact set | sources per item | 2 src: largest | 2 src: second | 3 src: largest | 3 src: second | 3 src: third |
| --- | --- | --- | --- | --- | --- | --- | --- |
| SpinTrace multi-source | 0.741 [0.69, 0.79] | 1.93 | 0.875 [0.82, 0.92] | 0.812 [0.75, 0.87] | 0.742 [0.66, 0.81] | 0.721 [0.63, 0.79] | 0.705 [0.62, 0.77] |
| Midterm single-source | 0.000 [0.00, 0.00] | 0.15 | 0.234 [0.15, 0.31] | 0.036 [0.01, 0.08] | 0.037 [0.01, 0.07] | 0.000 [0.00, 0.00] | 0.000 [0.00, 0.00] |
| One-shot multi | 0.000 [0.00, 0.00] | 0.15 | 0.234 [0.15, 0.31] | 0.036 [0.01, 0.08] | 0.037 [0.01, 0.07] | 0.000 [0.00, 0.00] | 0.000 [0.00, 0.00] |
| MinHash LSH | 0.018 [0.00, 0.04] | 0.26 | 0.260 [0.19, 0.33] | 0.099 [0.06, 0.15] | 0.116 [0.07, 0.16] | 0.021 [0.01, 0.04] | 0.021 [0.01, 0.04] |
| Shingle containment | 0.484 [0.43, 0.53] | 1.48 | 0.729 [0.65, 0.80] | 0.630 [0.55, 0.70] | 0.653 [0.57, 0.72] | 0.484 [0.42, 0.54] | 0.458 [0.40, 0.51] |
| Midterm, shipped threshold | 0.000 [0.00, 0.00] | 0.20 | 0.312 [0.23, 0.40] | 0.042 [0.01, 0.08] | 0.042 [0.01, 0.08] | 0.000 [0.00, 0.00] | 0.000 [0.00, 0.00] |

**Controls** (test): extra-source rate on single-source rewrites (any source besides the true one), false-source rate on facts-only articles and crawl hard negatives (any source at all):

| method | extra: light | extra: synonym | extra: seo | extra: summary | false: facts-only | false: hard neg | source found: seo | source found: summary |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SpinTrace multi-source | 0.000 [0.00, 0.00] | 0.000 [0.00, 0.00] | 0.000 [0.00, 0.00] | 0.000 [0.00, 0.00] | 0.078 [0.02, 0.14] | 0.033 [0.00, 0.08] | 0.644 | 0.800 |
| Midterm single-source | 0.000 [0.00, 0.00] | 0.000 [0.00, 0.00] | 0.000 [0.00, 0.00] | 0.000 [0.00, 0.00] | 0.111 [0.06, 0.18] | 0.017 [0.00, 0.05] | 0.700 | 0.956 |
| One-shot multi | 0.000 [0.00, 0.00] | 0.000 [0.00, 0.00] | 0.000 [0.00, 0.00] | 0.000 [0.00, 0.00] | 0.111 [0.06, 0.18] | 0.017 [0.00, 0.05] | 0.700 | 0.956 |
| MinHash LSH | 0.000 [0.00, 0.00] | 0.000 [0.00, 0.00] | 0.000 [0.00, 0.00] | 0.000 [0.00, 0.00] | 0.033 [0.00, 0.08] | 0.000 [0.00, 0.00] | 0.011 | 0.000 |
| Shingle containment | 0.000 [0.00, 0.00] | 0.000 [0.00, 0.00] | 0.000 [0.00, 0.00] | 0.000 [0.00, 0.00] | 0.178 [0.10, 0.27] | 0.000 [0.00, 0.00] | 0.189 | 0.556 |
| Midterm, shipped threshold | 0.000 [0.00, 0.00] | 0.000 [0.00, 0.00] | 0.000 [0.00, 0.00] | 0.000 [0.00, 0.00] | 0.156 [0.08, 0.23] | 0.017 [0.00, 0.05] | 0.756 | 0.967 |

**Ablations** (each variant's threshold and stopping share re-tuned on dev under the same false-source bar; test numbers):

| variant | F1 splices | F1 splice_seo | F1 all items | 2nd of 2 | 3rd of 3 | extra: seo | extra: summary | false: facts-only | false: hard neg |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| full | 0.870 | 0.329 | 0.871 | 0.812 | 0.705 | 0.000 | 0.000 | 0.078 | 0.033 |
| no_residual | 0.844 | 0.323 | 0.850 | 0.771 | 0.637 | 0.000 | 0.000 | 0.111 | 0.033 |
| no_facts_quotes | 0.867 | 0.304 | 0.869 | 0.807 | 0.700 | 0.000 | 0.000 | 0.078 | 0.033 |
| midterm_verifier_in_rounds | 0.118 | 0.000 | 0.383 | 0.036 | 0.000 | 0.000 | 0.000 | 0.111 | 0.017 |

**Without the false-source bar** (each method at its max dev F1 over all items; test numbers). This is the operating point a source-set F1 alone would pick; it buys recall with false sources on independent articles:

| method | threshold | F1 splices | F1 splice_seo | F1 all items | 2nd of 2 | 3rd of 3 | extra: seo | extra: summary | false: facts-only | false: hard neg |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SpinTrace multi-source | 0.500 | 0.972 | 0.894 | 0.954 | 0.927 | 0.942 | 0.011 | 0.000 | 0.522 | 0.117 |
| Midterm single-source | 0.046 | 0.513 | 0.362 | 0.621 | 0.219 | 0.063 | 0.000 | 0.000 | 0.922 | 0.133 |
| One-shot multi | 0.009 | 0.856 | 0.670 | 0.817 | 0.776 | 0.547 | 0.067 | 0.156 | 0.978 | 0.400 |
| MinHash LSH | 0.102 | 0.188 | 0.000 | 0.216 | 0.099 | 0.021 | 0.000 | 0.000 | 0.033 | 0.000 |
| Shingle containment | 0.044 | 0.917 | 0.599 | 0.893 | 0.865 | 0.837 | 0.011 | 0.033 | 0.467 | 0.233 |
| Midterm, shipped threshold | 0.804 | 0.147 | 0.025 | 0.404 | 0.042 | 0.000 | 0.000 | 0.000 | 0.156 | 0.017 |

What the numbers say:

- **The second source no longer hides.** SpinTrace finds the second of two sources in 81% of test splices and the
  third of three in 71%; the midterm finds 4% and 0%. Its exact source set is right for 74% of splices.
- **The midterm single-source pipeline and one-shot multi are structurally capped, and come out identical.** Both
  score the round-1 candidates with the midterm's whole-document verifier and, under the bar, land on the same
  threshold (0.854). A whole-document verifier passes a candidate only if it explains most of the suspect; in a 2- or
  3-source mashup at most one source can, and usually none does (the midterm finds even the larger of two sources in
  only 23% of 2-source splices). So no second candidate ever clears the threshold and one-shot never adds a source.
  Without the bar one-shot reaches splice F1 0.856, but gives a source to 98% of facts-only articles.
- **The main gain is the per-round verification criterion, not the residual query.** Putting the midterm's
  whole-document verifier inside the same rounds gives splice F1 0.118. Querying only the unexplained sentences adds
  2.6 points of splice F1 over re-querying the whole suspect each round (0.844 to 0.870), raises recall of the second
  of two sources from 0.771 to 0.812 and of the third of three from 0.637 to 0.705, and lowers the facts-only
  false-source rate from 0.111 to 0.078. It adds little because one rare-term query for the whole suspect already
  retrieves the second source in its top 5 for 91% of 2-source splices (85% for the third of three); the residual
  query helps for the rest. Dropping names, numbers and quotes from the residual query costs 0.3 points.
- **Against shingles, the gain is on paraphrase only.** Shingle containment against every earlier document scores
  0.981 on verbatim splices and 0.970 on light edits, as well as SpinTrace and about 60 times cheaper (1.5 ms against
  99 ms per suspect), and 0.742 over all splices. SpinTrace's lead is on synonym-spun splices (0.938 against 0.582)
  and SEO-rewrite splices (0.329 against 0.050). On copy-level mashups the textbook check is enough.
- **LLM mashups** (48; the 28 with both sources in the test split are the test subset, and none were used for tuning):
  under the bar SpinTrace finds 28% of their sources with no wrong source (F1 0.439 on all 48, 0.507 on the test
  subset), shingle containment 2% (F1 0.041) and the midterm, one-shot and MinHash none. Without the bar every
  method does far better (SpinTrace 0.973, one-shot 0.957, containment 0.720, midterm 0.529): gemma3:4b keeps much
  of each original's sentences when it combines two, and here too one query for the whole suspect already holds
  both sources. The bar, set by the facts-only articles, is what costs recall on these items.
- **Shares** are near exact for copy-level splices (mean absolute error 0.000-0.046 of the suspect) and poor for
  SEO-rewrite splices (0.314, counting missed sources as 0): most paraphrased sentences fall under the 0.62 alignment
  cut and stay unexplained.
- **Cost:** about 99 ms per suspect for retrieval and verification over 1.66 rounds (about 10,000 postings touched
  per round, 5 candidates verified), plus about 0.1 s to embed a new suspect's sentences on this CPU.

**Live pass** (`eval-multi --live`, unjudged; `evaldata/live_multisource.csv` holds URLs, shares and probabilities
only). Of the 174 crawled articles the single-source scan flagged, 153 keep one verified source, 13 (7.5%) get two or
more and 8 get none at the stricter round threshold; the first source equals the midterm's source for 88%. None of a
seeded sample of 300 unflagged crawled articles gets any source. In 6 of the 13 multi-source articles the two
sources share at least half their shingles, so they are one wire text printed by two outlets (the Sitharaman GST
op-ed in both the Hindustan Times and the Economic Times), not a mashup; the CSV's `max_source_overlap` column marks
them. The seeded examples the command prints include a NewsX Djokovic report traced 88% to Devdiscourse and 12% to
the Times of India, both agency-based. Nobody has judged these pairs, so they are counts and seeded examples, not
precision.

**Where multi-source tracing loses.**

- LLM-paraphrased blocks: splice_seo F1 is 0.329 under the bar (0.894 without it, at a 52% facts-only false-source
  rate). At the residual level an LLM paraphrase of part of an original and an article written independently from
  the original's fact sheet look alike; the two rest-of-suspect signals separate them only partly.
- Single-source paraphrases: the strict round threshold (0.95) finds the source of 64% of SEO rewrites and 80% of
  summaries, against 76% and 97% for the shipped midterm. In a crawler the midterm pass should stay first and the
  multi-source pass add further sources.
- False sources: 7.8% of facts-only articles and 3.3% of crawl hard negatives get a source (shipped midterm 15.6% and
  1.7%).
- Hard mashups are same-week, related-topic sources (Wikinews rarely covers one event twice), not same-event ones;
  only the LLM mashups and the live pass test mixing within one event.
- Two found sources can be copies of one wire story; they are reported as two sources (marked by overlap).
- Undated documents pass the earlier-only filter and the rounds make no further provenance check; mashup dates are
  synthetic.
- The false-source problem was first seen in the test numbers of the first run (59% of facts-only articles got a
  source at the max-F1 threshold). The diagnosis, the two new signals and the bar used dev only, and the bar is the
  shipped midterm's dev rate rather than a tuned value, but the decision to fix it was informed by a test number
  (`notes/critique_multisource.md`).

**Format changes (stretch): the order-free fact signals do not help.** A copy that turns its source into a listicle or
a Q&A keeps the facts but not the prose order the midterm verifier leans on. `spintrace/eval/formats.py` makes two
rule-based format rewrites of each of the 400 originals (`evaldata/formats.jsonl`, deterministic): a listicle (the
fact-bearing sentences as a shuffled numbered list under a "what we know" line, lightly synonym-spun) and a Q&A (a
templated question before each fact-bearing sentence, shuffled, answers lightly spun). Three optional order-free
signals were added to `detect/signals.py` (`OPTIONAL`, not used by the default verifier): the idf-weighted share of the
suspect's names, numbers and quotes found in the candidate, the same share in the other direction, and the share of the
suspect's fact-bearing sentences whose facts are mostly in the candidate. `python -m spintrace format-eval` scores
format rewrites, SEO rewrites and summaries against facts-only articles and crawl hard negatives (test split, each
refit and threshold on dev; `results/format_detection.csv`, `format_f1.png`):

| verifier | listicle F1 | Q&A F1 | SEO F1 | summary F1 | FPR facts-only | FPR hard neg |
| --- | --- | --- | --- | --- | --- | --- |
| midterm, as shipped | 0.964 | 0.943 | 0.791 | 0.911 | 0.156 | 0.000 |
| refit on dev, same signals | 0.941 | 0.941 | 0.819 | 0.871 | 0.256 | 0.006 |
| refit on dev, plus the fact signals | 0.931 | 0.931 | 0.826 | 0.854 | 0.300 | 0.006 |
| fact signals and rare-term cosine only | 0.886 | 0.886 | 0.759 | 0.719 | 0.511 | 0.018 |

The shipped verifier already handles these rule-based formats: shuffling keeps most sentences aligned, so coverage
stays high. Adding the fact signals lowers format F1 and raises the facts-only false-positive rate, because an article
written from the original's fact sheet shares exactly those facts. The NewsGuard TopGolf.kr page (a reshaped deals
list) still has its Wired source ranked first and verified by none of the four (p at most 0.001;
`results/format_newsguard.csv`). The default verifier is unchanged.

<!-- multisource:end -->

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
- **Two crawlers at once broke the delay.** The crawl figures above cover the main crawl. Short rehearsal crawls for the
  demo video on 7 Oct (23:13-23:16) ran two crawler processes at once; each kept its own 5 s gap, but together they
  contacted 19 hosts 0.6-5.0 s apart, so `crawl-report` on that database counts 39 delay violations (9 from the first
  night's race, 30 from this). A crawler now holds an exclusive lock and a second one refuses to start.
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
- Multi-source tracing (final phase): explain-away rounds with a per-round verifier, sentence shares, `sources` inspect command (mashup id, indexed article or `--file`), optional multi-source edges in `scan --multi` (off by default), rule-based and LLM mashup evaluation against four baselines with ablations, and an unjudged live pass.
- Cross-language tracing of Hindi (and, as a check, Tamil, Telugu and Bengali) rewrites to English sources through
  number, date and name anchors (`build-xl`, `kgram`, `suspect-xl`, `eval-xl`, `live-xl`), with translate-then-detect,
  multilingual dense and Latin-token baselines, ablations, and an unjudged live run over the crawled Indian-language pages.

Partial:

- Real farm pairs: two NewsGuard-documented pairs load from the Wayback Machine. NYT -> GlobalVillageSpace is retrieved first among 20.9k documents, verified (p = 0.999) and traced; Wired -> TopGolf.kr is retrieved first but not verified. The Bored Panda and People originals are not named in the report.
- Hard negatives and flagged live pairs are mined automatically. Both team members have judged the flagged live pairs (`evaldata/live_flagged.csv`) and the search results (`search_judgments.csv`); the hard-negative sheet (`hard_negatives.csv`) is not judged yet.
- The final's live results (cross-language flags and multi-source counts) are unjudged: nobody had time to label
  them, so they are counts with examples, not precision. Tamil, Telugu and Bengali are retrieved by anchors but not
  verified (the sentence model does not cover them); the format-change stretch did not help (reported as a negative
  result).
- No live farm sites were found among the seeds yet; suspects come from aggregators that republish agency copy (latestly, devdiscourse, socialnews, newsx).

Planned (not done in the final):

1. A mashup path for Hindi suspects (a Hindi article stitched from two English originals), which would join the two
   new features; verification for Tamil, Telugu and Bengali with a sentence model that covers them.
2. Judged live pairs for both new features, and rewrites from stronger LLMs (the Hindi and mashup tests use one 4B
   model) or a real translation benchmark.
3. Adaptive recrawl scheduling from estimated change rates; full Mercator queues with several fetcher threads.
4. Farm-aware focused crawling: raise the frontier priority of domains with high derivative rates (the `priority()`
   hook), and a crawl of the Hindi wires so live Hindi pages have their real sources in the index.
5. Learned weights for more signals, trained on judged live pairs instead of synthetic rewrites.
6. A measurement study of how fast farms copy; a much larger index and external benchmarks (PAN, METER), which is
   also where the cost advantage of the anchors over embedding the corpus should show.

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
| 21 Nov 15:05-15:45 | Final phase starts: network check (BBC, Wikinews, Wayback), Indian-language feeds added, crawler restarted; rewrite queue for Hindi, Tamil, Telugu, Bengali and mashups started |
| 15:45 | Two helper agents in separate worktrees: cross-language and multi-source |
| 15:50-16:15 | Final phase, multi-source: explain-away rounds, rule-based mashups, first smoke run |
| 16:15-16:40 | Multi-source evaluation with baselines; first run names a source for 59% of facts-only articles |
| 16:40-17:05 | Dev diagnosis, rest-of-suspect signals and the false-source bar; full rerun (17 min), live pass |
| 22 Nov 02:00-02:15 | LLM mashups added to the evaluation; critique notes |
| 16:00 | The Mac restarted; the crawl, the queue and both agents resumed from stored state, nothing lost |
| 17:47 | The crawler died on an unhandled connection reset while reading a response body |
| 22 Nov 02:00 | Crawler fixed (regression test) and restarted for 7 h; queue and agents resumed after a usage limit |
| 02:30 | Both branches merged; the length check finds Hindi summaries separable by word count (AUC 1.00), so length-matched negatives are added |
| 02:46-03:02 | Final evaluation run: mashups, Hindi levels, live passes, dataset summary, report tables and figures (16 min) |

## AI-use log

The team used Claude Code (Anthropic, models Claude Opus 5.5 and, for the final phase, Sonnet 5.5) as a coding agent throughout the build, including two helper agents in separate git worktrees for the cross-language and multi-source paths.

- Generated with Claude Code from the team's plan (PLAN.md, CLAUDE.md): the package code under `spintrace/`, the tests, the CLI, the evaluation harness and charts, this README, PROGRESS.md and the critique notes in `notes/`.
- The team wrote the plan, the brief and the seed strategy, chose the track and claim, reviewed the code and results, and judges the live pairs, hard negatives and search results.
- Synthetic SEO rewrites, summaries, round-trip translations and facts-only articles were generated by Meta Llama 3.2 3B run locally through Ollama (`spintrace/eval/rewrites.py`); they are evaluation data only.
- all-MiniLM-L6-v2 (a pretrained sentence-embedding model) is used for verification signals and the dense baseline.
- Cross-language extension (final project): Claude Code generated `text_indic.py`, `translit.py`, `kgram.py`,
  `embed_xl.py`, `detect/crosslingual.py`, `eval/xl_eval.py`, `eval/xl_live.py`, the `build-xl`, `kgram`,
  `suspect-xl`, `eval-xl` and `live-xl` commands, `tests/test_crosslingual.py`, `notes/critique_xl.md` and the
  cross-language parts of this README. The Hindi, Tamil, Telugu and Bengali rewrites and the back-translations
  were generated by Google Gemma 3 4B through Ollama (evaluation data only); paraphrase-multilingual-MiniLM-L12-v2
  (pretrained) aligns sentences across languages. No AI system labelled the live cross-language pairs; they are
  reported unjudged.
- No AI system labelled evaluation data: synthetic labels come from construction, hard negatives from a rule-based miner, and live pairs are for human judges.
- Final phase (21 Nov), generated with Claude Code: multi-source tracing (`spintrace/detect/multisource.py`, the `sources` command, the optional multi-source edges in `scan`), the read-only analysis context and embedding overlay, the rule-based mashup and format-rewrite generators, the multi-source and format-change evaluations, their tests, charts and README sections. The LLM mashups in `evaldata/rewrites_xl.jsonl` are gemma3:4b outputs, evaluation data only; mashup labels come from construction.
- Final phase, report and critique: Claude Code drafted `report/SpinTrace_final_report.tex`, the table and figure generators (`report/make_tables.py`, `report/make_figures.py`), `scripts/final_run.sh`, `spintrace/eval/dataset_stats.py`, the crawler fix for connection resets and `notes/critique_final.md`. The team chose the two problems, the claims and the experiments and reviews the results. The final's live results are unjudged.
