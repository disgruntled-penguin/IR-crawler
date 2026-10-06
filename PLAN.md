# SpinTrace: T4 Project Plan

> **For Claude Code:** this is the team's plan, kept for context. CLAUDE.md is the binding brief: where the two differ, CLAUDE.md wins. Treat the architecture below as a starting point, not a spec; change it if you find something better and say why in the README. Human tasks (judging, video, report) are not yours.

Oct 6, 2026

## Overview and scale decision

Build a deep vertical slice: every stage working end to end at modest scale, the novel detection stage built thoroughly, on an architecture that grows into the course project without a rewrite.

**SpinTrace in one line:** a polite, focused crawler that catches news articles rewritten from other outlets, including AI paraphrases that standard duplicate checks miss, and traces each one back to its original.

| Option | What it means | Effect on marks | Verdict |
| --- | --- | --- | --- |
| Thin prototype | Detection on a static dataset, with no crawler or a toy crawl of a few hundred pages | Weak track relevance; corpus too small for ranking to matter | No |
| Vertical slice | Polite crawl of 15 to 25 sites (several thousand pages), full detection pipeline, ranked search, rigorous evaluation | Maximises IR principles (30), working system (20) and evaluation (15) | Build this |
| Full scale | Distributed crawling, freshness scheduling, dashboard, very large corpus | No marks for scale; burns hours evaluation needs; more demo risk | Course project |

The hackathon build is Milestone 1 and must stand alone as a complete submission. This tab is for the two of you; what Claude Code builds, in what order and how it is evaluated lives in CLAUDE.md, which leaves implementation choices open.

## Novelty claim and positioning

**Claim:** SpinTrace extends the crawler's "content seen?" check from copy detection to paraphrase-aware derivative detection: it retrieves a rewrite's likely source using the facts a rewrite rarely changes, then traces provenance and demotes copies in ranking.

**The insight that makes it IR, not just NLP.** A rewrite changes wording but keeps facts: names, numbers, dates and direct quotes, which are often copied verbatim. Those are high-idf terms, so a suspect article's rare terms become a query against the inverted index (index elimination), and its quotes become phrase queries on a positional index. IR finds the candidates; semantic similarity only verifies them.

**Why it matters.** In NewsGuard's test, Grammarly's plagiarism checker could not find the source of 34 of 43 AI-rewritten articles, a 79% failure rate ([NewsGuard, Aug 2023](https://www.newsguardtech.com/misinformation-monitor/august-2023)). NewsGuard now tracks 3,749 AI content farm sites, and its full list is not public ([AI Tracking Center](https://www.newsguardtech.com/special-reports/ai-tracking-center)).

| Existing approach | What it catches | Why it fails on AI rewrites |
| --- | --- | --- |
| Exact hashing (textbook "content seen?") | Identical copies | Any edit breaks the hash |
| Shingles with MinHash or SimHash | Near-copies and light edits | Paraphrases share few shingles |
| Plagiarism checkers | Verbatim reuse | Missed 79% of sources in NewsGuard's test |
| Domain blocklists (browser extensions, GitHub lists) | Known farm domains | Manual and stale; new domains slip through |
| AI-text detectors | Text that looks machine-written | Unreliable, and never name the source |
| **SpinTrace** | Rewrites of any intensity, plus their original | Weak spots measured in the evaluation |

A GitHub API search (Oct 2026) found blocklists and generic plagiarism checkers, but no project combining a crawler, paraphrase-aware detection and provenance tracing.

## System architecture

Four stages run in order; only stage 3 is new, and it reuses stage 2's index and idf weights to find candidates before any embedding model runs.

1. **Crawl:** frontier with priority front queues and one back queue per host; fetch obeying robots.txt and crawl-delay, with per-host delays and backoff on 429; normalise URLs and extract main text, title and dates.
2. **Store and index:** the textbook content-seen check (exact hash, then shingles with MinHash and LSH); inverted index with positional postings and zones (title, body, quotes); tf-idf and BM25 weights.
3. **Detect paraphrased copies (the novelty):** candidate retrieval with the suspect's high-idf terms as a query plus quote phrase queries; verification by embedding-based sentence alignment (coverage and order kept); provenance from earliest publish date, containment and a copy graph.
4. **Use the results:** ranked search with relevance plus g(d); a per-domain integrity report; evaluation against baselines.

Exact hashing and MinHash stay in the pipeline as the textbook "content seen?" check, so they double as the baselines stage 3 must beat.

**Derivative score:** a weighted mix of alignment coverage, order preservation, and quote and entity overlap, with weights and threshold tuned on the dev split.

**Ranking:** each document's static quality g(d) is its originality, near 1 for originals and near 0 for confirmed copies.

```latex
\text{net score}(q,d) = \text{relevance}(q,d) + \lambda \, g(d)
```

## Data sources and ethics

Three real sources plus one synthetic set: a live crawl of originals and suspected farms, documented NewsGuard pairs, and graded rewrites generated from freely licensed news.

| Source | Role | Notes |
| --- | --- | --- |
| Live crawl: original outlets (e.g. The Guardian, BBC, NPR, The Conversation, Wired, The Verge, Indian outlets) | Originals for the index and the demo | Check each robots.txt at hour 0; discover articles via RSS and news sitemaps |
| Live crawl: suspected farm sites | Where real rewrites appear | Domains named in [NewsGuard's report](https://www.newsguardtech.com/misinformation-monitor/august-2023) if still live, plus new ones from manual farm hunting |
| NewsGuard-documented pairs | Real, third-party-verified ground truth for the reveal | Farm copies via Wayback snapshots; prefer pairs with open originals (Wired, The Verge, The Guardian, Bored Panda) |
| Wikinews dump | Originals for synthetic rewrites | Freely licensed (CC BY), so rewrites can be shared in the repo |
| LLM-generated rewrites | Labelled evaluation set at graded difficulty | Declared in the AI-use section |
| CC-NEWS, PAN, METER (optional) | Scale and external benchmarks | Check access and formats before relying on them |

**Farm hunting** is manual research, not automated search scraping: a teammate searches for chatbot error strings such as "As an AI language model" on news-style sites, verifies each hit, and adds the domain to the seed list.

Ethics rules (robots.txt, delays, no personal data, no crawled texts in the repo) are enforced in code; they are listed in CLAUDE.md.

## Timeline

Hours count from the track announcement (hour 0 = 12:00); planning used hours 0 to 9, so the build runs 27 hours from 21:00. The crawler is built first so it collects pages in the background for about 18 hours.

- 21:00 to 01:00: setup, crawler built and first crawl checked by a human; teammate builds the seed list and verifies NewsGuard pairs
- 01:00 to 11:00: Claude Code builds unattended under /goal while the live crawl runs in the background
- 11:00 to 16:00: human review and fixes, hard negatives, judged queries, judging flagged pairs
- 16:00: feature freeze
- 16:00 to 00:00: evaluation runs and charts, report, video, clean-clone test, submission

**Gates:**

1. **Hour 10 (22:00), real pairs verified:** at least three NewsGuard pairs load from the Wayback Machine and every seed's robots.txt is checked. If not, swap in other documented pairs or seeds before building further.
2. **Hour 18 (06:00), 2,000 pages crawled:** if the live crawl is short, ingest one CC-NEWS file so ranking still matters.
3. **Hour 28 (16:00 tomorrow), feature freeze:** after this, only bug fixes, evaluation runs, writing and recording.

The live crawl runs unattended during the first sleep shift, so it logs errors to a file and stops itself if a host returns repeated 429s.

## Fallbacks

| Risk | Fallback |
| --- | --- |
| Same-event coverage flagged as copying | Require alignment and order, not just shared names; report the false-positive rate honestly |
| Crawler slow or blocked | RSS and sitemaps first; drop blocked seeds; under 2,000 pages by 06:00, ingest one CC-NEWS file |
| No live farm rewrites found | NewsGuard pairs and the synthetic set carry evaluation and the demo |
| Rewrite generation slow or rate-limited | Cache every output; drop the translation level |
| Unreliable publish dates | Prefer original-outlet metadata, then crawl time, then Wayback first capture; otherwise mark provenance "uncertain" |
| Wayback snapshots fail to load | Use other documented pairs |

## Next steps (for the README's "planned" section)

1. Adaptive recrawl scheduling from estimated change rates, full Mercator queues, multi-process crawling.
2. Farm-aware focused crawling that prioritises domains with high derivative rates.
3. Cross-lingual derivatives, such as English originals rewritten into Hindi.
4. Learned weights for the detection signals.
5. A measurement study of how fast farms copy and how long copies outrank originals.
