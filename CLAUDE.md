# SpinTrace: brief for Claude Code

Read this with the assignment PDF (CSD358_IR_Mid-term_Assignment_2026.pdf in the repo). It fixes what the project must achieve and how it is judged; every implementation choice not stated here is yours to make and justify in the README. PLAN.md holds the team's wider plan for context; where it differs from this file, this file wins.

## Goal and claim

Build SpinTrace for the CSD358 IR hackathon, Track 4 (web crawling, freshness and web integrity): a polite, focused news crawler that catches articles rewritten from other outlets, including AI paraphrases that standard duplicate checks miss, and traces each one to its original. Read the assignment PDF in full before writing code.

**Novelty claim (fixed):** SpinTrace extends the crawler's "content seen?" check from copy detection to paraphrase-aware derivative detection: it retrieves a rewrite's likely source using the facts a rewrite rarely changes, then traces provenance and demotes copies in ranking.

**Core insight to preserve:** rewrites change wording but keep names, numbers, dates and direct quotes. Those are high-idf terms and exact phrases, so classical IR (inverted index, idf, positional phrase queries) finds candidate sources and semantic similarity only verifies them. A design that embeds everything and ranks by cosine has lost the point.

**Motivation:** NewsGuard found Grammarly's plagiarism checker missed the sources of 34 of 43 AI-rewritten articles (79%) ([report](https://www.newsguardtech.com/misinformation-monitor/august-2023)), and tracks 3,749 AI content farm sites as of June 2026 ([tracker](https://www.newsguardtech.com/special-reports/ai-tracking-center)).

**Target scale:** a vertical slice. Every stage works end to end on real data (several thousand crawled pages), and the detection stage is built deeply; breadth beyond that earns no marks.

## Non-negotiables

These protect marks or come straight from the assignment; everything else is open.

1. **Real inputs.** The system runs live on real crawled pages; nothing faked or hard-coded. Synthetic rewrites are for evaluation labels only.
2. **Own IR core.** Write the inverted index (positional, with zones), tf-idf and BM25 weighting, shingling, MinHash, LSH, and the frontier with politeness yourself. Libraries are fine for HTTP, HTML extraction, NER, embeddings and plotting; explain each in IR terms in the README.
3. **Inspectable.** A command (or notebook) prints intermediate output for the demo video: postings for a term, shingle overlap and Jaccard for a pair, candidate scores, the sentence alignment and the provenance verdict.
4. **Ethics in code.** Obey robots.txt including Crawl-delay; otherwise wait a few seconds between requests to one host. Send an identifying user agent with a contact email, back off on 429 and 503, and stop a host after repeated errors. Skip comments and profiles. Never commit crawled full texts: commit code, URLs, features and a re-crawl script.
5. **Reproducible.** One README covering setup, exact run commands, data sources with credits, and what works versus what is planned. It must work from a clean clone.
6. **Code style.** Comments short and only where the logic is not obvious; no decorative dashes or hashes.
7. **AI-use log.** Keep a running list of what you generated, for the report's declaration.

**Inputs the team provides:** a seed list of original outlets and suspected farm sites, a list of NewsGuard-documented pairs with Wayback links, and judged labels for flagged live pairs. Use Wikinews (freely licensed) as originals for synthetic rewrites.

**Commits:** the first commit is this plan exported as markdown. Commit small and often with real timestamps, and add a short timeline to the README: tracks announced 12:00, planning 12:00 to 21:00, build from 21:00.

## IR concept map

The assignment sets no minimum number of concepts, so use only what the design needs: if removing a concept would not weaken the system, leave it out. These are needed:

| Concept | Why SpinTrace needs it |
| --- | --- |
| Crawl loop, frontier, politeness, robots.txt, URL normalisation | The track itself: collecting news and farm pages responsibly |
| Content-seen check: exact hash, shingles, Jaccard, MinHash | The textbook duplicate check SpinTrace extends, and the main baseline |
| Tokenisation and normalisation | One shared pipeline for the index, shingles and queries |
| Inverted index with positional postings, phrase queries | Quotes survive rewriting, so phrase matching is a key signal |
| idf and index elimination | Rare terms survive rewriting, so a suspect's query is built from them |
| tf-idf, cosine, length normalisation | Candidate scoring; normalisation matters because summaries are short |
| Parametric filter on publish date | Provenance only compares a suspect with earlier articles |
| Static quality g(d) and net score | Originals outrank their copies in search |
| Precision, recall, P@k | Every evaluation |

**Optional, only because each answers a real question:**

- Stemming, compared with none: does it lift candidate recall when rewrites change word forms?
- BM25 and a dense-retrieval baseline: when does sparse retrieval win? Both also count as out-of-syllabus extras.

Do not add concepts for coverage; the README explains why each one used is there.

## Priorities

Get a rough version of every stage working end to end before any stage gets deep; stop adding features with about 8 hours left.

1. **End to end first:** polite crawl into storage, the index, the textbook content-seen check as baseline, candidate retrieval from rare terms, verification, provenance, search with originality, the inspect command, and the synthetic evaluation against baselines.
2. **Then depth:** quote and entity signals, the earlier-articles-only filter, hard negatives, real pairs, ablations, the stemming comparison, a dense-retrieval baseline, judged search queries, and a per-domain integrity report.
3. **Only if ahead:** one CC-NEWS file for scale, a round-trip translation rewrite level, an external benchmark (PAN or METER).

**Design principles, so the course project can extend rather than rebuild:** stages exchange stored data keyed by stable document IDs; each detection signal is an independent scorer; the frontier's priority lives in one replaceable function.

## Evaluation requirements

The headline result is detection F1 at each rewrite intensity, one line per method, showing where shingle Jaccard collapses. You choose set sizes, as long as metrics are stable.

**Data:**

- Synthetic rewrites of freely licensed originals at graded intensity: exact copy, light edit, synonym substitution, an LLM "SEO-friendly" rewrite like the one NewsGuard tested, and an LLM summary (round-trip translation optional). Mix them into the full crawled index as distractors.
- Hard negatives: independent articles about the same event from different outlets. Same story is not the same as copied, and this is the test that matters most.
- Real cases: NewsGuard-documented pairs, and flagged live pairs judged by both team members.

**Baselines:** exact hash, shingle Jaccard with MinHash, document-level tf-idf cosine, and BM25, each given its best threshold on the dev split.

| Question | Report |
| --- | --- |
| Is this article derived from another? | Precision, recall and F1 per level; PR curve |
| Is its original ranked first? | P@1, MRR, Recall@k; candidates scored and time versus all-pairs comparison |
| Does it work on real farms? | Rank of the original for each NewsGuard pair; precision of flagged live pairs, with judge agreement |
| Do originals outrank copies in search? | P@10 and nDCG@10 on judged queries with and without g(d); share of originals in the top 10 |
| Is the crawler correct and polite? | Pages fetched, rate, disallowed URLs fetched (must be 0), smallest gap per host, duplicate rate, trap guards fired |

**Rules:** remove one signal at a time and report the change; tune only on a dev split and report the test split; state where SpinTrace loses; never hand-pick demo examples to hide failures. Write every chart and table to files for the report and video.

## Working loop

Build, then attack your own work as a hostile reviewer, argue it out, and fix; run two or three rounds, starting only once the end-to-end slice works.

1. **Build** the slice with a smoke run on real data.
2. **Critique** as a reviewer who wants to fail the project, in scratch notes under `notes/` (not a deliverable). Ask at least:
   - Is IR doing real work, or decorating an embedding model? Remove the IR stage: does recall collapse?
   - Were the baselines given their best thresholds, or set up to lose?
   - Are the hard negatives genuinely hard: same event, same day, same names?
   - Was anything tuned on the test split, or leaked between originals and rewrites?
   - What happens to provenance when publish dates are missing or wrong?
   - Would a grader call this a standard near-duplicate demo? Is the claim still true?
   - Do the logs prove the crawler obeyed robots.txt and its delays?
   - Bugs, dead code, unclear modules, comment style.
3. **Argue:** defend as the builder, rebut as the reviewer, until a position survives both. If the novelty claim fails, change the design, not the wording.
4. **Fix** everything that survived, rerun all experiments, and update every number.

Stop when a round finds nothing serious, or after three rounds.

## Running unattended

When you run overnight under `/goal`, nobody will answer questions, so:

1. Never wait for input: log a blocker in PROGRESS.md and move to the next priority.
2. Update PROGRESS.md and commit after every working milestone, so the morning review and any resumed session start from it.
3. Monitor the live crawl through its log; never restart it in a way that breaks politeness.
4. Run the smoke test, tests and evaluation in the session, so their output is visible to the goal evaluator.
5. No destructive git operations (force-push, history rewrites) and nothing published publicly.

## Final output and your judgment

**Your call, documented in the README with reasons:** folder layout, module boundaries, libraries, data structures, thresholds, weights, set sizes and crawl scale. A CLI is enough; the frontend is not graded.

**Report back to the team with:**

1. The novelty claim and why it survived review
2. What changed across critique rounds
3. Benchmark tables and chart files
4. Known weaknesses, stated honestly
5. A checklist of every PDF requirement with its status
6. The AI-use log
7. Confirmation that the README commands run from a clean clone
