# Critique round 1 (interim run, 2026-10-06 ~23:55)

Run: 4,954 docs (3.2k crawl, 400 Wikinews, 1.35k rewrites; LLM summaries not generated yet, SEO partial).
Test: 190 per rule-based level, 65 SEO, 190 originals, 169 hard negatives.

Interim headline: SpinTrace F1 0.984 (exact/light/synonym), 0.948 (SEO); MinHash 0.395 synonym, 0.000 SEO;
tf-idf cosine 0.952/0.737; dense cosine 0.930/0.321. FPR on hard negatives: SpinTrace 0.036, tf-idf 0.112, dense 0.142.

## Reviewer

1. **Ranking proves nothing.** tf-idf, BM25 and dense all have P@1 = 1.0. The originals are Wikinews 2022-26
   stories among Oct 2026 crawl pages; nothing in the index is about the same story. "The original is ranked
   first" is trivially true for any method.
2. **Remove the IR stage and nothing collapses.** Dense candidates + the same verifier: macro F1 0.973 vs 0.975.
   So is IR doing real work?
3. **Hard negatives are mislabelled.** Of the 6 SpinTrace "false positives" on hard negatives, 3 have shingle
   containment 0.26-0.85 against some earlier doc (Indian Express <- Independent Paramount piece: 0.85). They are
   wire copies. The miner only checked the mined partner, not the whole index.
4. **Baselines set up to lose?** MinHash only scores LSH candidates (32 bands x 4 rows, ~0.42 Jaccard knee), so a
   rewrite below the knee scores 0 by construction.
5. **Are the signals pulling weight?** Ablations move macro F1 by at most 0.03; removing align_coverage or
   quote_overlap *raises* it slightly. Maybe the verifier is mostly fact_overlap + order.
6. **Cost claim unsupported.** SpinTrace wall time 17 ms/suspect vs 6 ms for full-doc tf-idf. "Touches fewer
   postings" was asserted, but the full-doc query's postings were never counted.
7. Print pages (`articleshowprint`) and listing pages got into the store.

## Builder

1. Agreed; the corpus lacks same-story distractors. Wikinews has ~20k published articles, many follow-ups of
   the same ongoing stories (elections, wars, epidemics). Adding the 4,000 next-most-recent as distractors makes
   ranking a real test, and it is exactly the case where rare facts (this article's numbers and quotes) should beat
   topical similarity.
2. Recall does not collapse, and the README must say so. But the claim is not "dense cannot retrieve"; it is that
   the facts a rewrite keeps are enough to retrieve the source with an inverted index, and that verification plus
   provenance, not similarity, makes the decision. The test that matters is the mirror image: keep IR, drop
   the embedding signals. If an IR-only verifier (rare_cos, shingle containment, quotes, facts) gets close, then the
   embeddings decorate the IR, not the other way round.
3. Agreed, a real bug in the evaluation. Fix: a hard negative needs containment < 0.1 against every earlier
   document, not just its partner. The remaining flagged ones go to the judges (`evaldata/hard_negatives.csv`).
4. Added an all-pairs shingle containment baseline with no LSH: the best case any shingle method can reach.
5. Report it as is. Signals are correlated on easy levels; the ablation must be read on SEO/summary and on hard
   negatives (order: 0.036 -> 0.065 FPR without it; fact_overlap: 0.975 -> 0.944).
6. Count postings touched by the full-document query too, and report both honestly. The wall-time gap is the
   phrase queries and Python overhead, not the scoring.
7. Skip print and terms/policy slugs in the URL filter.

## Position that survives

The claim holds on detection (the shingle family collapses on synonym/SEO; SpinTrace keeps precision against
same-event coverage), but the ranking evaluation was too easy and the IR-necessity question is open until the
"no embedding signals" ablation and harder distractors are in. Fixes F1-F7 below.

## Fixes

- F1 hard negatives: containment < 0.1 against all earlier docs; print pages excluded.
- F2 4,000 extra Wikinews articles as same-site distractors (origin `wikinews_distractor`).
- F3 ablation "no embedding signals" (IR-derived signals only).
- F4 postings touched by the full-document tf-idf query.
- F5 URL filter: print pages.
- F6 all-pairs shingle containment baseline; fact-boosted query (variant without it kept); undated provenance.
