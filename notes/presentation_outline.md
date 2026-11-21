# Presentation outline (25 Nov 2026)

Scratch notes for the team, not a deliverable. The format (slides, video, time limit) is not confirmed, so this is written for about 10 minutes and cuts down to 5 by dropping slides 7 and 9. Every number below is in `results/` or the report; the report's tables are generated from those files.

## One-sentence answer to "what did you do differently since the midterm?"

The midterm assumes a copy shares wording with one original. A translation shares none, and a mashup has no single original. The final handles both with the same idea (facts survive rewriting, found through our own inverted index) and measures where it works and where it does not.

## Slides

1. **The problem.** NewsGuard: Grammarly's checker missed the source of 34 of 43 AI-rewritten articles. One line on the midterm result: names, numbers and quotes find the source where shingles do not.
2. **What the midterm cannot do.** Two examples side by side: a Hindi article translated from an English one (no shared words), and an article stitched from two sources (best match finds one and stops).
3. **Translations: the bridge.** Diagram from the README: numbers (`3.5 करोड़` to 35000000), dates, Latin tokens, names (transliteration, phonetic key, k-gram index with Jaccard and edit distance, tolerant retrieval from the textbook). Then the same inverted index and cosine. A multilingual sentence model only verifies five candidates.
4. **Translation results.** `results/crosslingual_f1_by_level.png`. Say it plainly: accuracy is comparable to translating back and running the English system (0.860 / 0.852 / 0.818 against 0.906 / 0.771 / 0.753; intervals overlap); plain tf-idf on the back-translation is strong. The win is cost: 17.4 s of LLM time per suspect against 65 ms (483 ms with a cold embedding cache), 36 times less, and no generative model in the crawl loop. Multilingual dense cosine finds the source (P@1 0.93) but flags 72% of same-event articles.
5. **Mashups.** Explain-away rounds: query again with only the sentences no source explains (relevance feedback). `results/multisource_recall_by_rank.png`: the midterm finds the largest source and almost never the second (0.036); we find 0.81 of second sources. Splice F1 0.870 against 0.742 for shingle containment; on plain copy-and-paste shingles do as well as we do (0.981), the gap is on reworded blocks. The explain-away step itself adds only 2.6 points; the per-source verification criterion carries the result.
6. **Hard negatives and controls.** Same-event articles: SpinTrace-XL names a false source for 10% (long) and 22% (130 words). Say that the short negatives exist because word count alone separates summaries from the long negatives (AUC 1.00).
7. **Where it fails (say this unprompted).** Tamil, Telugu and Bengali: the source is retrieved (P@1 0.42, 0.67, 0.92) but verification fails because the sentence model probably does not cover them. Format changes: no gain. Hindi mashups: recovers 62% of sources, but names a false source for 30% of same-event articles; a prototype.
8. **Live crawl.** 2,375 Indian-language pages traced against the English crawl, 15 flagged, 2 credit their source (BBC Hindi from BBC News). Unjudged: no precision is claimed.
9. **Polite crawler.** `crawl-report --since 2026-11-21`: 0 disallowed fetches, 0 requests under the 5 s floor, closest pair 20 s apart; robots.txt skips and backoffs in the log. Earlier crash (dropped connection) found, fixed and regression-tested.
10. **Dataset and what is next.** Dataset table from the report (`results/dataset.md`): live crawl, Wikinews (CC BY 2.5), our synthetic rewrites, CC-NEWS, NewsGuard pairs. Next: a verifier for languages the sentence model does not cover; judged live pairs; a real format-change test.

## Demo (inspectable output, seeded, not hand-picked)

The rids were drawn with `random.Random(7)` from the test split; run them in this order and show the failure too.

```
python -m spintrace kgram "नरेंद्र मोदी"                 # transliteration, phonetic key, 3-grams, matches
python -m spintrace suspect-xl 963f6ceb56c895ce         # hi_translation: anchors, query, candidates, alignment, DERIVED
python -m spintrace suspect-xl 963f6ceb56c895ce --absent  # source masked out: must not flag
python -m spintrace suspect-xl d960236dc7e32b5e         # hi_facts_only (same event, not copied): not flagged here
python -m spintrace sources mash:bea3d8200d0e:splice_light  # explain-away rounds with shares
python -m spintrace postings teheran                    # the inverted index underneath
```

The facts-only item above is not flagged; the false-positive rate is 10%, so say that some are flagged and show one if asked (`eval-xl` writes the per-item scores to `data/eval/xl_raw.pkl`).

## Questions to expect, with answers that match the numbers

- **Cross-language plagiarism detection already exists (CL-ESA, PAN).** Yes. We do not claim a new field. The claim is that the midterm's premise, run on a crawler's own positional index, extends to translation and mashups, and we measure against the obvious alternatives.
- **Why not translate first?** It works about as well (the table), and tf-idf on the translation is strong. It costs an LLM call per crawled page: 17.4 s here against 65 ms. At crawl scale that is the difference.
- **Why not just embed everything?** It finds the source but cannot decide: it flags 72% of same-event articles, and embedding 20,860 documents took 15 minutes and must be redone for each new page. Dense retrieval followed by our verifier is nearly as good as the full system (0.829 against 0.843 macro F1): we say so.
- **Is it learning length?** Partly for summaries. Word count alone has AUC 1.00 against the long negatives; with 130-word negatives the false-positive rate goes from 10% to 22%, so the true precision lies between.
- **Was anything tuned on test?** No: the verifier, thresholds and anchor settings come from dev; splits are by original story; back-translation baselines use English dev thresholds and an oracle row shows what test tuning would add. One exception to name: the Hindi mashup false-source rate was tuned to 10% on dev and is 30% on test, which we report.
- **Why synthetic data?** Real farm translations come without ground truth. Wikinews is freely licensed, so originals are known. The live crawl is real but unjudged (the team had no judging time).
- **Why do Tamil, Telugu and Bengali fail?** Retrieval works, verification does not; the sentence model's card lists Hindi, Marathi, Gujarati and Urdu but not these three, which fits; we did not inspect the alignment scores directly.
- **How do you know the crawler was polite?** The log lines and `crawl-report`: every fetch re-audited against the stored robots.txt (over 62,000 fetches, 0 disallowed), final-crawl gaps at least 20 s.
- **What is the baseline from the midterm?** Exact hash, MinHash Jaccard, shingle containment, tf-idf, BM25, dense cosine, each at its best dev threshold; the midterm tables are unchanged.
- **Does the AI-use declaration cover the final?** Yes: README AI-use log and report Section 8 (Claude Code wrote code and drafts; Gemma 3 4B generated the Indian-language test data locally).
