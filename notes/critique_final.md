# Final-phase critique (cross-cutting round, 22 Nov)

Scratch notes, not a deliverable. The per-feature reviews are in `critique_xl.md` and `critique_multisource.md`; this round asks what a reviewer who wants the project to fail would ask of both together. Each point has the attack, the defence, what survived, and what changed.

## 1. Is IR doing the work across languages, or is this an embedding model with extra steps?

Attack. The cross-language ablation shows "dense retrieval + our verifier" equals the full system (macro F1 0.984, P@1 1.000 on the preliminary run). So anchor retrieval does not beat dense retrieval on recall at this scale.

Defence. True, and the write-up must say so. What the anchors buy is not recall at 9k documents but (a) no corpus-wide embedding pass (about 15 minutes for 9.3k English documents here, linear in the corpus), because the inverted index already exists and a query touches about 80k postings; (b) every step is inspectable (numbers, names with their phonetic keys, k-gram matches); (c) dense retrieval alone, with its own best threshold, is the weak method (macro F1 about 0.56, facts-only false-positive rate 0.73): the verifier carries it, not the retrieval. Translate-then-detect with tf-idf or BM25 also ranks the source as well as we do; we win on cost (about 37 times faster, no LLM call per page).

Survived. The claim is rewritten as "traces translations without translating and without embedding the corpus, at the same accuracy as translate-then-detect", not "more accurate". At web scale the cost argument is untested: stated as untested.

## 2. Were the baselines given a fair chance?

Attack. Back-translation baselines were run only on test items, so their thresholds come from the English dev split, not from their own distribution.

Defence. The CSV carries `@oracle` rows where each baseline's threshold is chosen on test itself; they are an upper bound and are not better than the tuned rows. Exact hash and MinHash score 0 because a translation shares no shingles, which is the point of the experiment, not a setup that makes them lose: their best threshold gives them the best possible chance.

Survived, with the oracle rows kept in the table.

## 3. Are the hard negatives hard, and is there a style artefact?

Attack. Positives are Hindi derivatives of the first 220 words of an original, written by one 4B model. Negatives (`hi_facts_only`) are written from the original's fact sheet, so they differ in length and in how closely they follow the source. A verifier could pick up length or generator style instead of derivation.

Defence. The negatives share names, numbers and quotes with the source by construction, so anchor overlap cannot separate them (the cross-validation on dev dropped the anchor-overlap signals for exactly this reason). Alignment signals decide.

Checked (22 Nov, 44 originals, same on dev and test): word count alone separates Hindi summaries (median 117 to 120 words) from the same-event negatives (median 218 to 230 words) perfectly, AUC 1.00. Translations (AUC 0.52) and SEO rewrites (0.76) are mostly not separable by length. So the summary level's low false-positive rate was weaker evidence than it looked, and the verifier's alignment features could in principle be using length indirectly.

Fix. A second same-event negative, `hi_facts_short`, written from the same fact sheet but asked for about 100 words, generated for every test original (with back-translations for the translate-then-detect baseline). It is a test-only diagnostic (dev thresholds are unchanged), so a threshold tuned against the long negatives is exposed if it lets short ones through. The final table reports the false-positive rate on both negative sets side by side.

## 4. Was anything tuned on test, or leaked between originals and rewrites?

Both agents checked: thresholds, verifier, stop share and cover value fit on dev; no original in both splits; synthetic rewrites of a query's own original are hidden from retrieval; the k-gram lexicon is built only from the English documents all methods search. Two disclosures remain. The mashup agent noticed the false-source problem on a run-1 test number, then fixed it using dev only; this is stated in its notes and README. And 5 of the first 16 LLM mashups mix a dev and a test source, so those are reported on the both-sources-in-test subset; the 32 later ones are same-split.

## 5. What happens to provenance when dates are missing or wrong?

Synthetic derivatives get a publish time after their original, so dates make provenance trivial there. The undated experiment from the midterm (coverage asymmetry) is unchanged and is not rerun for Hindi or mashups. Backdated farms remain a stated weakness (dates are trusted). Not fixed: out of scope this round.

## 6. Does the live crawl show it works on real farms?

No, and the report must not imply it. 1 of 633 crawled Indian-language pages is flagged against English sources; the flag is shared PTI wire copy, not a farm. The likely reasons are structural: most Hindi stories have no English source in our crawl. The multi-source live pass flags 13 of 174 articles, 6 of them shared wire text. Nobody has judged any of these (no judging time), so no live precision is claimed for either feature.

## 7. Would a grader call this known work, and is the claim still true?

Yes in part, and we must say so. Cross-language plagiarism detection is an established area (CL-ESA, cross-language character n-grams, translation-based detection), and finding every source of a suspicious document is the PAN source-retrieval task, which also builds queries from the suspicious text. What we claim is narrower and checkable: the same facts-survive-rewriting idea, run on a crawler's own positional inverted index for news, extends to translations (names, numbers and dates through our own transliteration and k-gram index) and to several sources (residual queries from the sentences not yet explained), with per-source shares, tested against the obvious baselines and against same-story negatives. The report's related-work paragraph names the prior work; bibliographic details to be verified before submission.

Is the midterm claim intact? Yes: the English single-source path is unchanged (checked by the unchanged tests and by regenerating the midterm tables from the cached scores).

## 8. What would make a reviewer fail us on honesty?

- Reporting the splice F1 of 0.870 without the shingle baseline at 0.742 and the 0.981 shingles reach on plain splices. Both are in the tables.
- Presenting the explain-away step as the source of the gain. The no-residual ablation shows +2.6 points; the larger gain is the per-source verification criterion. Stated.
- Hiding that the format-change stretch did not help. Reported as a negative result.
- Small samples. The Hindi set was grown from 11 to about 25 test positives per level overnight; intervals are reported on every number.

## Actions from this round

1. Reword the cross-language claim (comparable accuracy, cost and no-translation win) in the README and report. Done; reworded again after round 2 (below).
2. Related-work paragraph with the prior art in the report. Done.
3. Length-matched negatives (`hi_facts_short`) generated and scored on test. Done: 22% against 10% for the long negatives (round 2).
4. Keep the `@oracle` rows and the both-sources-in-test subset in the tables (already in the agents' output).

## Addendum: Hindi mashups (prototype added 22 Nov, 03:15)

Attack 1. The first pairing of originals (by sorted id) reshuffled every pair whenever a rewrite was added, so the cache silently dropped most items and a `--report-only` run reported on 15 test mashups instead of 68. Found while reviewing the summary file. Fixed by pairing in the originals' fixed generation order (decided on reproducibility grounds, before comparing the two sets of numbers). Side effect worth stating: the two pairings give F1 0.633 and 0.705 on the same originals, so differences of about 0.07 on this set are noise.

Attack 2. The method names a false source for 28% of long same-event articles on test although dev was under the 10% bar. The threshold was tuned at the edge of the bar on 25 dev negatives, which is one or two items. Not fixed by re-tuning on test (that would be tuning on test). Reported as a weakness; the final run, with a larger dev set, gives the threshold a better chance.

Attack 3. Splices are an easy mashup: contiguous blocks of whole translated sentences with no stitching. Real farm mashups interleave and paraphrase. The SEO-derived splices are the harder variant and score lower. No live Hindi mashup was tested.

Survived: the claim is only that the two features compose (passage tracing recovers about half the sources where whole-text tracing recovers under a third), with the false-source rate stated beside it.

## Round 2: after the rerun on the full sets (22 Nov, 05:20)

The Hindi set grew from 22 to 49-50 test originals per level, so every claim was rechecked against the new tables before the report was rebuilt.

Attack 1. The overnight headline (F1 0.957 / 0.933 / 0.884, ahead of translate-then-detect) did not survive. On 49-50 items SpinTrace-XL scores 0.860 / 0.852 / 0.818 and the back-translated English SpinTrace 0.906 / 0.771 / 0.753; plain tf-idf on the back-translation reaches 0.897 on translations and ranks the source first for every derivative. All intervals overlap. Fixed: the report and README no longer claim parity or a lead on accuracy; they say neither route is more accurate, that translate-then-detect is a strong baseline, and that the win is cost (17.4 s of LLM time per suspect against 65 ms cached and 483 ms cold, 36 times). The cold figure was measured on 30 suspects with the encoder warm; it excludes model load.

Attack 2. Precision is only about 0.75 at every level because the same-event false-positive rate is 10% on long negatives and 22% on 100-word ones. The earlier "0.000 / 0.091" were small-sample luck. Fixed: the numbers are in the text and the length paragraph now says the rise is in the direction the length worry predicts, so part of the summary-level precision was length; the short negatives still leave an AUC of 0.72 against summaries, so the truth lies between the two rates.

Attack 3. The length-AUC figures in the report were typed by hand from an earlier, smaller set. Fixed: `spintrace/eval/xl_length_check.py` writes them to `results/crosslingual_length_check.json` and the report reads them as macros. Same for the cold cost (`xl_cold_cost.py`). Both are in `scripts/final_run.sh`.

Attack 4. Hindi mashups on the full set: passage tracing F1 0.765 (recall 0.62, precision 1.00) against 0.528 for the whole text, but it names a false source for 30% and 28% of same-event articles although its dev rate was at most 10%. The block size also moved to 3 (it was 4 on the first run), which shows the setting is not stable on small sets. Stated as a prototype in the report, as before.

Attack 5. Other scripts: retrieval is uneven (P@1 0.42 Tamil, 0.67 Telugu, 0.92 Bengali) not simply "transfers". Reworded. Verification failure is unchanged; the sentence model's language coverage is still not checked against its model card.

Attack 6. The crawler stores category archive pages with a date in the markup as articles (for example "Health Archives", 368 words). They are extra distractors, not positives or negatives, so no result depends on them, but they inflate the document count. Left in and disclosed in the crawler section.

Attack 7. Do the logs prove politeness for the final crawl? `crawl-report --since 2026-11-21` now reports the final crawl alone: 0 disallowed fetches, 0 requests under the 5 s floor, closest pair 20 s apart. The 39 delay violations in the cumulative report are all from 6 and 7 October.

Survived: the novelty claim (translations and mashups need retrieval machinery the midterm lacks) with the honest result that accuracy is comparable to translate-then-detect and the gain is cost and inspectability. Nothing found in this round invalidates the midterm path.

Not done: no live result is judged; the dense baseline was never given the anchors' time window, which would only help it; the cost figures are one machine and a 20k-document index.
