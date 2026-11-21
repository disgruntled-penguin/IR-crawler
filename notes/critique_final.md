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

1. Reword the cross-language claim (accuracy parity, cost and no-translation win) in the README and report. Not done yet.
2. Related-work paragraph with the prior art in the report. Not done yet.
3. Length-matched negatives (`hi_facts_short`) generated and scored on test. Generation running; result goes here after the final run. If the short-negative false-positive rate is far above the long one, the summary level's precision claim is withdrawn and the verifier is refit with short negatives in dev.
4. Keep the `@oracle` rows and the both-sources-in-test subset in the tables (already in the agents' output).
