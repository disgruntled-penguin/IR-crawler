# Critique round 2 (run with summaries, 2026-10-07 ~00:55)

Index 11,607 docs (crawl ~5k, 400 Wikinews originals, 4,000 distractors, 1,600 rewrites + first facts-only).
Test: 190 per rule-based level, 90 SEO, 90 summary, 190 originals, 174 crawl hard negatives.

| test F1 | exact | light | synonym | SEO | summary | FPR hard |
| --- | --- | --- | --- | --- | --- | --- |
| SpinTrace | 0.995 | 0.995 | 0.995 | 0.989 | 0.978 | 0.011 |
| MinHash | 1.000 | 0.997 | 0.326 | 0.022 | 0.000 | 0.000 |
| Shingle containment | 0.948 | 0.948 | 0.948 | 0.800 | 0.844 | 0.109 |
| tf-idf cosine | 0.943 | 0.943 | 0.943 | 0.847 | 0.835 | 0.126 |
| BM25 | 0.974 | 0.974 | 0.974 | 0.844 | 0.942 | 0.052 |
| Dense cosine | 0.814 | 0.814 | 0.814 | 0.623 | 0.561 | 0.460 |

## Reviewer

1. **Ranking is still uninformative.** P@1 is 0.995-0.999 for tf-idf, BM25, dense and SpinTrace even with 4,000
   distractors. "Is the original ranked first?" cannot separate methods here.
2. **Most signals are dead weight.** Removing any one of seven embedding/structure signals moves macro F1 by
   <= 0.004, sometimes upwards. Only fact_overlap matters (0.990 -> 0.972, hard FPR 0.011 -> 0.034).
3. **Live precision is unmeasured and probably poor.** A spot check of 8 random live flags (round 1 model) found
   about 3 real derivatives; the rest were independent same-event reports (BBC vs CBS on one obituary, DW vs BBC on
   one outbreak) or shared wire copy. The crawl hard negatives exclude exactly this grey zone (containment > 0.1).
4. **Real farm content breaks verification.** TopGolf.kr's archived page is a list of AI title suggestions and
   reshaped deals; retrieval ranks the Wired source first, verification gives p = 0.001.
5. **Undated provenance mostly abstains.** Without dates, coverage asymmetry gets summaries right (0.96) but
   says "uncertain" for 80% of SEO rewrites, which are as long as their source.
6. **Dates are trusted blindly.** GlobalVillageSpace's metadata says it published 38 minutes after the NYT. A farm
   that backdates its pages slips past the earlier-only filter and the copy is never compared with its source.
7. **BM25 is closer than the headline suggests** on summaries (0.942 vs 0.978) and has a 5% hard FPR.

## Builder

1. Agreed that P@1 cannot rank methods here, and the README will say so. The ranking evidence that does
   discriminate is the query-budget experiment: 10 highest-idf terms reach Recall@1 0.987 touching ~160 postings;
   10 lowest-idf terms reach 0.691 (0.322 on SEO) touching ~26,600; 10 random terms 0.969 at ~6,400. That is
   index elimination doing the work the claim says it does. Facts-only same-event articles (being generated) are
   also ranking distractors, published hours after the original; the final run reports whether they confuse ranking.
2. Not changed on test evidence (that would be tuning on test). The finding supports the claim: the decisive
   verification signal is the IR-derived one (names and numbers), and an IR-signals-only verifier keeps 0.977.
   Embeddings buy ~1.3 F1 points and halve the hard-negative FPR. Pruning is future work, decided on dev.
3. The fix is data, not a hand-tuned feature: the facts-only level gives the LLM only an original's topic, names,
   numbers and quotes and asks for an original article, which is what independent coverage of one event looks
   like. It goes into dev (training negatives) and test (FPR). The live scan is rerun with the new verifier and
   the flags go to the two judges with `python -m spintrace judge live`. README states the spot check honestly.
4. A real limitation; reported with the case. The IR half of the claim holds on it.
5. Reported as is. With dates present the filter decides; only 0 of the crawl's saved articles lacked a date,
   because pages without one are rarely articles.
6. Fix: add a "dates untrusted" retrieval variant with no date filter and measure source recall, so the cost of
   trusting dates is a number. Provenance then rests on coverage asymmetry for those cases.
7. Reported as is; the summary row and the BM25 line stay in the chart.

## Fixes

- F8 facts-only negatives (generating; final run).
- F9 retrieval variant without the date filter.
- F10 rescan the live crawl with the final verifier; regenerate the judging sheet.
