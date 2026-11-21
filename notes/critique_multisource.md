# Critique: multi-source tracing (final phase, 21-22 Nov 2026)

Runs: `python -m spintrace eval-multi` on the 20,860-document English index. Test: 382 rule-based splices (96 per
level, 94 splice_seo), 260 single-source controls, 150 no-source controls (90 facts-only, 60 crawl hard negatives).
Every number below is test unless marked dev.

## Round 1 (first full run, tuned for source-set F1 alone)

Headline: splice F1 0.976, against midterm 0.513, one-shot multi 0.856, shingle containment 0.917 (at their own
max-F1 dev thresholds).

**Reviewer.** SpinTrace names a source for 59% of facts-only articles (written by an LLM from the original's fact
sheet, never its text), against 16% for the midterm verifier as shipped. The objective rewards that: mashup source
pairs outnumber the 90 facts-only items, so a low threshold buys recall with false sources. The brief says same
story is not the same as copied, and that this is the test that matters most.

**Builder.** Agreed. Diagnosis on dev only (`round signals of true and false sources`): at the residual level a
facts-only article looks more like a source than an LLM-paraphrased block does (residual coverage 0.56 against
0.33, word overlap 0.45 against 0.41). The separating evidence is the rest of the suspect: in a facts-only article
the sentences that do not align are still about the original's story (median similarity 0.50), in a mashup they
belong to another story (0.22-0.30). Two signals added: `res_rest_sim` and `res_related`. And every method is now
tuned for its best dev F1 among thresholds that give a source to at most 10.6% of dev no-source items, the shipped
midterm's own rate there.

**Disclosure.** The false-source problem was first seen in round 1's test numbers. The diagnosis, the two signals
and the bar used dev only, and the bar is the shipped midterm's dev rate, not a tuned value; but the decision to
fix it was informed by a test number.

## Round 2 (current numbers, under the false-source bar)

| | SpinTrace | shingle containment | midterm (= one-shot) |
| --- | --- | --- | --- |
| all splices | 0.870 | 0.742 | 0.116 |
| splice (verbatim) | 1.000 | 0.981 | 0.209 |
| splice_light | 0.992 | 0.970 | 0.182 |
| splice_synonym | 0.938 | 0.582 | 0.057 |
| splice_seo | 0.329 | 0.050 | 0.000 |
| false source, facts-only | 0.078 | 0.178 | 0.111 |
| false source, hard negatives | 0.033 | 0.000 | 0.017 |

**Reviewer 1: is IR doing real work, or is the residual step decoration?** The no-residual ablation (rounds query the
whole suspect each time, everything else equal, re-tuned on dev) drops splice F1 only from 0.870 to 0.844 (2.6
points), recall of the second of two sources from 0.812 to 0.771 and of the third of three from 0.705 to 0.637. The
retrieval table says why: one rare-term query for the whole suspect already has the second source in its top 5 for
91% of 2-source splices and the third for 85% of 3-source ones (top 20: 98% and 92%).

**Builder.** Conceded in part. The residual query matters for the 9-16% of minor sources outside the whole-text top 5,
and it lowers the facts-only false-source rate (0.111 to 0.078), but it is not the main gain. The main gain is the
per-round verification criterion: with the midterm's whole-document verifier inside the same rounds, splice F1 is
0.118. The README now says this plainly instead of crediting the residual query.

**Reviewer 2: is this a near-duplicate demo?** Shingle containment against every earlier document scores 0.981 on
verbatim splices and 0.970 on light edits, about as well as SpinTrace and 50 times cheaper (1.5 ms against 99 ms).

**Builder.** Agreed, and stated: on copy-level splices the textbook check is enough. SpinTrace's gain over shingles is
on paraphrase: synonym-spun splices (0.938 against 0.582) and SEO-rewrite splices (0.329 against 0.050).

**Reviewer 3: where does it lose?** splice_seo F1 is 0.329 under the bar (0.894 without it, at a 52% facts-only
false-source rate). Single-source SEO rewrites get their source only 64% of the time (shipped midterm 76%) and
summaries 80% (midterm 97%): the strict round threshold (0.95) costs single-source recall. Share estimates are good
for copy-level splices (MAE 0.000-0.046) and poor for splice_seo (0.314; 78% of its sentences stay unexplained
because paraphrased sentences fall under the 0.62 alignment cut).

**Builder.** All stated in the README. In a crawler the midterm single-source pass should stay first; the
multi-source pass adds further sources for what it flags.

**Reviewer 4: why does one-shot multi equal the midterm?** Both are tuned under the same bar and land on the same
threshold (0.854) of the midterm's whole-document probability. A whole-document verifier passes a candidate only if
it explains most of the suspect; in a 2- or 3-source mashup at most one source can, and usually none does (the
midterm finds the larger source of a 2-source splice 23% of the time). So no second candidate ever clears the bar and
one-shot is capped at one source like the midterm. Without the bar one-shot reaches 0.856 F1 but gives a source to
98% of facts-only articles.

**Reviewer 5: leakage?**
- Thresholds, the stop share, the cover value, the round verifier's weights and every baseline threshold are fit on
  dev; the trade-off chart sweeps thresholds on test as a curve, with the dev-tuned points marked.
- All 759 splices have every source in their own split (checked: 0 violations, no source in both splits).
- Synthetic rewrites of a mashup's sources (and of a control's original) are hidden from retrieval, so an exact copy
  cannot stand in for the original.
- LLM mashups are never used for tuning. Some early pairs mix a dev and a test original, so they are reported twice:
  all of them, and only those with both sources in test.

**Reviewer 6: hard mashups?** Wikinews rarely covers one event twice, so "hard" means same week and at least 3 shared
moderately rare terms, not the same event. Same-event mixing is only tested by the LLM mashups and the live pass.

**Reviewer 7: dates.** The parametric filter keeps undated documents, and the rounds apply no provenance check
beyond it, so an undated later page could be named as a source. Mashup dates are synthetic.

**Reviewer 8: live.** 13 of 174 flagged crawled articles get two or more verified sources and none of 300 unflagged
ones get any; in 6 of the 13 the two sources share at least half their shingles, so they are one wire text printed
by two outlets, not a mashup. Unjudged.

**Reviewer 9: LLM mashups.** On the 34 gemma3:4b mashups that existed at run time (23 with both sources in test),
SpinTrace under the bar finds 34% of the sources with no wrong one (F1 0.505 [0.34, 0.65]; test-only 0.562); every
baseline finds almost none (containment 1.5%). Without the bar SpinTrace reaches 0.985 and one-shot multi 0.962, so on
these items too the residual query adds almost nothing; what costs recall is the threshold the facts-only articles
force. n is small and the generator copies a lot of each original's wording.

## Conclusion

The claim survives in a narrower form: a per-round verifier over the unexplained sentences recovers second and third
sources that a whole-document verifier structurally cannot, at a false-source rate no worse than the midterm's, on
copy-level and synonym-level mashups. On LLM-paraphrased mashups it finds a minority of sources at that rate, and the
residual query itself adds little over one rare-term query.
