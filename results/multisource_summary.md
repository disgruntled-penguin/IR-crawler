Operating point: every method takes its best dev source-set F1 among the thresholds that name a source for at most 10.6% of the dev facts-only articles and hard negatives, the shipped midterm verifier's own rate there (same story is not the same as copied).

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
| SpinTrace multi-source | (all) | 34 | 1.000 | 0.338 | 0.505 [0.34, 0.65] | 0.235 | 0.68 |
| Midterm single-source | (all) | 34 | 0.000 | 0.000 | 0.000 [0.00, 0.00] | 0.000 | 0.00 |
| One-shot multi | (all) | 34 | 0.000 | 0.000 | 0.000 [0.00, 0.00] | 0.000 | 0.00 |
| MinHash LSH | (all) | 34 | 0.000 | 0.000 | 0.000 [0.00, 0.00] | 0.000 | 0.00 |
| Shingle containment | (all) | 34 | 1.000 | 0.015 | 0.029 [0.00, 0.08] | 0.000 | 0.03 |
| Midterm, shipped threshold | (all) | 34 | 0.000 | 0.000 | 0.000 [0.00, 0.00] | 0.000 | 0.00 |
| SpinTrace multi-source | (both sources test) | 23 | 1.000 | 0.391 | 0.562 [0.36, 0.72] | 0.261 | 0.78 |
| Midterm single-source | (both sources test) | 23 | 0.000 | 0.000 | 0.000 [0.00, 0.00] | 0.000 | 0.00 |
| One-shot multi | (both sources test) | 23 | 0.000 | 0.000 | 0.000 [0.00, 0.00] | 0.000 | 0.00 |
| MinHash LSH | (both sources test) | 23 | 0.000 | 0.000 | 0.000 [0.00, 0.00] | 0.000 | 0.00 |
| Shingle containment | (both sources test) | 23 | 1.000 | 0.022 | 0.043 [0.00, 0.12] | 0.000 | 0.04 |
| Midterm, shipped threshold | (both sources test) | 23 | 0.000 | 0.000 | 0.000 [0.00, 0.00] | 0.000 | 0.00 |
| SpinTrace multi-source | (all), max-F1 thresholds without the bar | 34 | 1.000 | 0.971 | 0.985 [0.96, 1.00] | 0.941 | 1.94 |
| Midterm single-source | (all), max-F1 thresholds without the bar | 34 | 0.931 | 0.397 | 0.557 [0.47, 0.63] | 0.000 | 0.85 |
| One-shot multi | (all), max-F1 thresholds without the bar | 34 | 0.985 | 0.941 | 0.962 [0.93, 0.99] | 0.853 | 1.91 |
| MinHash LSH | (all), max-F1 thresholds without the bar | 34 | 0.000 | 0.000 | 0.000 [0.00, 0.00] | 0.000 | 0.00 |
| Shingle containment | (all), max-F1 thresholds without the bar | 34 | 1.000 | 0.559 | 0.717 [0.63, 0.80] | 0.235 | 1.12 |

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

**Shares** (SpinTrace, test splices; a missed source counts as share 0):

| level | mashups | true sources | share MAE | MAE, found sources | unexplained (est.) | unexplained (true) |
| --- | --- | --- | --- | --- | --- | --- |
| splice | 96 | 240 | 0.000 | 0.000 | 0.001 | 0.000 |
| splice_light | 96 | 240 | 0.004 | 0.000 | 0.073 | 0.067 |
| splice_synonym | 96 | 240 | 0.046 | 0.010 | 0.110 | 0.000 |
| splice_seo | 94 | 234 | 0.314 | 0.094 | 0.782 | 0.000 |
| all | 382 | 954 | 0.090 | 0.009 | 0.239 | 0.017 |

**Where the true sources are retrieved** (test splices): in the round-1 candidates of one query for the whole suspect, against found by the explain-away rounds:

| sources | share rank | mashups | round-1 top 5 | round-1 top 20 | found by rounds |
| --- | --- | --- | --- | --- | --- |
| 2 | 1 | 192 | 0.995 | 1.000 | 0.875 |
| 2 | 2 | 192 | 0.911 | 0.984 | 0.812 |
| 3 | 1 | 190 | 0.963 | 0.984 | 0.742 |
| 3 | 2 | 190 | 0.842 | 0.932 | 0.721 |
| 3 | 3 | 190 | 0.847 | 0.916 | 0.705 |

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

**Cost** (test): 99 ms per suspect for retrieval and verification over 1.66 rounds (10283 postings touched per round, 5 candidates verified per round); the suspects' sentence embeddings came from the cache in this run.

Tuned on dev: round threshold 0.95, stop below 0.0 unexplained, cover at sentence similarity 0.62; Midterm single-source threshold 0.854; One-shot multi threshold 0.854; MinHash LSH threshold 0.102; Shingle containment threshold 0.162; Midterm, shipped threshold threshold 0.804.
