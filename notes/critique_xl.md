# Critique: cross-language tracing (SpinTrace-XL)

Scratch notes from the review rounds on the Indian-language extension; not a deliverable.

## Round 1 (first end-to-end run, 21 test positives, 30-day window)

**Reviewer: the candidate pool is tiny, so retrieval looks perfect for free.** With the live system's
30-day window, a synthetic suspect sees 150-380 earlier documents, because the crawl is concentrated in
Sep-Oct 2026 and the Wikinews originals are older. Every method ranked the source first.
*Builder:* the window is right for live use (rewrites appear within days). *Resolution:* the evaluation
now searches every earlier document from another site (about 9,000) for every method, and the window is
a separate variant row. Changed the design, not the wording.

**Reviewer: the name anchors are mostly noise.** The first queries turned common Hindi words into rare
English names by sound: रात (night) became Rohit, प्रधान (prime) became Pepperdine, बचाव (rescue) Bechev.
*Builder:* the true source still ranked first because it collects many anchors at once. *Resolution:*
a cross-language document-frequency check. A name keeps roughly its document frequency across
languages; a word seen in 3 or more crawled Hindi pages whose English match is much rarer (idf gap
above 1 at first, 0 after the round 2 dev sweep) is a common word. Measured as the `no_idf_check` ablation. Short keys (two consonants) get no
edits.

**Reviewer: a date phrase alone pulled unrelated stories in.** Any 'July 26' story scored 0.2 with no
other anchor and outranked real candidates. *Resolution:* date and name-bigram phrase hits only rerank
the cosine shortlist (3k documents); they never add documents on their own.

**Reviewer: is IR doing real work, or is the multilingual model doing it all?** In round 1 the dense
baseline also ranked every source first, and dense retrieval followed by our verifier matched the full
system. *Builder:* dense retrieval needs every document embedded (15 minutes for 20.9k heads on this
machine, and again for every crawled page), while the anchor query reuses the inverted index the crawler
already keeps; dense cosine alone cannot reject same-event articles (FPR 0.86 on facts-only). The honest
position is in the README: on this corpus, anchors and dense retrieval find the source equally often; the
anchors win on cost and inspectability, and verification separates derived from same-event.

**Reviewer: translate-then-detect is a strong baseline you are hiding.** Back-translating with the LLM
and running tf-idf or BM25 found the source as well. *Resolution:* reported in full, with its cost
(about 20 s of LLM time per suspect against about 0.3 s), and with an oracle test threshold as an upper
bound, since no Hindi dev item was back-translated.

**Reviewer: the anchor-overlap signals add nothing to the verifier.** The `no_anchor_signals` ablation
was not worse. *Resolution:* added the reverse directions (number_back, name_back: how much of the
candidate's numbers and English names the suspect repeats); kept the ablation and report it as is.

## Round 2 (after the 44-original run: 33 Hindi test positives, 44 test negatives)

**Reviewer: the anchor signals hurt on test, so you will quietly drop them.** Dropping them because test
said so would be tuning on test. *Resolution:* the verifier's feature set is now chosen by 5-fold
grouped cross-validation inside Hindi dev (groups are originals), as the midterm did for its signals.
Dev CV picked the alignment-only set (macro F1 0.914 against 0.848 with the anchor-overlap signals), so
anchors now do retrieval and alignment does verification. The anchor signals stay computed and printed
by `suspect-xl`, and the ablation table reports adding them back.

**Reviewer: F1 1.000 on hi_translation and 0/11 false positives on facts-only look too good.** Checks on
the cached run (`notes` only, no code path depends on them):
- No candidate of any item, in any variant, is an English synthetic rewrite of the item's own original
  (0 of all candidate lists); no source-absent item ever sees its source (0); no candidate is published at
  or after its suspect (0); no original has items in both splits (0).
- The k-gram lexicon is built from the names of the English documents in the index (the same documents
  any method searches; test originals are in the index for every method), and its source-language df
  table from crawled Indian-language pages only (origin 'crawl', fetched before a fixed time), never from
  the synthetic Hindi texts. The verifier, its features and every SpinTrace-XL threshold come from dev.
- The separation is real but the sample is small: translation probabilities are 0.985-0.996, facts-only
  0.026-0.725 (threshold 0.783), source-absent at most 0.296. A faithful translation aligns almost every
  sentence (coverage 0.86-1.00, mean cosine about 0.85); a facts-only article aligns at most half of its
  sentences, out of order (ordered coverage at most 0.40). The one facts-only article at 0.725 is the
  closest call. 0 of 11 still allows a false-positive rate up to 0.26 (Wilson), so the report now gives
  Wilson intervals for every proportion; the bootstrap interval of a perfect F1 collapses to [1, 1] and is
  not informative at n = 11. The extension to about 50 test originals is what will make these numbers
  meaningful.

**Reviewer: is the result reproducible?** Item sampling, bootstrap and cross-validation are seeded. Two
runs of a 40-item sample under different PYTHONHASHSEED values gave identical candidate lists after
three fixes: k-gram postings are read in sorted gram order with the key as the last tie-break, name words
are taken in sorted order (Counter.most_common keeps insertion order for ties), and the source-language
df table only counts pages fetched before 2026-11-22 00:00 IST, so the growing crawl does not change it.

**Reviewer: 1 flagged of 633 live pages means the method does not work on real data.** What the live run
measures, in order of effect:
1. *Coverage.* Today's pages are searched in an index of the English pages crawled in the same days
   (1.8k at the time of that run). The Hindi outlets (Aaj Tak, Bhaskar, Amar Ujala, India TV, TV9, ABP)
   mostly cover Indian regional, political, crime and entertainment news, written in Hindi or taken from
   the Hindi wires (PTI Bhasha, IANS, ANI), none of which is crawled. Their English source is usually not
   in the index, and then not flagging is the right answer.
2. *Same-outlet translations are excluded by design.* BBC Hindi translating BBC English, and HT Bangla or
   HT Telugu translating Hindustan Times, share one site, and the earlier-other-site filter removes them,
   because a publisher translating itself is not spin. The `--same-site` diagnostic lifts that filter for
   those two outlets to test recall on real translations whose source is in the index.
3. *Same event is not copied.* Most live pairs with shared anchors are independent coverage of one
   event; the alignment-based verifier is meant to reject those, as it rejects the facts-only articles.
4. *Wire copy.* The top live flags pair Aaj Tak with ThePrint, which republishes PTI: both are likely
   from the same wire story, which SpinTrace attributes to the earlier outlet (a known weakness).
The window and date filter are not what limits recall: the window is 30 days and the flagged lags were
1-16 hours. The name anchors are too loose rather than too strict on long live pages (median 131 matched
name tokens per page against about 50 on the synthetic items), which costs postings, not recall.

*Same-site diagnostic* (`live-xl --same-site`, 237 pages of BBC's language services and HT Bangla/Telugu,
allowed to match their own outlet): retrieval put the outlet's own English article first for 6 of 112 BBC
pages and 8 of 125 HT pages; one was flagged, a BBC Hindi article traced to its BBC News original 3.5 h
earlier (p = 0.90), and its Marathi sibling scored 0.73 (below threshold). BBC's language services adapt
rather than translate and HT's regional sites mostly run their own stories, so real full translations are
rare in this crawl; the diagnostic shows the path works on real pages, not a recall figure.

**Reviewer: the name check still lets common words through.** `suspect-xl` on a test rewrite showed भारी
(heavy; in 112 of 844 crawled Hindi pages) becoming 'bureau', the top-weighted query term, because the idf
gap allowed 1.0. *Resolution:* the gap and the minimum source df were swept on dev only (66 Hindi dev
positives): MRR 0.846 at gap 1.5, 0.912 at 1.0, 0.962 at 0.5, 1.000 at 0, with fewer postings touched, so the
gap is now 0 (a source word seen in 3 or more pages cannot map to an English term rarer than itself). The
cost, stated in the README: a name far more common in Hindi news than in our English crawl (मोदी) is no
longer an anchor. Also fixed: '8:00' produced the number 0 as an anchor.
