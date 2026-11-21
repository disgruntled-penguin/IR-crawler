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
languages; a word seen in 3 or more crawled Hindi pages whose English match is ten times rarer (idf gap
above 1) is a common word. Measured as the `no_idf_check` ablation. Short keys (two consonants) get no
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
