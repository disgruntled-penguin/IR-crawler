import numpy as np

from spintrace.detect import candidates
from spintrace.index import Index

ORIGINAL = ('Health Minister Aden Duale said a 41-year-old Kenyan citizen died in Kisumu on Tuesday. '
            '"We have activated our emergency teams in Kisumu," Duale told reporters.')
REWRITE = ('Kenyan authorities confirmed on Tuesday that a citizen aged 41 had passed away in Kisumu, according to '
           'Aden Duale. "We have activated our emergency teams in Kisumu," the minister stated.')
SAME_TOPIC = 'Kenya reported heavy rain on Tuesday as the health ministry warned citizens about floods.'


def corpus():
    docs = [
        ("orig", ORIGINAL, 100.0, "a.com"),
        ("topic", SAME_TOPIC, 110.0, "b.com"),
        ("later", ORIGINAL, 900.0, "c.com"),
        ("rewrite", REWRITE, 500.0, "farm.example"),
    ]
    for i in range(20):
        docs.append((f"filler{i}", f"Unrelated story number {i} about markets, football and the weather today.", 50.0, f"f{i}.com"))
    return Index().build({"doc_id": d, "title": "", "text": t, "published": p, "site": s, "kind": "original",
                          "origin": "crawl"} for d, t, p, s in docs)


def test_earlier_mask_keeps_only_earlier_other_site_docs():
    idx = corpus()
    n = idx.doc_num["rewrite"]
    mask = candidates.earlier_mask(idx, n)
    assert mask[idx.doc_num["orig"]] and mask[idx.doc_num["topic"]]
    assert not mask[idx.doc_num["later"]] and not mask[n]


def test_rare_term_query_prefers_names_and_numbers():
    idx = corpus()
    n = idx.doc_num["rewrite"]
    q = candidates.rare_term_query(idx, n, m=5, facts=candidates.fact_terms(idx, REWRITE))
    terms = {idx.terms[t] for t in q}
    assert {"kisumu", "duale"} <= terms
    assert not terms & {"the", "on", "in"}


def test_retrieve_finds_the_source_of_a_paraphrase_over_same_topic_coverage():
    idx = corpus()
    n = idx.doc_num["rewrite"]
    cands, stats = candidates.retrieve(idx, n, REWRITE)
    assert cands[0]["doc"] == idx.doc_num["orig"]
    assert cands[0]["quote_frac"] == 1.0
    assert idx.doc_num["later"] not in [c["doc"] for c in cands]
    assert stats["n_quotes"] == 1


def test_budget_query_modes_differ_by_idf():
    idx = corpus()
    n = idx.doc_num["rewrite"]
    rare = candidates.budget_query(idx, n, 3, "rare")
    common = candidates.budget_query(idx, n, 3, "common")
    assert min(idx.idf[t] for t in rare) >= max(idx.idf[t] for t in common)
    assert len(candidates.budget_query(idx, n, 3, "random")) == 3
