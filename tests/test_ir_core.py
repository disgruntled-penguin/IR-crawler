import numpy as np

from spintrace import dedup, text as tx
from spintrace.index import Index


def docs():
    return [
        {"doc_id": "a", "title": "Nobel prize", "text": 'Francis Halzen won the Nobel prize. He said "the neutrinos came from far away galaxies".',
         "published": 10.0, "site": "a.com", "kind": "original", "origin": "crawl"},
        {"doc_id": "b", "title": "Prize news", "text": "The Nobel committee praised Halzen. Neutrinos are hard to detect.",
         "published": 20.0, "site": "b.com", "kind": "original", "origin": "crawl"},
        {"doc_id": "c", "title": "Cats", "text": "A cat with a broken back learned to walk again in 2023.",
         "published": 30.0, "site": "c.com", "kind": "original", "origin": "crawl"},
    ]


def test_tokens_fold_accents_numbers_and_possessives():
    assert tx.tokens("Teherán's 1,260 strikeouts") == ["teheran", "1260", "strikeouts"]


def test_quotes_and_facts():
    t = 'Iran said "we will fight if confronted militarily" on Sunday, according to Mohammad Ghalibaf, who cited 1,200 ships.'
    assert tx.quotes(t) == ["we will fight if confronted militarily"]
    names, nums = tx.facts(t)
    assert "mohammad ghalibaf" in names and "1200" in nums


def test_positional_postings_and_phrase_query():
    idx = Index().build(docs())
    p = idx.postings("nobel")
    assert p.docs.tolist() == [0, 1]
    assert p.positions(0).tolist() == [4]
    assert idx.phrase("won the nobel prize") == {0}
    assert idx.phrase("nobel prize won") == set()
    assert idx.postings("neutrinos", zone="quote").docs.tolist() == [0]


def test_idf_and_cosine_rank_the_matching_doc_first():
    idx = Index().build(docs())
    t = idx.vocab["halzen"]
    assert idx.df[t] == 2
    assert abs(idx.idf[t] - np.log10(3 / 2)) < 1e-9
    q = idx.query_vector({idx.vocab["cat"]: 1, idx.vocab["walk"]: 1})
    assert idx.cosine(q, k=1)[0][0] == 2


def test_minhash_estimates_jaccard_and_lsh_finds_near_duplicates():
    base = " ".join(f"word{i}" for i in range(300))
    near = base.replace("word150", "changed")
    far = " ".join(f"other{i}" for i in range(300))
    sa, sb, sc = dedup.shingles(base), dedup.shingles(near), dedup.shingles(far)
    true = dedup.jaccard(sa, sb)
    est = dedup.est_jaccard(dedup.minhash(sa), dedup.minhash(sb))
    assert abs(true - est) < 0.1
    lsh = dedup.LSH()
    lsh.add("b", dedup.minhash(sb))
    lsh.add("c", dedup.minhash(sc))
    assert lsh.query(dedup.minhash(sa)) == {"b"}
    assert dedup.containment(sb, sa) > 0.9
