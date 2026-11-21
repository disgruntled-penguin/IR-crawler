from spintrace import text_indic as ti
from spintrace.detect import crosslingual_multi as xm
from spintrace.eval import xl_multi_eval as ev


def hindi(n, tag="क"):
    return " ".join(f"{tag} खबर संख्या {i} में कुछ नया हुआ है।" for i in range(n))


def test_blocks_fold_a_one_sentence_tail_into_the_previous_block():
    sizes = [len(b) for b in xm.blocks(hindi(9), 4)]
    assert sizes == [4, 5]
    assert [len(b) for b in xm.blocks(hindi(8), 4)] == [4, 4]


def test_sources_at_merges_blocks_and_applies_the_threshold():
    raw = [{"block": 0, "sentences": 3, "doc": 5, "prob": 0.90, "url": "a"},
           {"block": 1, "sentences": 3, "doc": 5, "prob": 0.80, "url": "a"},
           {"block": 2, "sentences": 4, "doc": 7, "prob": 0.99, "url": "b"},
           {"block": 3, "sentences": 2, "doc": None, "prob": 0.0, "url": None}]
    strict = xm.sources_at(raw, 0.85)
    assert [(e["doc"], e["sentences"]) for e in strict] == [(7, 4), (5, 3)]
    loose = {e["doc"]: e["share"] for e in xm.sources_at(raw, 0.5)}
    assert loose[5] == 0.5 and loose[7] == 4 / 12
    assert sum(loose.values()) <= 1.0


def test_mashup_items_splice_exact_sources_within_one_split():
    def item(rid, orig, split):
        return {"rid": rid, "level": "hi_translation", "split": split, "text": hindi(12, rid[0]), "published": 100.0 + orig,
                "site": "farm1.hi.example", "group": f"g{orig}", "orig": orig, "exclude": [orig + 1000]}
    items = [item("a1", 1, "test"), item("b2", 2, "test"), item("c3", 3, "dev")]
    out = ev.mashup_items(items)
    assert len(out) == len(ev.SHAPES) and all(m["split"] == "test" for m in out)
    for m in out:
        assert m["gold"] == [1, 2] and m["published"] == 102.0
        assert len(ti.sentences(m["text"])) == ev.LENGTH
        assert m["exclude"] == [1001, 1002]
    first = {round(m["shape"] * ev.LENGTH): m["text"] for m in out}
    assert first[5].count("a") > 0 and "b खबर" in first[5]
