import sqlite3

import numpy as np

from spintrace import index, kgram, text_indic as ti, translit as tr
from spintrace.detect import crosslingual as xl


def test_indic_digits_and_indian_number_words():
    assert ti.normalize("२२ लोग") == "22 लोग"
    assert ti.normalize("௨௦௨௪") == "2024"
    assert ti.normalize("১৭৫") == "175"
    assert ti.numbers("३.५ करोड़ लोग") == ["35000000"]
    assert ti.numbers("1,00,000 रुपये और 2 लाख") == ["100000", "200000"]
    assert ti.numbers("चार मिलियन वर्ग किलोमीटर") == ["4000000"]
    assert ti.numbers("35 million people and four million more") == ["35000000", "4000000"]
    assert ti.numbers("7 लोग लापता") == []
    assert ti.numbers("3 லட்சத்து") == ["300000"]


def test_dates_meet_across_languages():
    assert ti.dates("26 जुलाई को रात") == [(7, 26)]
    assert ti.dates("on July 26 at night") == [(7, 26)]
    assert ti.dates("26 July") == [(7, 26)]


def test_tokens_keep_vowel_signs():
    toks = ti.tokens("नरेंद्र मोदी ने UAE में कहा।")
    assert toks[:2] == ["नरेंद्र", "मोदी"]
    assert "uae" in toks and "।" not in "".join(toks)


def test_translit_key_matches_english_name():
    hindi = [tr.name_key(w) for w in ti.tokens("नरेंद्र मोदी")]
    english = [tr.key(w) for w in "Narendra Modi".split()]
    assert hindi == english
    assert tr.name_key(ti.normalize("अल्बानेज़")) == tr.key("Albanese")
    assert tr.name_key(ti.normalize("ऑस्ट्रेलिया")) == tr.key("Australia")


def test_same_name_across_scripts():
    # One ISCII-offset table serves every script; Tamil does not write voicing, so both sides fold it.
    deva, taml = ti.normalize("नरेंद्र"), ti.normalize("நரேந்திர")
    assert tr.key(tr.word(deva), fold_voicing=True) == tr.name_key(taml) == tr.key("Narendra", fold_voicing=True)
    assert tr.name_key(ti.normalize("নরেন্দ্র")) == tr.name_key(ti.normalize("నరేంద్ర")) == tr.name_key(deva)


def test_kgram_lookup_with_edit_distance():
    assert kgram.levenshtein("kitten", "sitting") == 3
    assert kgram.grams("md") == {"$md", "md$"}
    kg = kgram.KGramIndex()
    for w in ["Albanese", "Albania", "Lebanese", "Modi", "Media", "Narendra"]:
        kg.add(tr.key(w), w.lower())
    kg.freeze()
    hits = kg.lookup(tr.name_key(ti.normalize("अल्बानेज़")))
    assert hits[0][0] == tr.key("Albanese") and hits[0][2] == 0 and "albanese" in hits[0][3]
    near = [h for h in kg.lookup("nrndrn")]
    assert near and near[0][3].most_common(1)[0][0] == "narendra" and near[0][2] == 1


def _toy():
    docs = [
        {"doc_id": "a", "title": "Albanese sends Wedgetail to UAE",
         "text": "Prime Minister Anthony Albanese said Australia will deploy the E-7A Wedgetail to the UAE. "
                 "About 85 Australian Defence Force personnel will go. The aircraft covers 4 million square km.",
         "published": 1000.0, "site": "news.example", "kind": "article", "origin": "crawl"},
        {"doc_id": "b", "title": "Rain in Sichuan",
         "text": "Heavy rain in Sichuan province killed 22 people in Panzhihua on July 26. Officials said 38 were hurt.",
         "published": 1000.0, "site": "wire.example", "kind": "article", "origin": "crawl"},
        {"doc_id": "c", "title": "Cricket",
         "text": "The match in Mumbai drew a crowd as India won by 85 runs. The captain Rohit spoke after the game.",
         "published": 1000.0, "site": "sport.example", "kind": "article", "origin": "crawl"},
    ]
    idx = index.Index().build(docs)
    con = sqlite3.connect(":memory:")
    con.execute("CREATE TABLE docs (doc_id TEXT, text TEXT, origin TEXT)")
    con.executemany("INSERT INTO docs VALUES (?,?,?)", [(d["doc_id"], d["text"], d["origin"]) for d in docs])
    con.row_factory = sqlite3.Row
    return idx, kgram.build(idx, con, path=_tmp_path())


def _tmp_path():
    import tempfile
    from pathlib import Path
    return Path(tempfile.mkdtemp()) / "kg.pkl"


def test_end_to_end_retrieval_on_toy_index():
    idx, lex = _toy()
    hindi = ("प्रधानमंत्री एंथनी अल्बानेज़ ने कहा कि ऑस्ट्रेलिया ई-7ए वेजटेल विमान संयुक्त अरब अमीरात (UAE) भेजेगा। "
             "लगभग 85 कर्मी जाएंगे। विमान चार मिलियन वर्ग किलोमीटर पर नज़र रखता है।")
    cands, stats, anchors = xl.retrieve(idx, lex, hindi, published=5000.0, site="farm.example")
    assert idx.doc_ids[cands[0]["doc"]] == "a"
    assert "4000000" in anchors.numbers and "uae" in anchors.latin
    assert any(n["matches"] and n["matches"][0][0] == tr.key("Albanese") for n in anchors.names)
    assert stats["postings_touched"] > 0
    rain = "रविवार को सिचुआन प्रांत के पंजिहुआ में भारी बारिश से 22 लोगों की मौत हो गई और 38 घायल हुए। यह 26 जुलाई की बात है।"
    cands, _, anchors = xl.retrieve(idx, lex, rain, published=5000.0, site="farm.example")
    assert idx.doc_ids[cands[0]["doc"]] == "b" and anchors.dates == [(7, 26)]
    # Only earlier documents qualify: a suspect published before everything finds nothing.
    cands, _, _ = xl.retrieve(idx, lex, rain, published=10.0, site="farm.example")
    assert cands == []
    mask = xl.allowed_mask(idx, lex, 5000.0, "wire.example")
    assert not mask[idx.doc_num["b"]] and mask[idx.doc_num["a"]]
    assert np.all(lex.latin)
