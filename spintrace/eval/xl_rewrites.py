"""Cross-language rewrites and LLM mashups of Wikinews originals, used only as evaluation labels.

Hindi gets three derivative levels (faithful translation, SEO rewrite, summary) and one hard
negative (an independent Hindi article written from the original's fact sheet). Tamil, Telugu and
Bengali get a translation level each, to test the same bridge on other scripts. Test-split Hindi
items are also translated back to English for the translate-then-detect baseline. LLM mashups
combine two originals from the same fortnight. One ollama queue, appended to and cached in
evaldata/rewrites_xl.jsonl, so reruns never regenerate text.
"""
import json
import random
import re
import time

import requests

from .. import config, store
from .rewrites import OLLAMA, SRC, fact_sheet, rid_for
from .run import split_of

OUT = config.ROOT / "evaldata" / "rewrites_xl.jsonl"
MODEL = "gemma3:4b"
LANGS = {"hi": "Hindi", "ta": "Tamil", "te": "Telugu", "bn": "Bengali"}
HI_LEVELS = ["hi_translation", "hi_seo", "hi_summary", "hi_facts_only"]
NEGATIVE = {"hi_facts_only"}
MAX_WORDS = 220
N_HI, N_OTHER, N_MASH = 44, 12, 16

PROMPTS = {
    "translation": "Translate the following English news article into {lang}. Output only the translation.\n\n{text}",
    "seo": ("Rewrite the following English news article as an SEO-friendly, unique {lang} news article, so that it "
            "does not look translated or copied. Keep it as a news article of similar length. Output only the "
            "article in {lang}, without a headline or notes.\n\nArticle:\n{text}"),
    "summary": ("Summarise the following English news article as a short {lang} news brief of about 100 words. "
                "Output only the brief in {lang}, without a headline or notes.\n\nArticle:\n{text}"),
    "facts_only": ("You are a {lang} news reporter. Write an original {lang} news article of about 200 words "
                   "reporting the facts below. Use your own structure, angle and wording; do not invent new facts. "
                   "Output only the article in {lang}, without a headline.\n\n{text}"),
    "back": "Translate the following news article into English. Output only the translation.\n\n{text}",
    "mash": ("Combine the following two news articles into one new news article of about 300 words that uses "
             "material from both. Output only the article, without a headline or notes.\n\nArticle 1:\n{a}\n\n"
             "Article 2:\n{b}"),
}


def head(text, n=MAX_WORDS):
    """The first n words, cut back to a sentence end, so each generation stays under a minute."""
    words = text.split()
    if len(words) <= n:
        return text
    cut = " ".join(words[:n])
    end = max(cut.rfind(". "), cut.rfind(".\n"))
    return cut[:end + 1] if end > len(cut) // 2 else cut


def llm(prompt, num_predict=1200):
    r = requests.post(OLLAMA, json={"model": MODEL, "stream": False, "think": False, "prompt": prompt,
                                    "options": {"temperature": 0.7, "seed": 7, "num_predict": num_predict}},
                      timeout=900)
    r.raise_for_status()
    out = re.sub(r"<think>.*?</think>", "", r.json()["response"], flags=re.S).strip()
    lines = [l for l in out.split("\n") if l.strip() and not l.strip().startswith(("#", "**", "Note", "Here"))]
    return "\n".join(lines).strip()


def pick_originals():
    """Originals in file order, alternating dev and test so both splits get the same number."""
    origs = [json.loads(l) for l in SRC.open()]
    for o in origs:
        o["split"] = split_of(store.doc_id_for(o["url"]))
    dev = [o for o in origs if o["split"] == "dev"]
    test = [o for o in origs if o["split"] == "test"]
    out = []
    for a, b in zip(dev, test):
        out += [a, b]
    return out


def mash_pairs(origs, n):
    """Pairs of originals published within 14 days of each other, so the mix is plausible news."""
    rng = random.Random(358)
    pool = sorted(origs[:200], key=lambda o: o["published"])
    pairs = []
    for i, a in enumerate(pool):
        near = [b for b in pool[i + 1:i + 8] if abs(b["published"] - a["published"]) < 14 * 86400]
        if near and len(pairs) < n and rng.random() < 0.5:
            pairs.append((a, rng.choice(near)))
    return pairs


def jobs():
    origs = pick_originals()
    for o in origs[:N_HI]:
        for level in HI_LEVELS:
            yield {"kind": "xl", "orig": o, "level": level}
            if o["split"] == "test":
                yield {"kind": "back", "orig": o, "level": level}
    for lang in ("ta", "te", "bn"):
        for o in origs[:N_OTHER]:
            yield {"kind": "xl", "orig": o, "level": f"{lang}_translation"}
    for a, b in mash_pairs(origs, N_MASH):
        yield {"kind": "mash", "orig": a, "other": b, "level": "mash_llm"}


def build():
    done = {}
    if OUT.exists():
        for l in OUT.open():
            r = json.loads(l)
            done[r["rid"]] = r
    with OUT.open("a") as out:
        for j in jobs():
            o, level = j["orig"], j["level"]
            lang = level.split("_")[0]
            if j["kind"] == "xl":
                rid = rid_for(o["title"], level)
            elif j["kind"] == "back":
                rid = rid_for(o["title"], "bt:" + level)
            else:
                rid = rid_for(o["title"] + "|" + j["other"]["title"], level)
            if rid in done:
                continue
            rng = random.Random(rid)
            t0 = time.time()
            row = {"rid": rid, "orig_title": o["title"], "orig_url": o["url"], "generator": MODEL}
            if j["kind"] == "xl":
                kind = level.split("_", 1)[1]
                src = fact_sheet(o) if kind == "facts_only" else head(o["text"])
                text = llm(PROMPTS[kind].format(lang=LANGS[lang], text=src))
                negative = level in NEGATIVE
                delay = rng.uniform(3600, 12 * 3600) if negative else rng.uniform(2 * 3600, 3 * 86400)
                row.update(level=level, lang=lang, text=text, published=o["published"] + delay,
                           farm=f"{'khabar' if negative else 'farm'}{rng.randint(1, 8)}.{lang}.example")
            elif j["kind"] == "back":
                src = done.get(rid_for(o["title"], level))
                if src is None:
                    continue
                row.update(level="bt:" + level, lang="en", src_rid=src["rid"], text=llm(PROMPTS["back"].format(text=src["text"])))
            else:
                b = j["other"]
                text = llm(PROMPTS["mash"].format(a=head(o["text"], 200), b=head(b["text"], 200)), 1000)
                row.update(level=level, lang="en", text=text, orig_titles=[o["title"], b["title"]],
                           orig_urls=[o["url"], b["url"]],
                           published=max(o["published"], b["published"]) + rng.uniform(2 * 3600, 3 * 86400),
                           farm=f"farm{rng.randint(1, 12)}.example")
            row["seconds"] = round(time.time() - t0, 1)
            done[rid] = row
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()
            print(f"xl {row['level']} {row['seconds']}s {o['title'][:60]}", flush=True)


def load(levels=None):
    if not OUT.exists():
        return []
    rows = [json.loads(l) for l in OUT.open()]
    return [r for r in rows if levels is None or r["level"] in levels]
