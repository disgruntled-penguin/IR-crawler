"""Graded synthetic rewrites of Wikinews originals, used only as evaluation labels.

Levels, from easiest to hardest to trace: exact copy, light edit, synonym spin (a classic article
spinner), an LLM "SEO-friendly" rewrite like the one NewsGuard tested, and an LLM summary.
Every output is cached in evaldata/rewrites.jsonl so reruns never regenerate text.
"""
import hashlib
import json
import random
import re

import requests

from .. import config

LEVELS = ["exact", "light", "synonym", "seo", "summary"]
LLM_LEVELS = {"seo", "summary"}
OUT = config.ROOT / "evaldata" / "rewrites.jsonl"
SRC = config.ROOT / "evaldata" / "wikinews.jsonl"
OLLAMA = "http://localhost:11434/api/generate"
MODEL = "llama3.2:3b"

PROMPTS = {
    "seo": ("Rewrite the following news article to be SEO-friendly and unique so that it does not look copied. "
            "Keep it as a news article of similar length. Output only the rewritten article, without a headline "
            "or any notes.\n\nArticle:\n{text}"),
    "summary": ("Summarise the following news article as a short news brief of about 120 words. "
                "Output only the brief, without a headline or any notes.\n\nArticle:\n{text}"),
}

LIGHT_SWAPS = [
    (r"\bsaid\b", "stated"), (r"\bannounced\b", "revealed"), (r"\bhowever\b", "but"), (r"\balso\b", "additionally"),
    (r"\bdid not\b", "didn't"), (r"\bis not\b", "isn't"), (r"\bstarted\b", "began"), (r"\bpeople\b", "individuals"),
]


def rid_for(title, level):
    return hashlib.sha1(f"{title}|{level}".encode()).hexdigest()[:16]


def light_edit(text, rng):
    from ..text import sentences
    sents = sentences(text)
    keep = [s for i, s in enumerate(sents) if i == 0 or rng.random() > 0.12]
    out = " ".join(keep)
    for pat, rep in LIGHT_SWAPS:
        if rng.random() < 0.7:
            out = re.sub(pat, rep, out)
    return "According to reports, " + out + " Stay tuned for more updates on this story."


_wn = None


def _wordnet():
    global _wn
    if _wn is None:
        import nltk
        nltk_dir = config.DATA / "nltk"
        nltk.data.path.insert(0, str(nltk_dir))
        try:
            from nltk.corpus import wordnet
            wordnet.synsets("test")
        except LookupError:
            nltk.download("wordnet", download_dir=str(nltk_dir), quiet=True)
            from nltk.corpus import wordnet
        _wn = wordnet
    return _wn


def synonym_spin(text, rng, rate=0.5):
    """Replace about half of the lowercase content words with a WordNet synonym; names and numbers stay."""
    from .. import text as tx
    wn = _wordnet()

    def swap(m):
        w = m.group(0)
        if not w.islower() or w in tx.STOPWORDS or len(w) < 4 or rng.random() > rate:
            return w
        lemmas = [l.name().replace("_", " ") for s in wn.synsets(w)[:3] for l in s.lemmas()]
        lemmas = [l for l in lemmas if l.lower() != w and l.isascii()]
        return rng.choice(lemmas[:4]) if lemmas else w

    return re.sub(r"[A-Za-z]+", swap, text)


def llm(level, text):
    r = requests.post(OLLAMA, json={
        "model": MODEL, "stream": False, "prompt": PROMPTS[level].format(text=text),
        "options": {"temperature": 0.7, "seed": 7, "num_predict": 900},
    }, timeout=900)
    r.raise_for_status()
    out = r.json()["response"].strip()
    lines = out.split("\n")
    # Drop a leading headline line the model adds despite the instruction.
    if len(lines) > 1 and len(lines[0].split()) < 16 and not lines[0].rstrip().endswith("."):
        lines = lines[1:]
    return "\n".join(l for l in lines if not l.lower().startswith(("note:", "here is", "here's"))).strip()


def build(n_llm=200, levels=None):
    """Generate every missing (original, level) rewrite. LLM levels cover the first n_llm originals."""
    origs = [json.loads(l) for l in SRC.open()]
    done = set()
    if OUT.exists():
        done = {json.loads(l)["rid"] for l in OUT.open()}
    levels = levels or LEVELS
    with OUT.open("a") as out:
        for level in levels:
            for i, o in enumerate(origs):
                if level in LLM_LEVELS and i >= n_llm:
                    break
                rid = rid_for(o["title"], level)
                if rid in done:
                    continue
                rng = random.Random(rid)
                if level == "exact":
                    text = o["text"]
                elif level == "light":
                    text = light_edit(o["text"], rng)
                elif level == "synonym":
                    text = synonym_spin(o["text"], rng)
                else:
                    text = llm(level, o["text"])
                # A farm publishes hours to days after the original.
                delay = rng.uniform(2 * 3600, 3 * 86400)
                out.write(json.dumps({
                    "rid": rid, "orig_title": o["title"], "orig_url": o["url"], "level": level, "text": text,
                    "published": o["published"] + delay, "farm": f"farm{rng.randint(1, 12)}.example",
                    "generator": MODEL if level in LLM_LEVELS else "rule-based",
                }) + "\n")
                out.flush()
                print(f"rewrite {level} {i} {o['title'][:60]}", flush=True)
