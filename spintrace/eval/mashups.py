"""Rule-based mashups of Wikinews originals with known sources and shares, to evaluate multi-source tracing.

Each mashup splices one contiguous block of sentences from each of two or three originals, in known
proportions, with the blocks in a random order. Easy mashups combine random originals; hard ones combine
originals published within 7 days of each other that share rare terms, so the sources compete for one
query. Levels: the splice as is, a light edit (the word swaps and sentence drops of rewrites.light_edit,
with its boilerplate opening and closing), and a WordNet synonym spin, which defeats shingles. All
sources of a mashup come from one split. Deterministic, written to evaldata/mashups.jsonl.
"""
import hashlib
import json
import random
from collections import Counter

from .. import config, store, text as tx
from . import rewrites, wikinews
from .run import split_of

OUT = config.ROOT / "evaldata" / "mashups.jsonl"
MIXES = [(0.5, 0.5), (0.7, 0.3), (0.4, 0.3, 0.3), (0.5, 0.25, 0.25)]
LEVELS = ["splice", "splice_light", "splice_synonym"]
VARIANTS = ["easy", "hard"]
PER_CELL = 12
TARGET_SENTS = 16
MIN_BLOCK = 2
HARD_DAYS = 7
HARD_SHARED = 3
RARE_DF = 200


def originals():
    """The 400 originals, then the Wikinews distractors (also CC BY), which give same-week partners for hard mixes."""
    origs = [json.loads(l) for l in rewrites.SRC.open()]
    for o in origs:
        o["main"] = True
    origs += [dict(o, main=False) for o in wikinews.distractors()]
    for o in origs:
        o["doc_id"] = store.doc_id_for(o["url"])
        o["split"] = split_of(o["doc_id"])
        o["sents"] = tx.sentences(o["text"])
    df = Counter(t for o in origs for t in set(tx.content_terms(tx.tokens(o["text"]))))
    for o in origs:
        o["rare"] = {t for t in tx.content_terms(tx.tokens(o["text"])) if 2 <= df[t] <= RARE_DF and not t.isdigit()}
    return origs


def block_sizes(mix, lens):
    """Sentences per source: shares of a total that every source can supply, at most TARGET_SENTS."""
    total = min([TARGET_SENTS] + [int(n / p) for p, n in zip(mix, lens)])
    sizes = [max(MIN_BLOCK, round(p * total)) for p in mix]
    if any(s > n for s, n in zip(sizes, lens)) or sum(sizes) < 8:
        return None
    return sizes


def hard_partners(o, pool):
    """Originals within HARD_DAYS of o sharing at least HARD_SHARED rare terms, most shared first."""
    near = [(len(o["rare"] & b["rare"]), b) for b in pool if b is not o
            and abs(b["published"] - o["published"]) <= HARD_DAYS * 86400]
    near = [x for x in near if x[0] >= HARD_SHARED]
    near.sort(key=lambda x: (-x[0], x[1]["doc_id"]))
    return [b for _, b in near]


def pick_sources(pool, k, variant, rng, used):
    mains = [o for o in pool if o["main"]]
    for _ in range(400):
        a = rng.choice(mains)
        if variant == "easy":
            rest = rng.sample([b for b in mains if b is not a], k - 1)
        else:
            cands = hard_partners(a, pool)
            if len(cands) < k - 1:
                continue
            rest = []
            for b in cands:
                if all(abs(b["published"] - x["published"]) <= HARD_DAYS * 86400 for x in rest):
                    rest.append(b)
                if len(rest) == k - 1:
                    break
            if len(rest) < k - 1:
                continue
        key = tuple(sorted(x["doc_id"] for x in [a] + rest))
        if key not in used:
            used.add(key)
            return [a] + rest
    return None


def edit_block(sents, level, rng):
    text = " ".join(sents)
    if level == "splice_light":
        text = rewrites.light_edit(text, rng)
        text = text.removeprefix("According to reports, ").removesuffix(" Stay tuned for more updates on this story.")
    elif level == "splice_synonym":
        text = rewrites.synonym_spin(text, rng)
    return text


def make(srcs, mix, level, variant, rng_key):
    rng = random.Random(rng_key)
    sizes = block_sizes(mix, [len(o["sents"]) for o in srcs])
    if sizes is None:
        return None
    blocks = []
    for o, n in zip(srcs, sizes):
        start = rng.randint(0, len(o["sents"]) - n)
        blocks.append((o, start, n))
    order = list(range(len(blocks)))
    rng.shuffle(order)
    parts, labels = [], []
    for j in order:
        o, start, n = blocks[j]
        text = edit_block(o["sents"][start:start + n], level, rng)
        parts.append(text)
        labels += [j] * len(tx.sentences(text))
    if level == "splice_light":
        parts[0] = "According to reports, " + parts[0]
        parts.append("Stay tuned for more updates on this story.")
        labels.append(-1)
    text = " ".join(parts)
    if len(tx.sentences(text)) != len(labels):
        return None
    sources = [{"doc_id": o["doc_id"], "url": o["url"], "title": o["title"], "mix": p, "start": start,
                "length": n, "share": labels.count(j) / len(labels)}
               for j, ((o, start, n), p) in enumerate(zip(blocks, mix))]
    published = max(o["published"] for o in srcs) + rng.uniform(2 * 3600, 3 * 86400)
    return {"level": level, "variant": variant, "mix": "/".join(str(round(100 * p)) for p in mix),
            "n_sources": len(srcs), "sources": sources, "sentence_source": labels, "text": text,
            "published": published, "farm": f"mash{rng.randint(1, 12)}.example", "split": srcs[0]["split"],
            "generator": "rule-based"}


def build(per_cell=PER_CELL):
    origs = originals()
    rows = []
    for split in ("dev", "test"):
        pool = [o for o in origs if o["split"] == split and len(o["sents"]) >= 5]
        for variant in VARIANTS:
            for mix in MIXES:
                rng = random.Random(f"{split}|{variant}|{mix}")
                used, made = set(), 0
                while made < per_cell:
                    srcs = pick_sources(pool, len(mix), variant, rng, used)
                    if srcs is None:
                        break
                    base = hashlib.sha1("|".join([o["doc_id"] for o in srcs] + [str(mix)]).encode()).hexdigest()[:12]
                    out = []
                    for level in LEVELS:
                        m = make(srcs, mix, level, variant, f"{base}|{level}")
                        if m is None:
                            break
                        m["id"] = f"mash:{base}:{level}"
                        m["group"] = base
                        out.append(m)
                    if len(out) == len(LEVELS):
                        rows += out
                        made += 1
    OUT.parent.mkdir(exist_ok=True)
    with OUT.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return rows


def load():
    return [json.loads(l) for l in OUT.open()] if OUT.exists() else []
