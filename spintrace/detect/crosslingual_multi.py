"""Mashups across languages: a suspect in an Indian script stitched from several English articles.

The explain-away rounds of detect/multisource.py use a verifier fit on partial coverage of English
sentences. For other scripts the cross-language verifier already handles passages the size of a short
brief, so the suspect is cut into blocks of a few sentences, each block is traced on its own through the
anchor index, and blocks that name the same source are merged. A source's share is the fraction of the
suspect's sentences in the blocks that name it.
"""
from .. import text_indic as ti
from . import crosslingual as xl

BLOCK_SENTENCES = 4


def blocks(text, size=BLOCK_SENTENCES):
    """Consecutive runs of `size` sentences; a one-sentence tail joins the block before it."""
    sents = ti.sentences(text)
    out = [sents[i:i + size] for i in range(0, len(sents), size)]
    if len(out) > 1 and len(out[-1]) < 2:
        tail = out.pop()
        out[-1] += tail
    return out


def block_suspect(s, i, sentences):
    return {**s, "doc_id": f"{s['doc_id']}#b{i}", "text": " ".join(sentences)}


def trace_blocks(ctx, s, exclude=None, size=BLOCK_SENTENCES):
    """Per block: sentence count and the best non-undated candidate with its verifier probability.

    The probabilities are kept so a threshold can be applied afterwards (sources_at) without tracing again.
    """
    out = []
    for i, b in enumerate(blocks(s["text"], size)):
        rep = xl.analyse_text(ctx, block_suspect(s, i, b), exclude=exclude)
        ok = [v for v in rep["verified"] if v["verdict"] != "uncertain"]
        best = max(ok, key=lambda v: v["prob"]) if ok else None
        out.append({"block": i, "sentences": len(b), "doc": best["doc"] if best else None,
                    "prob": best["prob"] if best else 0.0, "url": best["url"] if best else None,
                    "anchors": rep["stats"].get("anchors")})
    return out


def sources_at(raw, threshold):
    """Merge blocks into sources at a probability threshold: [{doc, sentences, share, blocks, prob}]."""
    total = sum(b["sentences"] for b in raw) or 1
    found = {}
    for b in raw:
        if b["doc"] is not None and b["prob"] >= threshold:
            e = found.setdefault(b["doc"], {"doc": b["doc"], "url": b["url"], "sentences": 0, "blocks": [], "prob": 0.0})
            e["sentences"] += b["sentences"]
            e["blocks"].append(b["block"])
            e["prob"] = max(e["prob"], b["prob"])
    out = sorted(found.values(), key=lambda e: -e["sentences"])
    for e in out:
        e["share"] = e["sentences"] / total
    return out


def trace(ctx, s, exclude=None, size=BLOCK_SENTENCES, threshold=None):
    """Sources of one suspect and the share of the text that no source explains."""
    raw = trace_blocks(ctx, s, exclude, size)
    srcs = sources_at(raw, ctx.model["threshold"] if threshold is None else threshold)
    return {"blocks": raw, "sources": srcs, "unexplained": 1.0 - sum(e["share"] for e in srcs)}
