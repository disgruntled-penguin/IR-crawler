"""Format changes: a copy that turns its source into a listicle or a Q&A keeps the facts but not the prose order.

The midterm verifier leans on sentence alignment in the source's order (ordered_coverage, align_order), which a
reshaped copy breaks. This experiment generates rule-based format rewrites of the Wikinews originals
(deterministic, evaldata/formats.jsonl), scores them with every signal including the optional order-free fact
signals, and compares the midterm verifier as shipped with verifiers refit on dev with and without those signals.
Negatives are the facts-only articles and crawl hard negatives, because an order-free fact signal is exactly
what an article written from the same fact sheet also shares.
"""
import json
import pickle
import random
import re

import numpy as np

from .. import config, store, text as tx
from ..detect import candidates, pipeline, signals, verify
from . import hardneg, ms_eval, report, rewrites
from .run import split_of

OUT = config.ROOT / "evaldata" / "formats.jsonl"
RAW = config.DATA / "eval" / "ms_formats.pkl"
FORMATS = ["listicle", "qa"]
CONTROLS = ["seo", "summary", "facts_only"]
HARDNEG_SAMPLE = 150
MAX_ITEMS = 12
SPIN = 0.3
FACT_FEATURES = list(signals.OPTIONAL)
CONNECTIVE = re.compile(r"^(However|Meanwhile|Moreover|Furthermore|Additionally|Also|In addition|According to [^,]+),\s+")


def fact_sentences(text):
    return [s for s in tx.sentences(text) if any(tx.facts(s))]


def _bullet(s, rng):
    s = CONNECTIVE.sub("", s)
    s = s[0].upper() + s[1:]
    return rewrites.synonym_spin(s, rng, rate=SPIN)


def listicle(o, rng):
    """Fact-bearing sentences as a shuffled numbered list under a 'what we know' line, lightly spun."""
    sents = fact_sentences(o["text"])
    rng.shuffle(sents)
    lines = [f"Here is what we know so far about {o['title'].rstrip('.')}."]
    lines += [f"{i}. {_bullet(s, rng)}" for i, s in enumerate(sents[:MAX_ITEMS], 1)]
    return "\n".join(lines)


def question(s):
    for m in tx.CAPS.finditer(s):
        if m.start() > 0 and " " in m.group(0):
            return f"What do we know about {m.group(0)}?"
    nums = tx.NUMBER.findall(s)
    if nums:
        return f"What does the figure {nums[0]} refer to?"
    return "What else has been reported?"


def qa(o, rng):
    """A templated question before each fact-bearing sentence, in shuffled order, answers lightly spun."""
    sents = fact_sentences(o["text"])
    rng.shuffle(sents)
    lines = []
    for s in sents[:MAX_ITEMS]:
        lines += [question(s), _bullet(s, rng)]
    return "\n".join(lines)


def build():
    rows = []
    for l in rewrites.SRC.open():
        o = json.loads(l)
        for fmt in FORMATS:
            rid = rewrites.rid_for(o["title"], fmt)
            rng = random.Random(rid)
            text = listicle(o, rng) if fmt == "listicle" else qa(o, rng)
            rows.append({"rid": rid, "orig_title": o["title"], "orig_url": o["url"], "level": fmt, "text": text,
                         "published": o["published"] + rng.uniform(2 * 3600, 3 * 86400),
                         "farm": f"fmt{rng.randint(1, 12)}.example", "generator": "rule-based"})
    with OUT.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return rows


def items_for(ctx, syn):
    idx = ctx.idx
    items = []
    if not OUT.exists():
        build()
    for r in (json.loads(l) for l in OUT.open()):
        orig = store.doc_id_for(r["orig_url"])
        if orig in idx.doc_num:
            items.append({"id": f"fmt:{r['rid']}", "level": r["level"], "label": 1, "source": idx.doc_num[orig],
                          "split": split_of(orig), "group": orig, "text": r["text"], "published": r["published"],
                          "site": r["farm"], "n": None, "exclude": syn.get(orig, [])})
    for row in ctx.con.execute("SELECT doc_id, meta FROM docs WHERE origin='synthetic' ORDER BY doc_id"):
        m = json.loads(row["meta"])
        if m["level"] in CONTROLS and row["doc_id"] in idx.doc_num and m["orig_doc"] in idx.doc_num:
            n = idx.doc_num[row["doc_id"]]
            pos = m["level"] != "facts_only"
            items.append({"id": row["doc_id"], "level": m["level"], "label": int(pos),
                          "source": idx.doc_num[m["orig_doc"]] if pos else None, "split": split_of(m["orig_doc"]),
                          "group": m["orig_doc"], "n": n, "exclude": [x for x in syn.get(m["orig_doc"], []) if x != n]})
    pairs = [(a, b) for a, b in hardneg.load() if a in idx.doc_num and b in idx.doc_num]
    for a, _ in random.Random(358).sample(pairs, min(2 * HARDNEG_SAMPLE, len(pairs))):
        items.append({"id": a, "level": "hard_negative", "label": 0, "source": None, "split": split_of(a), "group": a,
                      "n": idx.doc_num[a], "exclude": []})
    return items


def score_item(ctx, it):
    idx = ctx.idx
    excl = np.zeros(idx.N, dtype=bool)
    excl[it["exclude"]] = True
    if it["n"] is None:
        s = {"doc_id": it["id"], "text": it["text"], "published": it["published"], "site": it["site"]}
        ctx.feats.add(it["id"], it["text"])
        cands, _ = candidates.retrieve_text(idx, it["text"], it["published"], it["site"], exclude=excl)
    else:
        s = ctx.doc(it["n"])
        cands, _ = candidates.retrieve(idx, it["n"], s["text"], exclude=excl)
    top = cands[:pipeline.VERIFY_TOP]
    ctx.emb.prefetch([(s["doc_id"], s["text"])] + [(ctx.doc(c["doc"])["doc_id"], ctx.doc(c["doc"])["text"]) for c in top])
    rows = []
    for c in top:
        cd = ctx.doc(c["doc"])
        pair = signals.Pair(s, cd, ctx.feats, ctx.emb, c, idx=idx)
        verdict, _ = pipeline.provenance(s, cd, pair)
        rows.append({"doc": c["doc"], "signals": signals.compute(pair, report.FEATURES + FACT_FEATURES),
                     "verdict": verdict, "published": pipeline.date_of(cd)})
    return {"verified": rows, "spintrace": {"cands": [(c["doc"], c["score"]) for c in cands]}}


def run(ctx=None, fresh=False):
    if RAW.exists() and not fresh:
        with RAW.open("rb") as f:
            return pickle.load(f)
    ctx = ctx or ms_eval.context()
    items = items_for(ctx, ms_eval.rewrites_of(ctx))
    for i, it in enumerate(items):
        it.update(score_item(ctx, it))
        if (i + 1) % 200 == 0:
            ctx.emb.save()
            print(f"formats {i + 1}/{len(items)}", flush=True)
    ctx.emb.save()
    for it in items:
        it.pop("text", None)
    with RAW.open("wb") as f:
        pickle.dump(items, f, protocol=pickle.HIGHEST_PROTOCOL)
    return items


LEVELS = FORMATS + ["seo", "summary"]
NEGS = ["facts_only", "hard_negative"]


def table(name, model, thr, test):
    rows = []
    scored = [(it, *report.spin_score(it, model)) for it in test]
    negs = [x for x in scored if x[0]["label"] == 0]
    for level in LEVELS:
        pos = [x for x in scored if x[0]["level"] == level]
        group = pos + negs
        p, r, f = report.prf([s >= thr for _, s, _ in group], [it["label"] for it, _, _ in group])
        rows.append({"model": name, "level": level, "n_pos": len(pos), "precision": p, "recall": r, "f1": f,
                     "source_correct": float(np.mean([s >= thr and src == it["source"] for it, s, src in pos])),
                     "threshold": thr})
    for level in NEGS:
        ns = [s for it, s, _ in negs if it["level"] == level]
        rows.append({"model": name, "level": f"fpr_{level}", "n_pos": 0, "f1": None,
                     "fpr": float(np.mean([s >= thr for s in ns])), "n_neg": len(ns), "threshold": thr})
    return rows


def newsguard(ctx, models):
    """Rerun the NewsGuard pairs, in particular TopGolf.kr's reshaped deals list, under each verifier."""
    out = []
    for pair_id in ("topgolf_wired_deals", "gvs_waller"):
        farm = ctx.con.execute("SELECT doc_id FROM docs WHERE origin='newsguard' AND kind='suspect' AND "
                               "json_extract(meta, '$.pair_id')=?", (pair_id,)).fetchone()
        orig = ctx.con.execute("SELECT doc_id FROM docs WHERE origin='newsguard' AND kind='original' AND "
                               "json_extract(meta, '$.pair_id')=?", (pair_id,)).fetchone()
        if not farm or not orig or farm[0] not in ctx.idx.doc_num or orig[0] not in ctx.idx.doc_num:
            out.append({"pair": pair_id, "status": "not in store"})
            continue
        it = {"id": farm[0], "n": ctx.num(farm[0]), "exclude": [], "label": 1, "source": ctx.num(orig[0])}
        it.update(score_item(ctx, it))
        cands = [d for d, _ in it["spintrace"]["cands"]]
        for name, (model, thr) in models.items():
            prob = next((verify.score(v["signals"], model) for v in it["verified"] if v["doc"] == it["source"]), None)
            best, src = report.spin_score(it, model)
            out.append({"pair": pair_id, "model": name,
                        "rank_of_original": cands.index(it["source"]) + 1 if it["source"] in cands else None,
                        "p_original": prob, "threshold": thr, "flagged": best >= thr,
                        "traced_to_original": best >= thr and src == it["source"]})
    return out


def main(fresh=False):
    ctx = ms_eval.context()
    items = run(ctx, fresh)
    dev = [it for it in items if it["split"] == "dev"]
    test = [it for it in items if it["split"] == "test"]
    shipped = verify.load()
    same = report.fit_model(dev, report.FEATURES)
    fact = report.fit_model(dev, report.FEATURES + FACT_FEATURES)
    fact_only = report.fit_model(dev, ["rare_cos"] + FACT_FEATURES)
    models = {"midterm verifier (as shipped)": (shipped, shipped["threshold"]),
              "refit on dev, same signals": (same, same["threshold"]),
              "refit on dev, + order-free fact signals": (fact, fact["threshold"]),
              "fact signals and rare_cos only": (fact_only, fact_only["threshold"])}
    rows = []
    for name, (m, thr) in models.items():
        rows += table(name, m, thr, test)
    report.write_csv(config.RESULTS / "format_detection.csv", rows)
    (config.RESULTS / "format_verifier.json").write_text(json.dumps(fact, indent=2))
    ng = newsguard(ctx, models)
    report.write_csv(config.RESULTS / "format_newsguard.csv", ng)
    chart(rows, list(models))
    for r in rows:
        print("  " + ms_eval._fmt(r))
    for r in ng:
        print("  " + ms_eval._fmt(r))
    return rows, ng


def chart(rows, names):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cols = LEVELS + ["fpr_facts_only", "fpr_hard_negative"]
    tones = ["#eb6834", "#eda100", "#2a78d6", "#4a3aa7"]
    fig, ax = plt.subplots(figsize=(8.5, 4.6), dpi=150)
    w = 0.8 / len(names)
    for j, name in enumerate(names):
        ys = [next((r["f1"] if r["f1"] is not None else r["fpr"]) for r in rows if r["model"] == name and r["level"] == c)
              for c in cols]
        ax.bar(np.arange(len(cols)) + (j - (len(names) - 1) / 2) * w, ys, width=w, color=tones[j % 4], label=name)
    ax.set_xticks(range(len(cols)), [c.replace("fpr_", "FPR ") for c in cols])
    ax.set_ylim(0, 1.05)
    report._style(ax, "Format changes: F1 per level and false-positive rate (test)", "", "F1 / FPR")
    ax.legend(frameon=False, fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(config.RESULTS / "format_f1.png")
    plt.close(fig)
