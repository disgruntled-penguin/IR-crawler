"""Hindi mashups: an Indian-script article spliced from two English originals' Hindi derivatives.

Items are built by rule from the Hindi translations and SEO rewrites already in evaldata/rewrites_xl.jsonl: the first
k sentences of one derivative followed by the first 10 - k sentences of another, in one split, so the true sources
and shares are exact. Controls are the single-source Hindi derivatives (the answer is one source) and the
same-event Hindi articles (the answer is none). Methods: the cross-language tracer on the whole text (it can name
one source) and passage tracing (detect/crosslingual_multi.py). The block size and the probability threshold are
tuned on dev under a false-source bar; every reported number is test.

Run as `python -m spintrace.eval.xl_multi_eval`; `--show N` prints the block-by-block trace of N seeded random test
mashups; `--limit K` runs K random items per split as a smoke test.
"""
import argparse
import csv
import json
import pickle
import random
import time
from collections import defaultdict

import numpy as np

from .. import config, text_indic as ti
from ..detect import crosslingual as xl, crosslingual_multi as xm
from . import xl_eval

RAW = config.DATA / "eval" / "xlm_raw.pkl"
SHAPES = (0.5, 0.6, 0.7)
LENGTH = 10
SIZES = (3, 4, 5)
FALSE_BAR = 0.10
GRID = [round(0.50 + 0.025 * i, 3) for i in range(20)]
CONTROL_LEVELS = ("hi_translation", "hi_seo", "hi_summary")
NEGATIVE_LEVELS = ("hi_facts_only", "hi_facts_short")


def mashup_items(items):
    """Splice mashups of two Hindi derivatives of different originals from one split, with exact sources."""
    pool = defaultdict(list)
    for it in items:
        if it["level"] in ("hi_translation", "hi_seo"):
            pool[(it["level"], it["split"])].append(it)
    out = []
    for (level, split), its in sorted(pool.items()):
        its = sorted(its, key=lambda x: x["rid"])
        for a, b in zip(its[0::2], its[1::2]):
            sa, sb = ti.sentences(a["text"]), ti.sentences(b["text"])
            for p in SHAPES:
                k = round(p * LENGTH)
                if len(sa) < k or len(sb) < LENGTH - k:
                    continue
                out.append({"rid": f"hm:{a['rid']}:{b['rid']}:{p}", "level": "mash_" + level[3:], "kind": "mash",
                            "text": " ".join(sa[:k] + sb[:LENGTH - k]), "published": max(a["published"], b["published"]),
                            "site": a["site"], "split": split, "group": a["group"], "gold": [a["orig"], b["orig"]],
                            "shape": p, "exclude": sorted(set(a["exclude"]) | set(b["exclude"]))})
    return out


def control_items(items):
    out = []
    for it in items:
        if it["level"] in CONTROL_LEVELS:
            out.append({**it, "kind": "single", "gold": [it["orig"]]})
        elif it["level"] in NEGATIVE_LEVELS:
            out.append({**it, "kind": "negative", "gold": []})
    return out


def best_whole(ctx, s, exclude):
    rep = xl.analyse_text(ctx, s, exclude=exclude)
    ok = [v for v in rep["verified"] if v["verdict"] != "uncertain"]
    best = max(ok, key=lambda v: v["prob"]) if ok else None
    return {"doc": best["doc"] if best else None, "prob": best["prob"] if best else 0.0}


def compute(ctx, items, limit=None):
    """Trace every item whole and in blocks of each size; cached per item."""
    cache = pickle.load(RAW.open("rb")) if RAW.exists() else {}
    if limit:
        rng = random.Random(358)
        keep = []
        for sp in ("dev", "test"):
            for kind in ("mash", "single", "negative"):
                pool = [i for i in items if i["split"] == sp and i["kind"] == kind]
                keep += rng.sample(pool, min(limit, len(pool)))
        items = keep
    t0, new = time.time(), 0
    for n, it in enumerate(items):
        key = (it["rid"], it["level"])
        if key in cache:
            continue
        s, excl = xl_eval.suspect(it), xl_eval.exclude_mask(ctx.idx, it)
        t = time.perf_counter()
        whole = best_whole(ctx, s, excl)
        t_whole = time.perf_counter() - t
        t = time.perf_counter()
        blocks = {size: xm.trace_blocks(ctx, s, excl, size) for size in SIZES}
        cache[key] = {"whole": whole, "blocks": blocks, "t_whole": t_whole, "t_blocks": (time.perf_counter() - t) / len(SIZES)}
        new += 1
        if new % 20 == 0:
            print(f"traced {new} new items ({n + 1}/{len(items)}) {time.time() - t0:.0f}s", flush=True)
            RAW.parent.mkdir(parents=True, exist_ok=True)
            pickle.dump(cache, RAW.open("wb"))
    RAW.parent.mkdir(parents=True, exist_ok=True)
    pickle.dump(cache, RAW.open("wb"))
    ctx.emb.save()
    for it in items:
        it["res"] = cache[(it["rid"], it["level"])]
    return items


def predicted(it, method, size, thr):
    r = it["res"]
    if method == "whole":
        return [r["whole"]["doc"]] if r["whole"]["doc"] is not None and r["whole"]["prob"] >= thr else []
    return [e["doc"] for e in xm.sources_at(r["blocks"][size], thr)]


def prf(rows):
    """Micro precision, recall, F1 and exact-set rate over (pred, gold) pairs."""
    tp = sum(len(set(p) & set(g)) for p, g in rows)
    np_, ng = sum(len(set(p)) for p, _ in rows), sum(len(g) for _, g in rows)
    pr, rc = tp / np_ if np_ else 0.0, tp / ng if ng else 0.0
    f = 2 * pr * rc / (pr + rc) if pr + rc else 0.0
    return pr, rc, f, float(np.mean([set(p) == set(g) for p, g in rows])) if rows else 0.0


def false_rate(items, method, size, thr):
    neg = [it for it in items if it["kind"] == "negative"]
    return float(np.mean([bool(predicted(it, method, size, thr)) for it in neg])) if neg else 0.0


def tune(dev, method):
    """Best dev mashup F1 over block sizes and thresholds with a false-source rate on dev negatives under the bar."""
    mash = [it for it in dev if it["kind"] == "mash"]
    best = None
    for size in (SIZES if method == "blocks" else (0,)):
        for thr in GRID:
            if false_rate(dev, method, size, thr) > FALSE_BAR:
                continue
            f = prf([(predicted(it, method, size, thr), it["gold"]) for it in mash])[2]
            if best is None or f > best[0] or (f == best[0] and thr > best[2]):
                best = (f, size, thr)
    return best or (0.0, SIZES[0], GRID[-1])


def boot(rows, seed=358, b=1000):
    rng = np.random.RandomState(seed)
    fs = []
    for _ in range(b):
        idx = rng.randint(0, len(rows), len(rows))
        fs.append(prf([rows[i] for i in idx])[2])
    return float(np.percentile(fs, 2.5)), float(np.percentile(fs, 97.5))


def report(items):
    dev = [it for it in items if it["split"] == "dev"]
    test = [it for it in items if it["split"] == "test"]
    methods = {"whole text, one source (dev threshold)": ("whole", 0, tune(dev, "whole")[2]),
               "whole text, shipped threshold": ("whole", 0, xl.load_model()["threshold"]),
               "passage tracing": ("blocks",) + tune(dev, "blocks")[1:]}
    det, ctrl = [], []
    for name, (method, size, thr) in methods.items():
        for lvl in ("all", "mash_translation", "mash_seo"):
            sel = [it for it in test if it["kind"] == "mash" and (lvl == "all" or it["level"] == lvl)]
            rows = [(predicted(it, method, size, thr), it["gold"]) for it in sel]
            if not rows:
                continue
            pr, rc, f, ex = prf(rows)
            lo, hi = boot(rows)
            det.append({"method": name, "set": lvl, "n": len(sel), "block_size": size or "", "threshold": thr,
                        "precision": pr, "recall": rc, "f1": f, "f1_lo": lo, "f1_hi": hi, "exact_set": ex})
        for p in SHAPES:
            sel = [it for it in test if it["kind"] == "mash" and it["shape"] == p]
            if sel:
                a = np.mean([it["gold"][0] in predicted(it, method, size, thr) for it in sel])
                b = np.mean([it["gold"][1] in predicted(it, method, size, thr) for it in sel])
                det.append({"method": name, "set": f"shape {p}/{round(1 - p, 1)}", "n": len(sel), "block_size": size or "",
                            "threshold": thr, "recall": (a + b) / 2, "f1": "", "precision": "",
                            "f1_lo": a, "f1_hi": b, "exact_set": ""})
        for lvl in CONTROL_LEVELS:
            sel = [it for it in test if it["kind"] == "single" and it["level"] == lvl]
            if sel:
                extra = np.mean([len(set(predicted(it, method, size, thr)) - set(it["gold"])) > 0 for it in sel])
                found = np.mean([it["gold"][0] in predicted(it, method, size, thr) for it in sel])
                ctrl.append({"method": name, "level": lvl, "n": len(sel), "extra_source_rate": extra, "source_found": found})
        for lvl in NEGATIVE_LEVELS:
            sel = [it for it in test if it["kind"] == "negative" and it["level"] == lvl]
            if sel:
                ctrl.append({"method": name, "level": lvl, "n": len(sel),
                             "false_source_rate": float(np.mean([bool(predicted(it, method, size, thr)) for it in sel]))})
    cost = {"ms_whole": 1000 * float(np.mean([it["res"]["t_whole"] for it in test])),
            "ms_blocks": 1000 * float(np.mean([it["res"]["t_blocks"] for it in test if it["kind"] == "mash"] or [0]))}
    return det, ctrl, methods, cost


def write_csv(path, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def show(ctx, items, n, size, thr):
    rng = random.Random(358)
    test = [it for it in items if it["split"] == "test" and it["kind"] == "mash"]
    for it in rng.sample(test, min(n, len(test))):
        r = it["res"]["blocks"][size]
        print(f"\nmashup {it['rid']}  shape {it['shape']}/{round(1 - it['shape'], 1)}  sources "
              f"{[ctx.row(g)['url'] for g in it['gold']]}")
        for b in r:
            tag = "-> " + str(b["url"]) if b["doc"] is not None and b["prob"] >= thr else "-> (none above threshold)"
            print(f"  block {b['block']} ({b['sentences']} sentences) best p={b['prob']:.3f} {tag}")
        got = xm.sources_at(r, thr)
        print("  sources found:", [(e["url"], round(e["share"], 2)) for e in got])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", type=int, default=0)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--report-only", action="store_true")
    a = ap.parse_args()
    ctx = xl.Context()
    base = xl_eval.build_items(ctx, xl_eval.load_rows())
    items = mashup_items(base) + control_items(base)
    if a.report_only and RAW.exists():
        cache = pickle.load(RAW.open("rb"))
        items = [{**it, "res": cache[(it["rid"], it["level"])]} for it in items if (it["rid"], it["level"]) in cache]
    else:
        items = compute(ctx, items, a.limit)
    det, ctrl, methods, cost = report(items)
    config.RESULTS.mkdir(exist_ok=True)
    write_csv(config.RESULTS / "xlmulti_detection.csv", det)
    write_csv(config.RESULTS / "xlmulti_controls.csv", ctrl)
    counts = {f"{sp}:{k}": sum(1 for it in items if it["split"] == sp and it["kind"] == k)
              for sp in ("dev", "test") for k in ("mash", "single", "negative")}
    (config.RESULTS / "xlmulti_params.json").write_text(json.dumps(
        {"methods": {k: {"method": v[0], "block_size": v[1], "threshold": v[2]} for k, v in methods.items()},
         "false_bar": FALSE_BAR, "cost_ms": cost, "items": counts}, indent=1, default=str))
    for r in det:
        if r["set"] in ("all", "mash_translation", "mash_seo"):
            print(f"{r['method']:42s} {r['set']:18s} n={r['n']:3d} P={r['precision']:.3f} R={r['recall']:.3f} "
                  f"F1={r['f1']:.3f} [{r['f1_lo']:.2f},{r['f1_hi']:.2f}] exact={r['exact_set']:.3f}")
    for r in ctrl:
        print(r)
    print("cost per suspect (ms):", cost, "| passage tracing params:", methods["passage tracing"])
    if a.show:
        _, size, thr = methods["passage tracing"]
        show(ctx, items, a.show, size, thr)


if __name__ == "__main__":
    main()
