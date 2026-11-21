"""Multi-source evaluation: are all sources of a mashup found, with their shares, without inventing sources for
single-source rewrites and same-event articles?

Items (split by the first source's story; every source of a mashup is in the same split):
  splice      rule-based mashups of 2 or 3 Wikinews originals (evaldata/mashups.jsonl), three edit levels
  mash_llm    LLM mashups of two originals (evaldata/rewrites_xl.jsonl): the source set is the label, shares unknown
  single      single-source rewrites already in the index (light, synonym, seo, summary): exactly one source
  none        facts-only articles and crawl hard negatives: no source
Methods: SpinTrace explain-away rounds; the midterm single-source pipeline (at most one source); one-shot multi
(every round-1 candidate the midterm verifier accepts, no residual); MinHash LSH and shingle containment against
every earlier document. Every threshold, the round verifier and the stopping rule are fit on dev; numbers are test.
"""
import csv
import json
import pickle
import random
import time
from pathlib import Path
from collections import defaultdict

import numpy as np

from .. import config, dedup, embed, store
from ..detect import candidates, multisource, pipeline, signals, verify
from . import hardneg, mashups, xl_rewrites
from .run import split_of

OUT = config.RESULTS
CACHE = config.DATA / "eval"
EMB = config.INDEX_DIR / "ms_emb.pkl"
CONTROL_SAMPLE = {"light": 40, "synonym": 40, "seo": None, "summary": None, "facts_only": None}
HARDNEG_SAMPLE = 60
COVERS = [0.56, 0.62]
METHODS = ["spintrace", "midterm", "oneshot", "minhash", "containment"]
# The midterm at the threshold it ships with (tuned for whole-document rewrites), as a reference row only.
REFERENCE = ["midterm_shipped"]
LABELS = {"spintrace": "SpinTrace multi-source", "midterm": "Midterm single-source", "oneshot": "One-shot multi",
          "minhash": "MinHash LSH", "containment": "Shingle containment"}
COLORS = {"spintrace": "#2a78d6", "midterm": "#eb6834", "oneshot": "#eda100", "minhash": "#1baf7a",
          "containment": "#4a3aa7"}
TOP_BASE = 20


def context():
    return pipeline.LazyContext(emb=embed.OverlayEmbedder(EMB))


def rewrites_of(ctx):
    """orig doc_id -> index numbers of its synthetic rewrites, so a rewrite never stands in for its original."""
    out = defaultdict(list)
    for r in ctx.con.execute("SELECT doc_id, meta FROM docs WHERE origin='synthetic'"):
        if r["doc_id"] in ctx.idx.doc_num:
            out[json.loads(r["meta"])["orig_doc"]].append(ctx.idx.doc_num[r["doc_id"]])
    return out


def llm_items(ctx, path, syn):
    items = []
    for r in xl_rewrites.load(levels={"mash_llm"}) if path is None else _xl_rows(path):
        gold_ids = [store.doc_id_for(u) for u in r["orig_urls"]]
        if not all(g in ctx.idx.doc_num for g in gold_ids):
            continue
        gold = [ctx.idx.doc_num[g] for g in gold_ids]
        items.append({"id": f"mashllm:{r['rid']}", "kind": "mash_llm", "level": "mash_llm", "variant": "llm",
                      "mix": "llm", "n_sources": len(gold), "split": split_of(gold_ids[0]),
                      "split_all": sorted({split_of(g) for g in gold_ids}), "group": r["rid"], "gold": gold,
                      "gold_share": None, "text": r["text"], "published": r["published"], "site": r["farm"],
                      "n": None, "exclude": [m for g in gold_ids for m in syn.get(g, [])]})
    return items


def _xl_rows(path):
    if not path.exists():
        return []
    rows = [json.loads(l) for l in path.open()]
    return [r for r in rows if r.get("level") == "mash_llm"]


def build_items(ctx, syn):
    idx = ctx.idx
    items = []
    for m in mashups.load():
        ids = [s["doc_id"] for s in m["sources"]]
        if not all(d in idx.doc_num for d in ids):
            continue
        gold = [idx.doc_num[d] for d in ids]
        first = {j: m["sentence_source"].index(j) for j in range(len(ids)) if j in m["sentence_source"]}
        order = sorted(range(len(ids)), key=lambda j: (-m["sources"][j]["share"], first.get(j, 99)))
        items.append({"id": m["id"], "kind": "splice", "level": m["level"], "variant": m["variant"], "mix": m["mix"],
                      "n_sources": len(ids), "split": m["split"], "group": m["group"], "gold": gold,
                      "gold_rank": [gold[j] for j in order],
                      "gold_share": {gold[j]: m["sources"][j]["share"] for j in range(len(ids))},
                      "text": m["text"], "published": m["published"], "site": m["farm"], "n": None,
                      "exclude": [x for d in ids for x in syn.get(d, [])]})
    rng = random.Random(358)
    by_level = defaultdict(list)
    for r in ctx.con.execute("SELECT doc_id, meta FROM docs WHERE origin='synthetic' ORDER BY doc_id"):
        m = json.loads(r["meta"])
        if m["level"] in CONTROL_SAMPLE and r["doc_id"] in idx.doc_num and m["orig_doc"] in idx.doc_num:
            by_level[(m["level"], split_of(m["orig_doc"]))].append((r["doc_id"], m["orig_doc"]))
    for (level, split), rows in sorted(by_level.items()):
        k = CONTROL_SAMPLE[level]
        rows = rng.sample(rows, k) if k and len(rows) > k else rows
        for doc_id, orig in rows:
            n = idx.doc_num[doc_id]
            items.append({"id": doc_id, "kind": "none" if level == "facts_only" else "single", "level": level,
                          "variant": "-", "mix": "-", "n_sources": 0 if level == "facts_only" else 1, "split": split,
                          "group": orig, "gold": [] if level == "facts_only" else [idx.doc_num[orig]],
                          "gold_share": None, "n": n, "exclude": [x for x in syn.get(orig, []) if x != n]})
    hn = defaultdict(list)
    for a, b in hardneg.load():
        if a in idx.doc_num and b in idx.doc_num:
            hn[split_of(a)].append(a)
    for split, ids in sorted(hn.items()):
        for a in rng.sample(ids, min(HARDNEG_SAMPLE, len(ids))):
            items.append({"id": a, "kind": "none", "level": "hard_negative", "variant": "-", "mix": "-", "n_sources": 0,
                          "split": split, "group": a, "gold": [], "gold_share": None, "n": idx.doc_num[a],
                          "exclude": []})
    return items


def suspect_of(ctx, it):
    if it["n"] is not None:
        return multisource.Suspect.from_index(ctx, it["n"])
    return multisource.Suspect(it["id"], it["text"], it["published"], it["site"])


def exclusion(ctx, it):
    m = np.zeros(ctx.idx.N, dtype=bool)
    m[it["exclude"]] = True
    return m


def compact(tr):
    """A trace without the candidate lists, small enough to cache for every item and pass."""
    rounds = []
    for rd in tr["rounds"]:
        rounds.append({k: rd[k] for k in ("round", "share_left", "residual_sents", "prob", "new_covered",
                                         "postings_touched", "t_retrieve", "t_round")}
                      | {"sims": rd["sims"].astype(np.float16), "n_terms": len(rd["query_terms"]),
                         "chosen": {k: rd["chosen"][k] for k in ("doc", "doc_id", "url", "site", "prob")},
                         "verified": [{"doc": v["doc"], "prob": v["prob"], "signals": v["signals"]}
                                      for v in rd["verified"]]})
    return {"n_sents": tr["n_sents"], "rounds": rounds, "params": tr["params"],
            "round1": [(c["doc"], c["rare_cos"], c["quote_frac"], c["score"]) for c in
                       (tr["rounds"][0]["candidates"] if tr["rounds"] else [])]}


def midterm_rows(ctx, sus, cands):
    """The midterm verifier on the same round-1 candidates (whole-document signals and provenance)."""
    sdoc = sus.as_doc()
    rows = []
    for c in cands[:pipeline.VERIFY_TOP]:
        cd = ctx.doc(c["doc"])
        pair = signals.Pair(sdoc, cd, ctx.feats, ctx.emb, c)
        sig = signals.compute(pair)
        verdict, _ = pipeline.provenance(sdoc, cd, pair)
        rows.append({"doc": c["doc"], "prob": verify.score(sig, ctx.model), "verdict": verdict,
                     "published": pipeline.date_of(cd)})
    return rows


def midterm_scorer(ctx):
    """The midterm's whole-document verifier as the round criterion (ablation): it ignores what is explained."""
    def fn(pair, _):
        return verify.score(signals.compute(pair), ctx.model)
    return fn


def run_pass(ctx, items, model, cover, residual=True, use_facts=True, use_quotes=True, midterm=False, log_every=100,
             label="", scorer=None):
    out = {}
    t_all = time.time()
    for i, it in enumerate(items):
        sus = suspect_of(ctx, it)
        t = time.perf_counter()
        ctx.emb.doc(sus.doc_id, sus.text)
        t_emb = time.perf_counter() - t
        tr = multisource.trace(ctx, sus, model, threshold=0.0, stop_share=0.0, cover=cover,
                               exclude=exclusion(ctx, it), residual=residual, use_facts=use_facts,
                               use_quotes=use_quotes, scorer=scorer)
        rec = compact(tr)
        rec["t_embed"] = t_emb
        if midterm:
            cands = tr["rounds"][0]["candidates"] if tr["rounds"] else []
            rec["midterm"] = midterm_rows(ctx, sus, cands)
        out[it["id"]] = rec
        if (i + 1) % log_every == 0:
            ctx.emb.save()
            print(f"ms pass {label} {i + 1}/{len(items)} {time.time() - t_all:.0f}s", flush=True)
    ctx.emb.save()
    return out


def shingle_baselines(ctx, items):
    """MinHash LSH and shingle containment of each item against every allowed earlier document, in one streaming pass
    over the index's documents (the features file is too large to hold beside the index here)."""
    idx = ctx.idx
    item_sh = {}
    wanted = set()
    for it in items:
        text = it["text"] if it["n"] is None else ctx.doc(it["n"])["text"]
        item_sh[it["id"]] = dedup.shingles(text)
        wanted |= item_sh[it["id"]]
    post = defaultdict(list)
    sigs = np.zeros((idx.N, dedup.N_PERM), dtype=np.uint64)
    lsh = dedup.LSH()
    for row in ctx.con.execute("SELECT doc_id, text FROM docs"):
        n = idx.doc_num.get(row["doc_id"])
        if n is None:
            continue
        sh = dedup.shingles(row["text"])
        for s in sh & wanted:
            post[s].append(n)
        sigs[n] = dedup.minhash(sh)
        lsh.add(n, sigs[n])
    out = {"_times": {}}
    for it in items:
        allowed = candidates.mask_before(idx, it["published"] if it["n"] is None else idx.meta["published"][it["n"]],
                                         it["site"] if it["n"] is None else idx.meta["site"][it["n"]])
        allowed[it["exclude"]] = False
        if it["n"] is not None:
            allowed[it["n"]] = False
        sh = item_sh[it["id"]]
        t = time.perf_counter()
        cont = []
        lists = [np.array(post[s], dtype=np.int64) for s in sh if post.get(s)]
        if lists:
            acc = np.bincount(np.concatenate(lists), minlength=idx.N) / len(sh)
            acc = np.where(allowed, acc, 0.0)
            top = np.argsort(-acc)[:TOP_BASE]
            cont = [(int(d), float(acc[d])) for d in top if acc[d] > 0]
        t_cont = time.perf_counter() - t
        t = time.perf_counter()
        sig = dedup.minhash(sh)
        mh = sorted(((m, dedup.est_jaccard(sig, sigs[m])) for m in lsh.query(sig) if allowed[m]), key=lambda x: -x[1])
        out[it["id"]] = {"containment": cont, "minhash": mh[:TOP_BASE]}
        out["_times"][it["id"]] = {"containment": t_cont, "minhash": time.perf_counter() - t}
    return out


def predict(method, rec, base, p):
    """Predicted [(doc, share or None)] for one item under parameters p."""
    if method == "spintrace":
        fin = multisource.finalize(rec, p["threshold"], p["stop_share"], p["cover"])
        return [(s["doc"], s["share"]) for s in fin["sources"]]
    if method in ("midterm", "oneshot", "midterm_shipped"):
        vs = [v for v in rec.get("midterm", []) if v["verdict"] != "uncertain"]
        if method == "oneshot":
            return [(v["doc"], None) for v in vs if v["prob"] >= p["threshold"]][:multisource.MAX_SOURCES]
        if not vs:
            return []
        best = max(v["prob"] for v in vs)
        close = [v for v in vs if v["prob"] >= best - pipeline.TIE_MARGIN]
        src = min(close, key=lambda v: v["published"] if v["published"] is not None else float("inf"))
        return [(src["doc"], None)] if best >= p["threshold"] else []
    return [(d, None) for d, s in base[method] if s >= p["threshold"]][:multisource.MAX_SOURCES]


def counts(items, preds):
    """Per item: true positives, false positives, false negatives, exact-set match."""
    rows = []
    for it, pr in zip(items, preds):
        got = {d for d, _ in pr}
        gold = set(it["gold"])
        rows.append((len(got & gold), len(got - gold), len(gold - got), int(got == gold), len(got)))
    return np.array(rows, dtype=np.float64).reshape(-1, 5)


def f1_of(c):
    tp, fp, fn = c[:, 0].sum(), c[:, 1].sum(), c[:, 2].sum()
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def boot(items, values, fn, b=1000, seed=358):
    """95% bootstrap interval of fn over items, resampling whole groups (all levels of one mashup together)."""
    groups = defaultdict(list)
    for i, it in enumerate(items):
        groups[it["group"]].append(i)
    keys = list(groups)
    rng = np.random.RandomState(seed)
    stats = []
    for _ in range(b):
        pick = rng.randint(0, len(keys), len(keys))
        ix = [i for k in pick for i in groups[keys[k]]]
        stats.append(fn(values[ix]))
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def false_rate(items, preds):
    """Share of the no-source items (facts-only articles, hard negatives) given any source."""
    v = [len(pr) > 0 for it, pr in zip(items, preds) if it["kind"] == "none"]
    return float(np.mean(v)) if v else 0.0


def tune(method, items, recs, base, grid, cap=None):
    """Parameters maximising micro source-set F1 over all dev items (mashups and controls together).

    With cap, only parameters whose false-source rate on the dev no-source items is at most cap qualify (the
    strictest one if none does): same story is not the same as copied."""
    best, strict = (-1.0, None), (float("inf"), None)
    for p in grid:
        preds = [predict(method, recs[it["id"]], base.get(it["id"]), p) for it in items]
        f = f1_of(counts(items, preds))[2]
        if cap is not None:
            fr = false_rate(items, preds)
            if fr < strict[0]:
                strict = (fr, p)
            if fr > cap:
                continue
        if f > best[0]:
            best = (f, p)
    return (best[1], best[0]) if best[1] is not None else (strict[1], 0.0)


def shipped_cap(items, recs):
    """The shipped midterm verifier's false-source rate on the dev no-source items: the bar every method must meet."""
    thr = verify.load()["threshold"]
    return false_rate(items, [predict("midterm_shipped", recs[it["id"]], None, {"threshold": thr}) for it in items])


def score_grid(values, n=60):
    vals = np.unique(np.asarray([v for v in values if v > 0]))
    if len(vals) > n:
        vals = np.unique(np.quantile(vals, np.linspace(0, 1, n)))
    return [{"threshold": float(t)} for t in vals]


def method_grid(method, items, recs, base):
    if method == "spintrace":
        ts = np.round(np.linspace(0.05, 0.95, 19), 2).tolist() + [0.96, 0.97, 0.98, 0.99, 0.995]
        return [{"threshold": t, "stop_share": s, "cover": recs["_cover"]} for t in ts
                for s in (0.0, 0.1, 0.15, 0.2, 0.25, 0.3)]
    if method in ("midterm", "oneshot"):
        return score_grid([v["prob"] for it in items for v in recs[it["id"]].get("midterm", [])])
    return score_grid([s for it in items for _, s in base[it["id"]][method]])


def fit_round_verifier(items, recs, features=multisource.FEATURES):
    """Logistic regression over the round signals; a candidate is positive if it is a not-yet-found true source."""
    from sklearn.linear_model import LogisticRegression
    X, y = [], []
    for it in items:
        gold = set(it["gold"])
        found = set()
        for rd in recs[it["id"]]["rounds"]:
            for v in rd["verified"]:
                X.append([v["signals"][f] for f in features])
                y.append(int(v["doc"] in gold and v["doc"] not in found))
            found.add(rd["chosen"]["doc"])
    clf = LogisticRegression(C=1.0, max_iter=3000, class_weight="balanced").fit(np.array(X), np.array(y))
    return {"features": features, "weights": clf.coef_[0].tolist(), "bias": float(clf.intercept_[0]),
            "threshold": 0.5, "cover": signals.ALIGN_THRESHOLD, "stop_share": 0.2, "source": "dev split",
            "n_pairs": len(y), "n_pos": int(sum(y))}


PREFIX = "ms"


def cached(name, fn, fresh=False):
    path = CACHE / f"{PREFIX}_{name}.pkl"
    if path.exists() and not fresh:
        with path.open("rb") as f:
            return pickle.load(f)
    val = fn()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(val, f, protocol=pickle.HIGHEST_PROTOCOL)
    return val


def run(fresh=False, xl=None, ablations=True, smoke=None):
    """Every pass the report needs, cached under data/eval/ms_*.pkl so a rerun only recomputes what is missing.

    smoke=k runs on k seeded-random items per split, cached separately, to check the harness end to end.
    """
    global PREFIX
    if not mashups.OUT.exists():
        mashups.build()
    ctx = context()
    syn = rewrites_of(ctx)
    items = build_items(ctx, syn)
    if smoke:
        PREFIX = "ms_smoke"
        rng = random.Random(7)
        items = [it for sp in ("dev", "test") for it in rng.sample([x for x in items if x["split"] == sp], smoke)]
    dev = [it for it in items if it["split"] == "dev"]
    test = [it for it in items if it["split"] == "test"]
    print(f"items dev={len(dev)} test={len(test)}", flush=True)
    base = cached("base", lambda: shingle_baselines(ctx, items), fresh)
    p1 = cached("dev_default", lambda: run_pass(ctx, dev, multisource.DEFAULT, signals.ALIGN_THRESHOLD,
                                                label="dev default"), fresh)
    model = fit_round_verifier(dev, p1)
    dev_runs = {}
    for cv in COVERS:
        dev_runs[cv] = cached(f"dev_c{cv}", lambda cv=cv: run_pass(ctx, dev, model, cv, midterm=(cv == COVERS[-1]),
                                                                   label=f"dev cover {cv}"), fresh)
    cap = shipped_cap(dev, dev_runs[COVERS[-1]])
    best = (-1, None, None)
    for cv, recs in dev_runs.items():
        recs = dict(recs, _cover=cv)
        p, f = tune("spintrace", dev, recs, base, method_grid("spintrace", dev, recs, base), cap)
        print(f"dev cover={cv} best F1={f:.3f} with false-source rate <= {cap:.3f}: {p}", flush=True)
        if f > best[0]:
            best = (f, cv, p)
    _, cover, params = best
    model.update(threshold=params["threshold"], stop_share=params["stop_share"], cover=cover, false_source_cap=cap)
    multisource.MODEL_PATH.parent.mkdir(exist_ok=True)
    multisource.MODEL_PATH.write_text(json.dumps(model, indent=2))
    tst = cached(f"test_c{cover}", lambda: run_pass(ctx, test, model, cover, midterm=True, label="test"), fresh)
    out = {"items": items, "base": base, "model": model, "dev": dev_runs[cover], "dev_mid": dev_runs[COVERS[-1]],
           "test": tst, "cover": cover, "cap": cap, "ablations": {}}
    if ablations:
        for name, kw in (("no_residual", {"residual": False}), ("no_facts_quotes", {"use_facts": False, "use_quotes": False}),
                         ("midterm_verifier_in_rounds", {"scorer": midterm_scorer(ctx)})):
            d = cached(f"dev_{name}", lambda kw=kw: run_pass(ctx, dev, model, cover, label=f"dev {name}", **kw), fresh)
            t = cached(f"test_{name}", lambda kw=kw: run_pass(ctx, test, model, cover, label=f"test {name}", **kw), fresh)
            out["ablations"][name] = (d, t)
    llm = llm_items(ctx, xl, syn)
    if llm:
        key = f"llm_{len(llm)}"
        lb = cached(f"base_{key}", lambda: shingle_baselines(ctx, llm), fresh)
        lr = cached(key, lambda: run_pass(ctx, llm, model, cover, midterm=True, label="llm"), fresh)
        out["llm"] = (llm, lr, lb)
    return out


def executed(rec, p):
    """Rounds an early-stopping run with parameters p would compute: the accepted ones plus the one that failed."""
    keep = multisource.accepted_rounds(rec, p["threshold"], p["stop_share"])
    k = len(keep)
    if k < len(rec["rounds"]):
        nxt = rec["rounds"][k]
        if not (nxt["round"] > 1 and (nxt["share_left"] < p["stop_share"] or nxt["residual_sents"] < multisource.MIN_NEW)):
            k += 1
    return rec["rounds"][:k]


def write_csv(path, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})


def set_rows(method, items, preds, groups):
    rows = []
    for name, keep in groups:
        ix = [i for i, it in enumerate(items) if keep(it)]
        if not ix:
            continue
        sub = [items[i] for i in ix]
        c = counts(sub, [preds[i] for i in ix])
        p, r, f = f1_of(c)
        f_lo, f_hi = boot(sub, c, lambda v: f1_of(v)[2])
        r_lo, r_hi = boot(sub, c, lambda v: f1_of(v)[1])
        e_lo, e_hi = boot(sub, c, lambda v: v[:, 3].mean())
        rows.append({"method": method, "group": name, "n": len(sub), "precision": p, "recall": r,
                     "recall_lo": r_lo, "recall_hi": r_hi, "f1": f, "f1_lo": f_lo, "f1_hi": f_hi,
                     "exact_set": float(c[:, 3].mean()), "exact_lo": e_lo, "exact_hi": e_hi,
                     "mean_sources_predicted": float(c[:, 4].mean())})
    return rows


MASH_GROUPS = [("splice (all)", lambda it: it["kind"] == "splice")]
MASH_GROUPS += [(f"splice {lv}", lambda it, lv=lv: it["kind"] == "splice" and it["level"] == lv) for lv in mashups.ALL_LEVELS]
MASH_GROUPS += [(f"splice {v}", lambda it, v=v: it["kind"] == "splice" and it["variant"] == v) for v in mashups.VARIANTS]
MASH_GROUPS += [(f"splice {k} sources", lambda it, k=k: it["kind"] == "splice" and it["n_sources"] == k) for k in (2, 3)]
MASH_GROUPS += [(f"splice mix {m}", lambda it, m=m: it["kind"] == "splice" and it["mix"] == m)
                for m in ("50/50", "70/30", "40/30/30", "50/25/25")]
ALL_GROUP = [("all items (mashups and controls)", lambda it: True)]


def rank_rows(method, items, preds):
    rows = []
    for k in (2, 3):
        sub = [(it, pr) for it, pr in zip(items, preds) if it["kind"] == "splice" and it["n_sources"] == k]
        for rank in range(k):
            hits = np.array([float(it["gold_rank"][rank] in {d for d, _ in pr}) for it, pr in sub])
            lo, hi = boot([it for it, _ in sub], hits, lambda v: v.mean())
            rows.append({"method": method, "n_sources": k, "source_rank": rank + 1, "n": len(sub),
                         "recall": float(hits.mean()), "lo": lo, "hi": hi})
    return rows


def control_rows(method, items, preds):
    rows = []
    for level in ("light", "synonym", "seo", "summary", "facts_only", "hard_negative"):
        sub = [(it, pr) for it, pr in zip(items, preds) if it["level"] == level]
        if not sub:
            continue
        got = [{d for d, _ in pr} for _, pr in sub]
        its = [it for it, _ in sub]
        if level in ("facts_only", "hard_negative"):
            v = np.array([float(len(g) > 0) for g in got])
            lo, hi = boot(its, v, lambda x: x.mean())
            rows.append({"method": method, "level": level, "n": len(sub), "false_source_rate": float(v.mean()),
                         "lo": lo, "hi": hi})
        else:
            extra = np.array([float(len(g - set(it["gold"])) > 0) for it, g in zip(its, got)])
            exact = np.array([float(g == set(it["gold"])) for it, g in zip(its, got)])
            found = np.array([float(it["gold"][0] in g) for it, g in zip(its, got)])
            lo, hi = boot(its, extra, lambda x: x.mean())
            rows.append({"method": method, "level": level, "n": len(sub), "extra_source_rate": float(extra.mean()),
                         "lo": lo, "hi": hi, "source_found": float(found.mean()), "exactly_one_correct": float(exact.mean())})
    return rows


def share_rows(items, preds):
    rows = []
    for lv in mashups.ALL_LEVELS + ["all"]:
        sub = [(it, pr) for it, pr in zip(items, preds) if it["kind"] == "splice" and lv in ("all", it["level"])]
        err, err_found, unexp, gold_unexp = [], [], [], []
        for it, pr in sub:
            got = dict(pr)
            for g, s in it["gold_share"].items():
                err.append(abs(got.get(g, 0.0) - s))
                if g in got:
                    err_found.append(abs(got[g] - s))
            unexp.append(1 - sum(s for _, s in pr))
            gold_unexp.append(1 - sum(it["gold_share"].values()))
        rows.append({"level": lv, "items": len(sub), "gold_sources": len(err), "share_mae": float(np.mean(err)),
                     "share_mae_found_only": float(np.mean(err_found)) if err_found else None,
                     "unexplained_predicted": float(np.mean(unexp)), "unexplained_gold": float(np.mean(gold_unexp))})
    return rows


def retrieval_rows(items, recs, preds):
    """Where each true source sits in the round-1 (whole-text) candidate list, by its share rank, against how often
    the explain-away rounds find it: what the residual query adds over one query for the whole suspect."""
    rows = []
    for k in (2, 3):
        sub = [(it, pr) for it, pr in zip(items, preds) if it["kind"] == "splice" and it["n_sources"] == k]
        for rank in range(k):
            r1 = [[d for d, *_ in recs[it["id"]]["round1"]] for it, _ in sub]
            g = [it["gold_rank"][rank] for it, _ in sub]
            rows.append({"n_sources": k, "source_rank": rank + 1, "n": len(sub),
                         "round1_top5": float(np.mean([x in l[:multisource.K_VERIFY] for x, l in zip(g, r1)])),
                         "round1_top20": float(np.mean([x in l for x, l in zip(g, r1)])),
                         "found_by_rounds": float(np.mean([x in {d for d, _ in pr} for x, (_, pr) in zip(g, sub)]))})
    return rows


def cost_rows(items, recs, base_times, params):
    ms, rounds, touched, emb = [], [], [], []
    for it in items:
        rec = recs[it["id"]]
        ex = executed(rec, params)
        ms.append(1000 * sum(r["t_round"] for r in ex))
        rounds.append(len(ex))
        touched += [r["postings_touched"] for r in ex]
        emb.append(1000 * rec["t_embed"])
    rows = [{"method": "spintrace", "ms_per_suspect": float(np.mean(ms)), "rounds_per_suspect": float(np.mean(rounds)),
             "postings_touched_per_round": float(np.mean(touched)), "ms_embed_suspect": float(np.mean(emb)),
             "docs_verified_per_round": float(multisource.K_VERIFY)}]
    for m in ("containment", "minhash"):
        ts = [base_times[it["id"]][m] for it in items if it["id"] in base_times]
        if ts:
            rows.append({"method": m, "ms_per_suspect": 1000 * float(np.mean(ts))})
    return rows


def charts(rank, by_n, scatter):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .report import _style

    fig, ax = plt.subplots(figsize=(7.5, 4.6), dpi=150)
    for m in METHODS:
        xs = [1, 2, 3]
        ax.plot(xs, [by_n[m].get(x, np.nan) for x in xs], marker="o", markersize=8,
                linewidth=2.5 if m == "spintrace" else 2, color=COLORS[m], zorder=5 if m == "spintrace" else 2,
                label=LABELS[m])
    ax.set_xticks([1, 2, 3], ["1 (single-source\nrewrites)", "2 (mashups)", "3 (mashups)"])
    ax.set_ylim(-0.02, 1.02)
    _style(ax, "Source recall by number of sources mixed (test)", "sources in the suspect", "share of true sources found")
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(OUT / "multisource_recall_by_n.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.6, 5.2), dpi=150)
    marks = {"splice": "o", "splice_light": "s", "splice_synonym": "^", "splice_seo": "D"}
    tones = {"splice": "#2a78d6", "splice_light": "#eda100", "splice_synonym": "#eb6834", "splice_seo": "#1baf7a"}
    for lv in mashups.ALL_LEVELS:
        pts = [(g, p) for l, g, p in scatter if l == lv]
        if pts:
            g, p = zip(*pts)
            jit = np.random.RandomState(0).uniform(-0.008, 0.008, len(g))
            ax.scatter(np.array(g) + jit, p, s=16, marker=marks[lv], color=tones[lv], alpha=0.6, label=lv)
    ax.plot([0, 1], [0, 1], color="#b5b4ae", linewidth=1)
    ax.set_xlim(0, 0.85)
    ax.set_ylim(-0.02, 0.85)
    _style(ax, "Share of the suspect per true source (test)", "true share (sentences)", "estimated share (0 = missed)")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "multisource_share_scatter.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.4), dpi=150)
    cats = [(2, 1), (2, 2), (3, 1), (3, 2), (3, 3)]
    w = 0.16
    for j, m in enumerate(METHODS):
        ys = [next((r["recall"] for r in rank if r["method"] == m and r["n_sources"] == k and r["source_rank"] == q),
                   np.nan) for k, q in cats]
        ax.bar(np.arange(len(cats)) + (j - 2) * w, ys, width=w, color=COLORS[m], label=LABELS[m])
    ax.set_xticks(range(len(cats)), [f"{k} sources:\n{['largest', 'second', 'third'][q - 1]}" for k, q in cats])
    ax.set_ylim(0, 1.05)
    _style(ax, "Recall of each source by its share rank (test splices)", "", "recall")
    ax.legend(frameon=False, fontsize=7, loc="upper right", ncol=2)
    fig.tight_layout()
    fig.savefig(OUT / "multisource_recall_by_rank.png")
    plt.close(fig)


def evaluate(out):
    items, base, model = out["items"], out["base"], out["model"]
    dev = [it for it in items if it["split"] == "dev"]
    test = [it for it in items if it["split"] == "test"]
    cap = out["cap"]
    params = {"spintrace": {"threshold": model["threshold"], "stop_share": model["stop_share"], "cover": model["cover"]}}
    free = {}
    dev_sp = dict(out["dev"], _cover=model["cover"])
    free["spintrace"], _ = tune("spintrace", dev, dev_sp, base, method_grid("spintrace", dev, dev_sp, base))
    for m in ("midterm", "oneshot", "minhash", "containment"):
        grid = method_grid(m, dev, out["dev_mid"], base)
        params[m], _ = tune(m, dev, out["dev_mid"], base, grid, cap)
        free[m], _ = tune(m, dev, out["dev_mid"], base, grid)
    params["midterm_shipped"] = free["midterm_shipped"] = {"threshold": verify.load()["threshold"]}
    preds = {m: [predict(m, out["test"][it["id"]], base[it["id"]], params[m]) for it in test] for m in METHODS + REFERENCE}
    det, rank, ctrl, allr = [], [], [], []
    by_n = {}
    for m in METHODS + REFERENCE:
        det += set_rows(m, test, preds[m], MASH_GROUPS)
        allr += set_rows(m, test, preds[m], ALL_GROUP)
        rank += rank_rows(m, test, preds[m])
        ctrl += control_rows(m, test, preds[m])
        by_n[m] = {}
        for k in (1, 2, 3):
            sub = [(it, pr) for it, pr in zip(test, preds[m])
                   if (it["kind"] == "single" if k == 1 else it["kind"] == "splice" and it["n_sources"] == k)]
            by_n[m][k] = f1_of(counts([it for it, _ in sub], [pr for _, pr in sub]))[1]
    if out.get("llm"):
        llm, lr, lb = out["llm"]
        # The first LLM pairs can mix a dev and a test original; only pairs with both sources in test are clean.
        for subset, keep in (("LLM mashups (all)", lambda it: True),
                             ("LLM mashups (both sources test)", lambda it: it["split_all"] == ["test"])):
            sub = [it for it in llm if keep(it)]
            for m in METHODS + REFERENCE:
                pr = [predict(m, lr[it["id"]], lb[it["id"]], params[m]) for it in sub]
                det += set_rows(m, sub, pr, [(subset, lambda it: True)])
        for m in METHODS:
            pr = [predict(m, lr[it["id"]], lb[it["id"]], free[m]) for it in llm]
            det += set_rows(m, llm, pr, [("LLM mashups (all), max-F1 thresholds without the bar", lambda it: True)])
    shares = share_rows(test, preds["spintrace"])
    scatter = [(it["level"], s, dict(pr).get(g, 0.0)) for it, pr in zip(test, preds["spintrace"])
               if it["kind"] == "splice" for g, s in it["gold_share"].items()]
    abl = []
    variants = [("full", out["dev"], out["test"])] + [(k, d, t) for k, (d, t) in out["ablations"].items()]
    for name, d, t in variants:
        d = dict(d, _cover=model["cover"])
        p, f_dev = tune("spintrace", dev, d, base, method_grid("spintrace", dev, d, base), cap)
        pr = [predict("spintrace", t[it["id"]], None, p) for it in test]
        abl.append(dict({"variant": name, "dev_f1": f_dev, "threshold": p["threshold"], "stop_share": p["stop_share"]},
                        **summary_row(name, test, pr)))
    maxf1 = []
    for m in METHODS + REFERENCE:
        pr = [predict(m, out["test"][it["id"]], base[it["id"]], free[m]) for it in test]
        maxf1.append(dict({"method": m, "threshold": free[m]["threshold"]}, **summary_row(m, test, pr)))
    tradeoff_chart(test, out["test"], base, dev_sp, params)
    cost = cost_rows(test, out["test"], base.get("_times", {}), params["spintrace"])
    retr = retrieval_rows(test, out["test"], preds["spintrace"])
    OUT.mkdir(exist_ok=True)
    write_csv(OUT / "multisource_detection.csv", det + allr)
    write_csv(OUT / "multisource_by_rank.csv", rank)
    write_csv(OUT / "multisource_controls.csv", ctrl)
    write_csv(OUT / "multisource_shares.csv", shares)
    write_csv(OUT / "multisource_ablations.csv", abl)
    write_csv(OUT / "multisource_cost.csv", cost)
    write_csv(OUT / "multisource_retrieval.csv", retr)
    write_csv(OUT / "multisource_maxf1.csv", maxf1)
    sizes = defaultdict(int)
    for it in items:
        sizes[f"{it['split']} {it['kind']} {it['level']}"] += 1
    (OUT / "multisource_params.json").write_text(json.dumps(
        {"params": params, "params_max_f1": free, "false_source_cap": cap, "items": sizes,
         "llm_items": len(out["llm"][0]) if out.get("llm") else 0}, indent=2, sort_keys=True))
    charts(rank, by_n, scatter)
    return {"det": det, "all": allr, "rank": rank, "ctrl": ctrl, "shares": shares, "abl": abl, "cost": cost,
            "retr": retr, "params": params, "maxf1": maxf1, "cap": cap}


def summary_row(name, test, pr):
    """One line per variant: F1 on splices and on everything, recall of minor sources, control error rates."""
    row = {}
    for r in set_rows(name, test, pr, MASH_GROUPS[:1] + [MASH_GROUPS[4]] + ALL_GROUP):
        row[f"F1 {r['group']}"] = r["f1"]
    for r in rank_rows(name, test, pr):
        if r["source_rank"] > 1:
            row[f"recall {r['n_sources']} src rank {r['source_rank']}"] = r["recall"]
    for r in control_rows(name, test, pr):
        key = "false_source_rate" if "false_source_rate" in r else "extra_source_rate"
        row[f"{r['level']} {key}"] = r[key]
    return row


def tradeoff_chart(test, recs, base, dev_sp, params):
    """Source recall on test splices against the false-source rate on test no-source items, sweeping each method's
    threshold (a curve, like a PR curve; the dots are the dev-tuned operating points)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .report import _style
    splice = [it for it in test if it["kind"] == "splice"]
    none = [it for it in test if it["kind"] == "none"]

    def point(m, p):
        ps = [predict(m, recs[it["id"]], base[it["id"]], p) for it in splice]
        pn = [predict(m, recs[it["id"]], base[it["id"]], p) for it in none]
        return false_rate(none, pn), f1_of(counts(splice, ps))[1]

    fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
    for m in METHODS:
        grid = method_grid(m, test, dict(recs, _cover=params["spintrace"]["cover"]) if m == "spintrace" else recs, base)
        if m == "spintrace":
            grid = [g for g in grid if g["stop_share"] == params["spintrace"]["stop_share"]]
        pts = sorted(point(m, p) for p in grid)
        ax.plot([x for x, _ in pts], [y for _, y in pts], linewidth=2.5 if m == "spintrace" else 1.8, color=COLORS[m],
                label=LABELS[m], zorder=5 if m == "spintrace" else 2)
        x, y = point(m, params[m])
        ax.plot([x], [y], marker="o", markersize=8, color=COLORS[m], zorder=6)
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    _style(ax, "Sources found against false sources (test)", "false-source rate on facts-only and hard negatives",
           "recall of true sources in splices")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "multisource_tradeoff.png")
    plt.close(fig)


def _fmt(r):
    return "  ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in r.items())


def print_tables(res):
    print("\nSource-set detection (test)")
    for r in res["det"] + res["all"]:
        print(f"  {r['method']:<12} {r['group']:<34} n={r['n']:<4} P={r['precision']:.3f} R={r['recall']:.3f} "
              f"F1={r['f1']:.3f} [{r['f1_lo']:.3f}, {r['f1_hi']:.3f}] exact={r['exact_set']:.3f} "
              f"sources/item={r['mean_sources_predicted']:.2f}")
    for title, key in (("Recall by source rank (test splices)", "rank"), ("Controls (test)", "ctrl"),
                       ("Shares (SpinTrace, test splices)", "shares"), ("Ablations (tuned on dev, test numbers)", "abl"),
                       ("Where the true sources are retrieved (test splices)", "retr"), ("Cost", "cost")):
        print(f"\n{title}")
        for r in res[key]:
            print("  " + _fmt(r))
    print(f"\nparameters: {json.dumps(res['params'])}")


def _t(header, rows):
    return "\n".join(["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
                     + ["| " + " | ".join(str(x) for x in r) + " |" for r in rows])


def _ci(r, k, lo, hi):
    return f"{r[k]:.3f} [{r[lo]:.2f}, {r[hi]:.2f}]"


def markdown(res):
    """README tables, written to results/multisource_summary.md."""
    names = METHODS + REFERENCE
    lab = dict(LABELS, midterm_shipped="Midterm, shipped threshold")
    groups = ["splice (all)"] + [f"splice {lv}" for lv in mashups.ALL_LEVELS] + ["splice hard", "splice 3 sources"]
    out = [f"Operating point: every method takes its best dev source-set F1 among the thresholds that name a source "
           f"for at most {res['cap']:.1%} of the dev facts-only articles and hard negatives, the shipped midterm "
           f"verifier's own rate there (same story is not the same as copied).\n",
           "**Source-set F1 on test mashups** (micro over true sources; 95% bootstrap interval over mashups):\n"]
    rows = []
    for m in names:
        cells = [lab[m]]
        for g in groups:
            r = next((r for r in res["det"] if r["method"] == m and r["group"] == g), None)
            cells.append(_ci(r, "f1", "f1_lo", "f1_hi") if r else "-")
        rows.append(cells)
    out.append(_t(["method"] + [g.replace("splice ", "") if g != "splice (all)" else "all splices" for g in groups], rows))
    llm = [r for r in res["det"] if r["group"].startswith("LLM mashups")]
    if llm:
        out.append("\n**LLM mashups** (gemma3:4b asked to combine two originals into one article; the source set is the "
                   "label, shares unknown; never used for tuning; small n, so the intervals are wide):\n")
        out.append(_t(["method", "set", "mashups", "precision", "recall", "F1 [95% CI]", "exact set", "sources per item"],
                      [[lab[r["method"]], r["group"].replace("LLM mashups ", ""), r["n"], f"{r['precision']:.3f}",
                        f"{r['recall']:.3f}", _ci(r, "f1", "f1_lo", "f1_hi"), f"{r['exact_set']:.3f}",
                        f"{r['mean_sources_predicted']:.2f}"] for r in llm]))
    out.append("\n**Exact source set and recall by share rank** (test splices):\n")
    rows = []
    for m in names:
        r = next(r for r in res["det"] if r["method"] == m and r["group"] == "splice (all)")
        rk = {(x["n_sources"], x["source_rank"]): x for x in res["rank"] if x["method"] == m}
        rows.append([lab[m], _ci(r, "exact_set", "exact_lo", "exact_hi"), f"{r['mean_sources_predicted']:.2f}"]
                    + [_ci(rk[k], "recall", "lo", "hi") for k in ((2, 1), (2, 2), (3, 1), (3, 2), (3, 3))])
    out.append(_t(["method", "exact set", "sources per item", "2 src: largest", "2 src: second", "3 src: largest",
                   "3 src: second", "3 src: third"], rows))
    out.append("\n**Controls** (test): extra-source rate on single-source rewrites (any source besides the true one), "
               "false-source rate on facts-only articles and crawl hard negatives (any source at all):\n")
    levels = ["light", "synonym", "seo", "summary", "facts_only", "hard_negative"]
    rows = []
    for m in names:
        c = {r["level"]: r for r in res["ctrl"] if r["method"] == m}
        cells = [lab[m]]
        for lv in levels:
            r = c.get(lv)
            k = "false_source_rate" if lv in ("facts_only", "hard_negative") else "extra_source_rate"
            cells.append(_ci(r, k, "lo", "hi") if r else "-")
        cells += [f"{c[lv]['source_found']:.3f}" if lv in c else "-" for lv in ("seo", "summary")]
        rows.append(cells)
    out.append(_t(["method", "extra: light", "extra: synonym", "extra: seo", "extra: summary", "false: facts-only",
                   "false: hard neg", "source found: seo", "source found: summary"], rows))
    out.append("\n**Shares** (SpinTrace, test splices; a missed source counts as share 0):\n")
    out.append(_t(["level", "mashups", "true sources", "share MAE", "MAE, found sources", "unexplained (est.)",
                   "unexplained (true)"],
                  [[r["level"], r["items"], r["gold_sources"], f"{r['share_mae']:.3f}",
                    f"{r['share_mae_found_only']:.3f}" if r["share_mae_found_only"] is not None else "-",
                    f"{r['unexplained_predicted']:.3f}", f"{r['unexplained_gold']:.3f}"] for r in res["shares"]]))
    out.append("\n**Where the true sources are retrieved** (test splices): in the round-1 candidates of one query for "
               "the whole suspect, against found by the explain-away rounds:\n")
    out.append(_t(["sources", "share rank", "mashups", "round-1 top 5", "round-1 top 20", "found by rounds"],
                  [[r["n_sources"], r["source_rank"], r["n"], f"{r['round1_top5']:.3f}", f"{r['round1_top20']:.3f}",
                    f"{r['found_by_rounds']:.3f}"] for r in res["retr"]]))
    keys = ["F1 splice (all)", "F1 splice splice_seo", "F1 all items (mashups and controls)", "recall 2 src rank 2",
            "recall 3 src rank 3", "seo extra_source_rate", "summary extra_source_rate", "facts_only false_source_rate",
            "hard_negative false_source_rate"]
    head = ["F1 splices", "F1 splice_seo", "F1 all items", "2nd of 2", "3rd of 3", "extra: seo", "extra: summary",
            "false: facts-only", "false: hard neg"]
    out.append("\n**Ablations** (each variant's threshold and stopping share re-tuned on dev under the same false-source "
               "bar; test numbers):\n")
    out.append(_t(["variant"] + head, [[r["variant"]] + [f"{r[k]:.3f}" if k in r else "-" for k in keys]
                                       for r in res["abl"]]))
    out.append("\n**Without the false-source bar** (each method at its max dev F1 over all items; test numbers). This "
               "is the operating point a source-set F1 alone would pick; it buys recall with false sources on "
               "independent articles:\n")
    out.append(_t(["method", "threshold"] + head, [[lab[r["method"]], f"{r['threshold']:.3f}"]
                                                   + [f"{r[k]:.3f}" if k in r else "-" for k in keys]
                                                   for r in res["maxf1"]]))
    c = res["cost"][0]
    out.append(f"\n**Cost** (test): {c['ms_per_suspect']:.0f} ms per suspect for retrieval and verification over "
               f"{c['rounds_per_suspect']:.2f} rounds ({c['postings_touched_per_round']:.0f} postings touched per round, "
               f"{multisource.K_VERIFY} candidates verified per round)"
               + (f", plus {c['ms_embed_suspect']:.0f} ms to embed the suspect's sentences." if c["ms_embed_suspect"] >= 1
                  else "; the suspects' sentence embeddings came from the cache in this run."))
    p = res["params"]
    out.append(f"\nTuned on dev: round threshold {p['spintrace']['threshold']}, stop below "
               f"{p['spintrace']['stop_share']} unexplained, cover at sentence similarity {p['spintrace']['cover']}; "
               + "; ".join(f"{lab[m]} threshold {p[m]['threshold']:.3f}" for m in names if m != "spintrace") + ".")
    text = "\n".join(out) + "\n"
    (OUT / "multisource_summary.md").write_text(text)
    return text


def main(fresh=False, xl=None, ablations=True, smoke=None):
    res = evaluate(run(fresh, xl, ablations, smoke))
    print_tables(res)
    markdown(res)
    return res


def suspect_for_key(ctx, key, xl=None):
    """(suspect, {true source: share or None} or None, exclusion mask, note) for a mashup id, doc_id or URL.

    Synthetic rewrites of the true sources are hidden, exactly as in the evaluation, so a copy of an original
    cannot stand in for it."""
    syn = rewrites_of(ctx)
    idx = ctx.idx
    excl = np.zeros(idx.N, dtype=bool)
    if key.startswith("mash:"):
        m = next((r for r in mashups.load() if r["id"] == key), None)
        if m is None:
            raise SystemExit(f"no mashup {key!r} in {mashups.OUT}")
        gold = {idx.doc_num[s["doc_id"]]: s["share"] for s in m["sources"]}
        note = f"  [{m['level']}, {m['variant']}, mix {m['mix']}; rewrites of its sources hidden]"
        sus = multisource.Suspect(m["id"], m["text"], m["published"], m["farm"])
    elif key.startswith("mashllm:"):
        rows = _xl_rows(Path(xl)) if xl else xl_rewrites.load(levels={"mash_llm"})
        r = next((r for r in rows if f"mashllm:{r['rid']}" == key), None)
        if r is None:
            raise SystemExit(f"no LLM mashup {key!r}")
        gold = {idx.doc_num[store.doc_id_for(u)]: None for u in r["orig_urls"]}
        note = "  [LLM mashup; shares unknown; rewrites of its sources hidden]"
        sus = multisource.Suspect(key, r["text"], r["published"], r["farm"])
    else:
        row = ctx.con.execute("SELECT doc_id, origin, meta FROM docs WHERE doc_id=? OR url=?", (key, key)).fetchone()
        if not row or row["doc_id"] not in idx.doc_num:
            raise SystemExit(f"no indexed document matches {key!r}")
        sus = multisource.Suspect.from_index(ctx, idx.doc_num[row["doc_id"]])
        gold, note = None, ""
        if row["origin"] == "synthetic":
            orig = json.loads(row["meta"])["orig_doc"]
            gold = {idx.doc_num[orig]: None} if json.loads(row["meta"])["level"] != "facts_only" else {}
            note = f"  [synthetic {json.loads(row['meta'])['level']}; other rewrites of its original hidden]"
            excl[syn.get(orig, [])] = True
        return sus, gold, excl, note
    for g in gold:
        excl[syn.get(idx.doc_ids[g], [])] = True
    return sus, gold, excl, note


LIVE = config.ROOT / "evaldata" / "live_multisource.csv"


def source_overlap(ctx, sources):
    """Largest shingle containment between two found sources: near 1 means they are copies of one text (syndicated
    wire copy), so the 'second source' is the same original printed twice."""
    best = 0.0
    sh = [ctx.feats[s["doc_id"]]["shingles"] for s in sources]
    for i in range(len(sh)):
        for j in range(len(sh)):
            if i != j:
                best = max(best, dedup.containment(sh[i], sh[j]))
    return round(best, 3)


def live(sample=300, xl=None, seed=358, examples=5, unflagged=300):
    """The multi-source pass over crawled articles the single-source scan flagged, plus a seeded sample of crawled
    articles it did not flag (a mashup's whole-document score can stay under the midterm threshold). Unjudged:
    counts and seeded examples only."""
    ctx = context()
    edges = {}
    for r in ctx.con.execute("""SELECT e.suspect, e.source, e.prob FROM edges e JOIN docs s ON s.doc_id=e.suspect
                                WHERE s.origin='crawl' ORDER BY e.suspect, e.prob DESC"""):
        if r["suspect"] in ctx.idx.doc_num and r["suspect"] not in edges:
            edges[r["suspect"]] = (r["source"], r["prob"])
    ids = sorted(edges)
    pick = random.Random(seed).sample(ids, min(sample, len(ids))) if sample else ids
    crawl = sorted(d for d, o in zip(ctx.idx.doc_ids, ctx.idx.meta["origin"]) if o == "crawl" and d not in edges)
    rest = random.Random(seed).sample(crawl, min(unflagged, len(crawl)))
    print(f"flagged crawled articles in the index: {len(ids)}; tracing {len(pick)} flagged and {len(rest)} unflagged",
          flush=True)
    rows, t0 = [], time.time()
    for i, doc_id in enumerate(pick + rest):
        n = ctx.idx.doc_num[doc_id]
        tr = multisource.trace(ctx, multisource.Suspect.from_index(ctx, n))
        fin = multisource.finalize(tr)
        d = ctx.doc(n)
        src, prob = edges.get(doc_id, ("", None))
        row = {"suspect": doc_id, "suspect_url": d["url"], "suspect_site": d["site"], "n_sentences": fin["n_sents"],
               "flagged_by_midterm": doc_id in edges, "midterm_source": src,
               "midterm_source_url": ctx.doc(ctx.idx.doc_num[src])["url"] if src in ctx.idx.doc_num else "",
               "midterm_prob": round(prob, 3) if prob is not None else "", "n_sources": len(fin["sources"]),
               "first_source_is_midterm_source": bool(src and fin["sources"] and fin["sources"][0]["doc_id"] == src),
               "unexplained": round(fin["unexplained"], 3), "max_source_overlap": source_overlap(ctx, fin["sources"])}
        for j, s in enumerate(fin["sources"][:multisource.MAX_SOURCES], 1):
            row.update({f"source{j}": s["doc_id"], f"source{j}_url": s["url"], f"source{j}_site": s["site"],
                        f"source{j}_share": round(s["share"], 3), f"source{j}_prob": round(s["prob"], 3)})
        rows.append(row)
        if (i + 1) % 50 == 0:
            ctx.emb.save()
            print(f"live {i + 1}/{len(pick) + len(rest)} {time.time() - t0:.0f}s", flush=True)
    ctx.emb.save()
    keys = ["suspect", "suspect_url", "suspect_site", "n_sentences", "flagged_by_midterm", "midterm_source",
            "midterm_source_url", "midterm_prob", "n_sources", "first_source_is_midterm_source", "unexplained",
            "max_source_overlap"]
    keys += [f"source{j}{k}" for j in range(1, multisource.MAX_SOURCES + 1)
             for k in ("", "_url", "_site", "_share", "_prob")]
    with LIVE.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    summary = []
    for name, keep in (("flagged", True), ("unflagged sample", False)):
        part = [r for r in rows if r["flagged_by_midterm"] == keep]
        by_n = defaultdict(int)
        for r in part:
            by_n[r["n_sources"]] += 1
        summary.append({"set": name, "measure": "articles traced", "value": len(part)})
        summary += [{"set": name, "measure": f"with {k} verified sources", "value": by_n[k]} for k in sorted(by_n)]
        many = sum(v for k, v in by_n.items() if k >= 2)
        summary += [{"set": name, "measure": "with >= 2 verified sources", "value": many},
                    {"set": name, "measure": "share with >= 2 verified sources", "value": round(many / max(1, len(part)), 3)},
                    {"set": name, "measure": "median unexplained share",
                     "value": round(float(np.median([r["unexplained"] for r in part])), 3) if part else ""}]
        if keep and part:
            summary.append({"set": name, "measure": "first source equals the midterm source",
                            "value": round(float(np.mean([r["first_source_is_midterm_source"] for r in part])), 3)})
    multi = [r for r in rows if r["n_sources"] >= 2]
    summary.append({"set": "all", "measure": "multi-source articles whose sources share >= 50% of shingles (one text "
                    "printed twice)", "value": sum(r["max_source_overlap"] >= 0.5 for r in multi)})
    pairs = defaultdict(int)
    for r in multi:
        pairs[" + ".join(sorted({r[f"source{j}_site"] for j in range(1, r["n_sources"] + 1)}))] += 1
    summary += [{"set": "all", "measure": f"source sites: {k}", "value": v}
                for k, v in sorted(pairs.items(), key=lambda x: -x[1])[:8]]
    write_csv(OUT / "multisource_live.csv", summary)
    for s in summary:
        print(f"  {s['set']}: {s['measure']}: {s['value']}")
    print(f"\n{examples} seeded-random articles with >= 2 verified sources (unjudged):")
    for r in random.Random(seed + 1).sample(multi, min(examples, len(multi))):
        print(f"  {r['suspect_url']}  flagged by midterm={r['flagged_by_midterm']}  unexplained={r['unexplained']}")
        for j in range(1, r["n_sources"] + 1):
            print(f"     {r[f'source{j}_share']:.0%} p={r[f'source{j}_prob']} {r[f'source{j}_url']}")
    print(f"written to {LIVE} and {OUT / 'multisource_live.csv'}")
    return rows
