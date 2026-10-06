"""Metrics, verifier fitting, ablations and charts from the cached evaluation run.

Everything that is tuned (verifier weights, every method's threshold) is fit on the dev split;
every reported number comes from the test split.
"""
import csv
import json
import pickle

import numpy as np

from .. import config
from ..detect import pipeline, verify
from . import run as runner

LEVELS = ["exact", "light", "synonym", "seo", "summary"]
NEG_LEVELS = ["original", "hard_negative"]
BASELINES = ["exact_hash", "minhash_jaccard", "shingle_containment", "tfidf_cosine", "bm25", "dense_cosine"]
FEATURES = ["rare_cos", "shingle_containment", "align_coverage", "align_mean", "align_order", "quote_overlap",
            "fact_overlap"]
LABELS = {"spintrace": "SpinTrace", "exact_hash": "Exact hash", "minhash_jaccard": "MinHash Jaccard",
          "shingle_containment": "Shingle containment (all pairs)", "tfidf_cosine": "tf-idf cosine", "bm25": "BM25",
          "dense_cosine": "Dense cosine"}
COLORS = {"spintrace": "#2a78d6", "exact_hash": "#eb6834", "minhash_jaccard": "#1baf7a", "shingle_containment": "#008300",
          "tfidf_cosine": "#eda100", "bm25": "#e87ba4", "dense_cosine": "#4a3aa7"}
OUT = config.RESULTS


def load(raw=runner.RAW):
    with raw.open("rb") as f:
        data = pickle.load(f)
    for it in data["items"]:
        if "verified" not in it:
            it["verified"] = it["spintrace"].pop("verified")
    return data


def pair_rows(items, key="verified", features=FEATURES):
    X, y = [], []
    for it in items:
        for v in it.get(key) or []:
            X.append([v["signals"].get(f, 0.0) for f in features])
            y.append(int(it["label"] == 1 and v["doc"] == it["source"]))
    return np.array(X), np.array(y)


def spin_score(it, model, key="verified"):
    """Suspect score = best verified probability; predicted source = earliest near-tied candidate."""
    vs = [v for v in (it.get(key) or []) if v["verdict"] != "uncertain"]
    if not vs:
        return 0.0, None
    probs = [verify.score(v["signals"], model) for v in vs]
    best = max(probs)
    close = [v for v, p in zip(vs, probs) if p >= best - pipeline.TIE_MARGIN]
    src = min(close, key=lambda v: v["published"] if v["published"] is not None else float("inf"))
    return best, src["doc"]


def spin_ranking(it, model, key="verified"):
    vs = it.get(key) or []
    order = [v["doc"] for v in sorted(vs, key=lambda v: -verify.score(v["signals"], model))]
    rest = [d for d, _ in it["spintrace"]["cands"] if d not in order]
    return order + rest


def baseline_score(it, m):
    r = it["baselines"][m][0]
    return (r[0][1], r[0][0]) if r else (0.0, None)


def best_threshold(scores, labels):
    """Threshold maximising F1 on the given (dev) scores."""
    scores, labels = np.asarray(scores), np.asarray(labels)
    best = (0.0, float("inf"))
    for t in np.unique(scores):
        if t <= 0:
            continue
        pred = scores >= t
        tp = int((pred & (labels == 1)).sum())
        fp = int((pred & (labels == 0)).sum())
        fn = int((~pred & (labels == 1)).sum())
        f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
        if f1 > best[0]:
            best = (f1, float(t))
    return best[1]


def prf(pred, labels):
    pred, labels = np.asarray(pred, dtype=bool), np.asarray(labels)
    tp = int((pred & (labels == 1)).sum())
    fp = int((pred & (labels == 0)).sum())
    fn = int((~pred & (labels == 1)).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


class Method:
    """A scorer over items: returns (score, predicted source) and a ranked candidate list."""

    def __init__(self, name, score_fn, rank_fn):
        self.name, self.score_fn, self.rank_fn = name, score_fn, rank_fn


def methods_for(model):
    ms = [Method("spintrace", lambda it: spin_score(it, model), lambda it: spin_ranking(it, model))]
    for b in BASELINES:
        ms.append(Method(b, lambda it, b=b: baseline_score(it, b), lambda it, b=b: [d for d, _ in it["baselines"][b][0]]))
    return ms


def fit_model(dev, features=FEATURES, key="verified"):
    X, y = pair_rows(dev, key, features)

    def thr(model):
        return best_threshold([spin_score(it, model, key)[0] for it in dev], [it["label"] for it in dev])

    return verify.fit(X, y, features, thr)


def detection_table(method, dev, test):
    thr = best_threshold([method.score_fn(it)[0] for it in dev], [it["label"] for it in dev])
    scored = [(it, *method.score_fn(it)) for it in test]
    negs = [(it, s, src) for it, s, src in scored if it["label"] == 0]
    rows = []
    for level in LEVELS:
        pos = [(it, s, src) for it, s, src in scored if it["level"] == level]
        if not pos:
            continue
        group = pos + negs
        p, r, f = prf([s >= thr for _, s, _ in group], [it["label"] for it, _, _ in group])
        src_acc = np.mean([s >= thr and src == it["source"] for it, s, src in pos])
        rows.append({"method": method.name, "level": level, "n_pos": len(pos), "n_neg": len(negs), "threshold": thr,
                     "precision": p, "recall": r, "f1": f, "source_correct": float(src_acc)})
    for level in NEG_LEVELS:
        ns = [(it, s) for it, s, _ in negs if it["level"] == level]
        if ns:
            rows.append({"method": method.name, "level": f"fpr_{level}", "n_pos": 0, "n_neg": len(ns),
                         "threshold": thr, "precision": None, "recall": None,
                         "f1": float(np.mean([s >= thr for _, s in ns])), "source_correct": None})
    return rows


def ranking_table(method, test, ks=(1, 5, 20)):
    rows = []
    pos = [it for it in test if it["label"] == 1]
    for level in LEVELS + ["all"]:
        group = [it for it in pos if level == "all" or it["level"] == level]
        if not group:
            continue
        ranks = []
        for it in group:
            lst = method.rank_fn(it)
            ranks.append(lst.index(it["source"]) + 1 if it["source"] in lst else None)
        row = {"method": method.name, "level": level, "n": len(group),
               "mrr": float(np.mean([1 / r if r else 0 for r in ranks]))}
        for k in ks:
            row[f"recall@{k}"] = float(np.mean([bool(r and r <= k) for r in ranks]))
        rows.append(row)
    return rows


def write_csv(path, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})


def macro_f1(rows):
    fs = [r["f1"] for r in rows if r["level"] in LEVELS]
    return float(np.mean(fs)) if fs else 0.0


def ablations(dev, test, full_model):
    out = []

    def add(name, method, dev_items=dev, test_items=test):
        rows = detection_table(method, dev_items, test_items)
        row = {"variant": name, "macro_f1": macro_f1(rows)}
        for r in rows:
            if r["level"] in LEVELS:
                row[r["level"]] = r["f1"]
            elif r["level"] == "fpr_hard_negative":
                row["fpr_hard_negative"] = r["f1"]
        out.append(row)

    add("full", Method("spintrace", lambda it: spin_score(it, full_model), None))
    for f in FEATURES:
        feats = [x for x in FEATURES if x != f]
        m = fit_model(dev, feats)
        add(f"minus {f}", Method("spintrace", lambda it, m=m: spin_score(it, m), None))
    ir_feats = ["rare_cos", "shingle_containment", "quote_overlap", "fact_overlap"]
    m = fit_model(dev, ir_feats)
    add("no embedding signals (IR signals only)", Method("x", lambda it, m=m: spin_score(it, m), None))
    dense_feats = [x for x in FEATURES if x != "rare_cos"]
    m = fit_model(dev, dense_feats, key="dense_verified")
    add("dense candidates instead of IR", Method("x", lambda it, m=m: spin_score(it, m, "dense_verified"), None))
    add("IR retrieval only, no verification",
        Method("x", lambda it: (it["spintrace"]["cands"][0][1], it["spintrace"]["cands"][0][0])
               if it["spintrace"]["cands"] else (0.0, None), None))
    return out


def retrieval_variants(test):
    """Candidate recall of the retrieval stage under variants of the query."""
    rows = []
    pos = [it for it in test if it["label"] == 1]
    variants = {
        "rare terms + quote phrases": lambda it: [d for d, _ in it["spintrace"]["cands"]],
        "rare terms only (no quotes)": lambda it: it["no_quotes_cands"],
        "no fact boost": lambda it: it.get("no_facts_cands"),
        "rare terms, Porter stemming": lambda it: it.get("stem_cands"),
        "dense retrieval": lambda it: [d for d, _ in it["baselines"]["dense_cosine"][0]],
        "full-document tf-idf": lambda it: [d for d, _ in it["baselines"]["tfidf_cosine"][0]],
    }
    for name, fn in variants.items():
        for level in LEVELS + ["all"]:
            group = [it for it in pos if level == "all" or it["level"] == level]
            lists = [fn(it) for it in group]
            if not group or any(l is None for l in lists):
                continue
            rows.append({"variant": name, "level": level, "n": len(group),
                         "recall@1": float(np.mean([bool(l) and l[0] == it["source"] for it, l in zip(group, lists)])),
                         "recall@5": float(np.mean([it["source"] in l[:5] for it, l in zip(group, lists)])),
                         "recall@20": float(np.mean([it["source"] in l for it, l in zip(group, lists)]))})
    return rows


def undated_provenance(test):
    """With dates removed, how often does containment asymmetry name the right direction?"""
    rows = []
    for level in LEVELS + ["hard_negative"]:
        group = [it for it in test if it["level"] == level and it.get("direction")]
        if not group:
            continue
        margin = 0.2
        derived = [f - b > margin for f, b in (it["direction"] for it in group)]
        reverse = [b - f > margin for f, b in (it["direction"] for it in group)]
        rows.append({"level": level, "n": len(group), "suspect_called_derived": float(np.mean(derived)),
                     "direction_reversed": float(np.mean(reverse)),
                     "uncertain": float(1 - np.mean(derived) - np.mean(reverse))})
    return rows


def efficiency(items):
    rows = []
    allowed = np.mean([it["allowed"] for it in items])
    sp = [it["spintrace"] for it in items]
    rows.append({"method": "spintrace", "ms_per_suspect": 1000 * np.mean([s["t_retrieve"] + s["t_verify"] for s in sp]),
                 "ms_retrieve": 1000 * np.mean([s["t_retrieve"] for s in sp]),
                 "docs_compared": float(np.mean([len(s["cands"]) for s in sp])),
                 "postings_touched": float(np.mean([s["touched"] for s in sp])), "earlier_docs": float(allowed)})
    if all("touched_full" in it for it in items):
        rows.append({"method": "full-document query (tf-idf/BM25)", "postings_touched":
                     float(np.mean([it["touched_full"] for it in items])), "earlier_docs": float(allowed)})
    for b in BASELINES:
        rows.append({"method": b, "ms_per_suspect": 1000 * np.mean([it["baselines"][b][1] for it in items]),
                     "docs_compared": float(allowed) if b not in ("exact_hash", "minhash_jaccard") else None,
                     "earlier_docs": float(allowed)})
    return rows


def _style(ax, title, xlabel, ylabel):
    ax.set_title(title, loc="left", fontsize=12, color="#0b0b0b")
    ax.set_xlabel(xlabel, color="#52514e")
    ax.set_ylabel(ylabel, color="#52514e")
    ax.grid(axis="y", color="#e4e3df", linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#b5b4ae")
    ax.tick_params(colors="#52514e")


def charts(det_rows, pr_data, rank_rows, abl_rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=150)
    levels = [l for l in LEVELS if any(r["level"] == l for r in det_rows)]
    for m in ["spintrace"] + BASELINES:
        ys = [next((r["f1"] for r in det_rows if r["method"] == m and r["level"] == l), np.nan) for l in levels]
        ax.plot(levels, ys, marker="o", markersize=8, linewidth=2.5 if m == "spintrace" else 2, color=COLORS[m],
                label=LABELS[m])
        ax.annotate(LABELS[m], (len(levels) - 1, ys[-1]), xytext=(8, 0), textcoords="offset points",
                    va="center", fontsize=8, color="#52514e")
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlim(-0.2, len(levels) - 0.4)
    _style(ax, "Detection F1 by rewrite intensity (test split)", "rewrite level, easiest to hardest", "F1")
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(OUT / "detection_f1_by_level.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
    for m, (p, r) in pr_data.items():
        ax.plot(r, p, linewidth=2.5 if m == "spintrace" else 2, color=COLORS[m], label=LABELS[m])
    ax.set_xlim(0, 1.01)
    ax.set_ylim(0, 1.02)
    _style(ax, "Precision-recall, all levels pooled (test)", "recall", "precision")
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(OUT / "pr_curve.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=150)
    for m in ["spintrace"] + BASELINES:
        ys = [next((r["recall@1"] for r in rank_rows if r["method"] == m and r["level"] == l), np.nan) for l in levels]
        ax.plot(levels, ys, marker="o", markersize=8, linewidth=2.5 if m == "spintrace" else 2, color=COLORS[m],
                label=LABELS[m])
    ax.set_ylim(-0.02, 1.02)
    _style(ax, "Original ranked first (P@1) by rewrite level (test)", "rewrite level", "P@1")
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(OUT / "p_at_1_by_level.png")
    plt.close(fig)

    full = next(r["macro_f1"] for r in abl_rows if r["variant"] == "full")
    rows = [r for r in abl_rows if r["variant"] != "full"]
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(rows) + 1.2), dpi=150)
    deltas = [r["macro_f1"] - full for r in rows]
    ax.barh([r["variant"] for r in rows], deltas, color="#2a78d6", height=0.6)
    for i, d in enumerate(deltas):
        ax.annotate(f"{d:+.3f}", (d, i), xytext=(-4 if d < 0 else 4, 0), textcoords="offset points",
                    ha="right" if d < 0 else "left", va="center", fontsize=8, color="#52514e")
    ax.axvline(0, color="#b5b4ae", linewidth=1)
    ax.invert_yaxis()
    _style(ax, f"Ablations: change in macro F1 (full = {full:.3f})", "change in macro F1 over levels", "")
    ax.grid(axis="x", color="#e4e3df", linewidth=0.8)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(OUT / "ablations.png")
    plt.close(fig)


def main(raw=runner.RAW):
    from sklearn.metrics import precision_recall_curve
    data = load(raw)
    items = data["items"]
    dev = [it for it in items if it["split"] == "dev"]
    test = [it for it in items if it["split"] == "test"]
    model = fit_model(dev)
    verify.save(model)
    OUT.mkdir(exist_ok=True)
    det, rank, pr = [], [], {}
    for m in methods_for(model):
        det += detection_table(m, dev, test)
        rank += ranking_table(m, test)
        s = [m.score_fn(it)[0] for it in test]
        p, r, _ = precision_recall_curve([it["label"] for it in test], s)
        pr[m.name] = (p, r)
    abl = ablations(dev, test, model)
    ret = retrieval_variants(test)
    eff = efficiency(test)
    write_csv(OUT / "undated_provenance.csv", undated_provenance(test))
    write_csv(OUT / "detection_by_level.csv", det)
    write_csv(OUT / "ranking.csv", rank)
    write_csv(OUT / "ablations.csv", abl)
    write_csv(OUT / "retrieval_variants.csv", ret)
    write_csv(OUT / "efficiency.csv", eff)
    counts = {}
    for it in items:
        counts.setdefault(it["split"], {}).setdefault(it["level"], 0)
        counts[it["split"]][it["level"]] += 1
    (OUT / "eval_sets.json").write_text(json.dumps({"counts": counts, "index_docs": data["meta"]["N"]}, indent=2))
    charts(det, pr, rank, abl)
    return det, rank, abl, ret, eff, counts, model
