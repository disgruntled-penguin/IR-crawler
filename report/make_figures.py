"""Redraw the report's charts from results/*.csv in the report palette: python report/make_figures.py

Reads only; nothing in results/ is written. The PR curve has no CSV, so it is drawn from the cached
evaluation scores (data/eval/raw.pkl) with the saved dev-fit verifier (results/verifier.json), not refit.
"""
import csv
import sys
from pathlib import Path

import numpy as np

import plotstyle as ps
from plotstyle import plt

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
LEVELS = ["exact", "light", "synonym", "seo", "summary", "translation"]


def rows(name):
    with (RESULTS / name).open() as f:
        return list(csv.DictReader(f))


def end_labels(ax, ends, x, gap):
    """Labels at the right end of each line, pushed apart where lines finish close together."""
    ends = sorted(ends, key=lambda e: e[0])
    placed = []
    for y, label, color in ends:
        p = max(y, placed[-1] + gap) if placed else y
        placed.append(p)
        ax.annotate(label, (x, y), xytext=(x + 0.12, p), textcoords="data", va="center", fontsize=7.5, color=color,
                    annotation_clip=False)


def f1_by_level():
    det = rows("detection_by_level.csv")
    fig, ax = plt.subplots(figsize=(ps.TEXT_WIDTH_IN * 0.98, 2.45))
    xs = np.arange(len(LEVELS))
    ends = []
    for key, label, style in ps.METHODS:
        ys = [float(next(r["f1"] for r in det if r["method"] == key and r["level"] == l)) for l in LEVELS]
        ax.plot(xs, ys, **style, label=label, clip_on=False)
        ends.append((ys[-1], label, ps.OXBLOOD if key == "spintrace" else ps.INK))
    end_labels(ax, ends, xs[-1], gap=0.06)
    ax.set_xticks(xs, LEVELS)
    ax.set_xlim(-0.15, len(LEVELS) - 0.85)
    ax.set_ylim(0, 1.02)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xlabel("rewrite level, easiest to hardest")
    ax.set_ylabel("F1")
    fig.savefig(ps.FIG_DIR / "detection_f1_by_level.pdf")
    plt.close(fig)


def index_elimination():
    qb = {r["query"]: r for r in rows("query_budget.csv") if r["level"] == "all"}
    fig, ax = plt.subplots(figsize=(3.05, 2.35))
    rare = [qb[k] for k in ("rare3", "rare5", "rare10", "rare30")]
    ax.plot([float(r["postings_touched"]) for r in rare], [float(r["recall@1"]) for r in rare], color=ps.OXBLOOD,
            marker="o", lw=1.8, ms=4.5, label="m highest-idf terms", zorder=5)
    for r in rare:
        above = r["query"] == "rare30"
        ax.annotate(f"m={r['query'][4:]}", (float(r["postings_touched"]), float(r["recall@1"])),
                    xytext=(-8, 6) if above else (4, -10), textcoords="offset points", fontsize=7, color=ps.OXBLOOD,
                    ha="right" if above else "left")
    for key, color, marker, label in (("common10", ps.OCHRE, "s", "10 lowest-idf terms"),
                                      ("random10", ps.OLIVE, "^", "10 random terms")):
        r = qb[key]
        ax.plot([float(r["postings_touched"])], [float(r["recall@1"])], marker=marker, ms=5, ls="none", color=color,
                label=label)
    ax.set_xscale("log")
    ax.set_ylim(0, 1.03)
    ax.set_xlabel("postings touched per query (log)")
    ax.set_ylabel("Recall@1 of the true source")
    ax.legend(fontsize=7, loc="lower right", handletextpad=0.3)
    fig.savefig(ps.FIG_DIR / "index_elimination.pdf")
    plt.close(fig)


def pr_curve():
    raw = ROOT / "data" / "eval" / "raw.pkl"
    if not raw.exists():
        print("skip pr_curve: no", raw)
        return
    sys.path.insert(0, str(ROOT))
    from sklearn.metrics import precision_recall_curve
    from spintrace.detect import verify
    from spintrace.eval import report

    data = report.load(raw)
    test = [it for it in data["items"] if it["split"] == "test"]
    methods = {m.name: m for m in report.methods_for(verify.load())}
    labels = [it["label"] for it in test]
    fig, ax = plt.subplots(figsize=(3.6, 3.0))
    for key, label, style in reversed(ps.METHODS):
        p, r, _ = precision_recall_curve(labels, [methods[key].score_fn(it)[0] for it in test])
        style = {k: v for k, v in style.items() if k not in ("marker", "ms", "mfc")}
        ax.plot(r, p, **style, label=label)
    ax.set_xlim(0, 1.01)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    h, l = ax.get_legend_handles_labels()
    ax.legend(h[::-1], l[::-1], fontsize=7, loc="lower left")
    fig.savefig(ps.FIG_DIR / "pr_curve.pdf")
    plt.close(fig)


if __name__ == "__main__":
    ps.setup()
    f1_by_level()
    index_elimination()
    pr_curve()
    print("figures in", ps.FIG_DIR)
