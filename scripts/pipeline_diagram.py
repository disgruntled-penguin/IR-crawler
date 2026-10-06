"""Draw the SpinTrace pipeline diagram for the report: python scripts/pipeline_diagram.py -> results/pipeline.png"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

INK, MUTED, LINE = "#0b0b0b", "#52514e", "#b5b4ae"
STAGE = {"crawl": "#e8f0fb", "index": "#e8f0fb", "detect": "#dbe8f8", "use": "#e8f0fb", "data": "#f3f2ee"}

BOXES = [
    ("data", 0.2, 6.2, 2.6, 1.0, "Seeds", "RSS, Atom, news sitemaps\n(36 outlets, robots-checked)"),
    ("crawl", 3.4, 6.2, 3.6, 1.0, "1  Polite crawl", "Mercator frontier, robots.txt,\n>= 5 s per host, backoff, trap guards"),
    ("data", 7.6, 6.2, 2.6, 1.0, "Wikinews dump", "400 originals, 4,000 distractors\n+ graded rewrites"),
    ("index", 3.4, 4.5, 3.6, 1.2, "2  Store and index",
     "extract -> tokenise -> positional index\n(title, body, quote zones), tf-idf, BM25\ncontent-seen: hash, shingles, MinHash, LSH"),
    ("detect", 0.2, 2.4, 3.2, 1.4, "3a  Candidates (IR)",
     "30 rarest terms, names x2 -> lnc.ltc\nquote windows -> phrase queries\nonly other sites, published earlier"),
    ("detect", 3.8, 2.4, 3.2, 1.4, "3b  Verify (top 5)",
     "sentence alignment, order,\nback coverage, quotes, facts,\nshingle containment -> logistic p"),
    ("detect", 7.4, 2.4, 2.8, 1.4, "3c  Provenance",
     "earlier date, or coverage\nasymmetry when undated\n-> copy graph"),
    ("use", 1.4, 0.4, 3.6, 1.2, "4a  Search", "net = relevance + lambda g(d)\ng(d) = 1 - p(source edge)"),
    ("use", 5.6, 0.4, 3.6, 1.2, "4b  Evaluation and reports",
     "baselines, ablations, judged pairs,\nper-domain integrity, crawl audit"),
]
ARROWS = [((2.8, 6.7), (3.4, 6.7)), ((5.2, 6.2), (5.2, 5.7)), ((7.6, 6.7), (7.0, 5.1)), ((4.2, 4.5), (2.2, 3.8)),
          ((3.4, 3.1), (3.8, 3.1)), ((7.0, 3.1), (7.4, 3.1)), ((8.8, 2.4), (7.4, 1.6)), ((5.4, 2.4), (3.6, 1.6))]


def main():
    fig, ax = plt.subplots(figsize=(10.6, 7.8), dpi=160)
    ax.set_xlim(0, 10.4)
    ax.set_ylim(0, 7.9)
    ax.axis("off")
    for kind, x, y, w, h, title, body in BOXES:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=STAGE[kind],
                                    ec=LINE, lw=1))
        ax.text(x + 0.12, y + h - 0.12, title, ha="left", va="top", fontsize=10, weight="bold", color=INK)
        ax.text(x + 0.12, y + h - 0.42, body, ha="left", va="top", fontsize=8, color=MUTED, linespacing=1.35)
    for a, b in ARROWS:
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=12, color=MUTED, lw=1.2))
    ax.text(0.2, 7.85, "SpinTrace: IR finds the candidate sources, verification decides, provenance orders them",
            fontsize=11, color=INK, va="top")
    out = Path(__file__).resolve().parent.parent / "results" / "pipeline.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    print(out)


if __name__ == "__main__":
    main()
