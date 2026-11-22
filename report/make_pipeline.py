"""Draws the final-phase pipeline (figures/pipeline_final.pdf): the mid-term detection path plus the cross-language
and explain-away extensions. Run from the repo root: python3 report/make_pipeline.py"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).parent / "figures"
INK, OX, OCHRE, GREY = "#1E1C1A", "#7A2E2E", "#B5832A", "#8A837A"
PAPER, ROW = "#F3EFE7", "#F5ECE9"


def box(ax, x, y, w, h, title, body="", kind="crawl"):
    face, edge = (ROW, OX) if kind == "detect" else (("#F4F1E4", OCHRE) if kind == "new" else (PAPER, GREY))
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06", fc=face, ec=edge, lw=1.1))
    ax.text(x + w / 2, y + h - 0.16, title, ha="center", va="top", fontsize=7.6, fontweight="bold", color=INK)
    if body:
        ax.text(x + w / 2, y + h - 0.44, body, ha="center", va="top", fontsize=6.3, color=INK, linespacing=1.25)


def arrow(ax, p, q, style="-|>", color=INK, rad=0.0, lw=1.0, ls="-"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=9, color=color, lw=lw, ls=ls,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0))


def main():
    fig, ax = plt.subplots(figsize=(10.2, 4.55))
    ax.set_xlim(0, 10.2)
    ax.set_ylim(0, 4.55)
    ax.axis("off")

    ax.text(0.05, 4.38, "Crawl and index (runs continuously)", fontsize=7.4, color=GREY, fontstyle="italic")
    box(ax, 0.05, 3.25, 2.5, 1.0, "Polite crawl", "Mercator front and back queues\nrobots.txt and Crawl-delay, 5 s floor\nURL normalisation, trap guards")
    box(ax, 3.0, 3.25, 2.5, 1.0, "Store and content-seen check", "main text, publish date, script tag\nexact hash, word 5-gram shingles,\nMinHash and LSH")
    box(ax, 5.95, 3.25, 2.5, 1.0, "Positional inverted index", "title, body and quote zones\nidf, lnc.ltc and BM25\nname k-gram index (phonetic keys)")
    arrow(ax, (2.55, 3.75), (3.0, 3.75))
    arrow(ax, (5.5, 3.75), (5.95, 3.75))

    ax.text(0.05, 2.95, "Detection of a suspect page (English or Indian language)", fontsize=7.4, color=GREY, fontstyle="italic")
    w, h, y = 1.55, 1.7, 1.15
    xs = [0.05, 1.78, 3.51, 5.24, 6.97, 8.70]
    box(ax, xs[0], y, w, h, "Normalise", "one pipeline for index\nand queries; Indian\nscripts: NFC, digits,\nlakh and crore as\nnumbers", "new")
    box(ax, xs[1], y, w, h, "Anchor query", "high-idf terms, numbers,\ndates, quotes; names via\ntransliteration and\nk-gram lookup; earlier\nsources only", "new")
    box(ax, xs[2], y, w, h, "Candidates", "index elimination:\ncosine over the top\nterms, phrase hits on\nquotes; top 20", "detect")
    box(ax, xs[3], y, w, h, "Verify top 5", "sentence alignment\n(multilingual for\nHindi) and nine more\nsignals; logistic\nverifier", "new")
    box(ax, xs[4], y, w, h, "Provenance", "earlier article is the\nsource; without dates,\ncoverage asymmetry;\ncopy graph and g(d)", "detect")
    box(ax, xs[5], y, 1.45, h, "Output", "source(s) with shares\nand evidence;\nsearch ranks by\nrelevance + λ g(d)", "detect")
    for i in range(5):
        a, b = xs[i] + w, xs[i + 1]
        arrow(ax, (a, y + h / 2), (b, y + h / 2))
    arrow(ax, (6.95, 3.25), (4.3, y + h), color=GREY, rad=-0.15, ls="--")
    ax.text(5.5, 2.98, "postings", fontsize=6.2, color=GREY, rotation=0)

    arrow(ax, (xs[4] + w / 2, y), (xs[1] + w / 2, y), color=OCHRE, rad=-0.15, lw=1.3)
    ax.text(5.05, 0.66, "explain-away loop (mashups): query again with only the suspect sentences that no\nfound source explains yet, excluding the sources already found; stop when none verifies",
            fontsize=6.6, color=OCHRE, ha="center", va="top")

    ax.add_patch(FancyBboxPatch((0.05, 0.05), 0.22, 0.16, boxstyle="round,pad=0.01", fc=ROW, ec=OX, lw=1))
    ax.text(0.33, 0.13, "mid-term detection stage", fontsize=6.4, va="center")
    ax.add_patch(FancyBboxPatch((2.15, 0.05), 0.22, 0.16, boxstyle="round,pad=0.01", fc="#F4F1E4", ec=OCHRE, lw=1))
    ax.text(2.43, 0.13, "extended in the final project", fontsize=6.4, va="center")
    ax.add_patch(FancyBboxPatch((4.5, 0.05), 0.22, 0.16, boxstyle="round,pad=0.01", fc=PAPER, ec=GREY, lw=1))
    ax.text(4.78, 0.13, "crawler and index", fontsize=6.4, va="center")

    fig.savefig(OUT / "pipeline_final.pdf", bbox_inches="tight")
    fig.savefig(OUT / "pipeline_final.png", dpi=150, bbox_inches="tight")
    print("wrote", OUT / "pipeline_final.pdf")


if __name__ == "__main__":
    main()
