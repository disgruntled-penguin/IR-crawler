"""Pipeline diagram for the report, in the report palette: python report/pipeline_diagram.py

Adapted from scripts/pipeline_diagram.py; coordinates are in centimetres on a 16.6 cm text width.
"""
from matplotlib.patches import FancyArrowPatch, Rectangle

import plotstyle as ps
from plotstyle import plt

W, H = 16.6, 9.0
ROWS = {1: (6.55, 2.4), 2: (3.2, 2.6), 3: (0.05, 2.15)}

# (row, x, w, title, module path, body, novel)
BOXES = {
    "seeds": (1, 0.05, 3.5, "Seeds", "seeds/seeds.csv",
              "36 outlets: RSS and\nAtom feeds, three\nhomepages; robots.txt\nchecked for each", False),
    "crawl": (1, 4.35, 3.6, "1  Polite crawl", "spintrace/crawl/",
              "Mercator front and back\nqueues; robots.txt and\nCrawl-delay, ≥ 5 s per\nhost; backoff; trap guards", False),
    "index": (1, 8.75, 3.6, "2  Store and index", "text.py  index.py  dedup.py",
              "positional index with title,\nbody and quote zones;\ntf-idf, BM25; content-seen:\nhash, MinHash, LSH", False),
    "wiki": (1, 13.05, 3.5, "Wikinews dump", "eval/wikinews.py, rewrites.py",
             "400 originals, 4,000\ndistractors; graded\nrewrites mixed into\nthe index as suspects", False),
    "cand": (2, 0.05, 5.1, "3a  Candidates (IR)", "detect/candidates.py",
             "30 highest-weight terms, df ≥ 2,\nnames and numbers ×2; lnc.ltc\ncosine, heap top 20; quote windows\nas phrase queries; other sites only,\npublished earlier", True),
    "verify": (2, 5.75, 5.1, "3b  Verify the top 5", "detect/signals.py, verify.py",
               "sentence alignment (MiniLM),\norder, back coverage, quotes,\nnames and numbers, shingle\ncontainment; logistic p with\nweights fit on dev", True),
    "prov": (2, 11.45, 5.1, "3c  Provenance", "detect/pipeline.py, scan.py",
             "earlier publish date wins;\nundated: coverage asymmetry,\nelse 'uncertain'; verified edges\nform the copy graph", True),
    "search": (3, 2.4, 5.6, "4a  Search", "search.py",
               "net(q, d) = relevance + λ g(d)\ng(d) = 1 − p(strongest source edge)\nper-domain integrity report", False),
    "eval": (3, 8.6, 5.6, "4b  Evaluation", "spintrace/eval/",
             "six baselines with dev thresholds;\nablations; judged live pairs and\nqueries; crawl politeness audit", False),
}
ARROWS = [("seeds", "crawl"), ("crawl", "index"), ("wiki", "index"), ("cand", "verify"), ("verify", "prov")]


def geom(k):
    row, x, w = BOXES[k][:3]
    y, h = ROWS[row]
    return x, y, w, h


def box(ax, k):
    x, y, w, h = geom(k)
    title, path, body, novel = BOXES[k][3:]
    ax.add_patch(Rectangle((x, y), w, h, fc="white" if novel else ps.PAPER, ec=ps.OXBLOOD if novel else ps.GREY,
                           lw=1.1 if novel else 0.6))
    ax.text(x + 0.2, y + h - 0.2, title, ha="left", va="top", fontsize=7.8, weight="semibold",
            color=ps.OXBLOOD if novel else ps.INK)
    ax.text(x + 0.2, y + h - 0.62, path, ha="left", va="top", fontsize=5.8, color=ps.GREY)
    ax.text(x + 0.2, y + h - 0.98, body, ha="left", va="top", fontsize=6.3, color=ps.INK, linespacing=1.28)


def arrow(ax, a, b):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=7, color=ps.GREY, lw=0.8,
                                 shrinkA=0, shrinkB=0))


def line(ax, xs, ys):
    ax.plot(xs, ys, color=ps.GREY, lw=0.8, solid_capstyle="butt")


def main():
    ps.setup()
    fig = plt.figure(figsize=(W / 2.54, H / 2.54))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    for k in BOXES:
        box(ax, k)
    for a, b in ARROWS:
        ax_, ay, aw, ah = geom(a)
        bx, by, bw, bh = geom(b)
        y = ay + ah / 2
        if ax_ < bx:
            arrow(ax, (ax_ + aw, y), (bx, y))
        else:
            arrow(ax, (ax_, y), (bx + bw, y))

    # the index feeds detection; provenance feeds search and evaluation
    ix, iy, iw, _ = geom("index")
    cx, cy, cw, ch = geom("cand")
    mid = (iy + cy + ch) / 2
    line(ax, [ix + iw / 2, ix + iw / 2, cx + 1.4], [iy, mid, mid])
    arrow(ax, (cx + 1.4, mid), (cx + 1.4, cy + ch))
    ax.text(cx + 1.6, mid + 0.1, "the suspect's rare terms, idf, postings", fontsize=5.8, color=ps.GREY, va="bottom")

    px, py, pw, _ = geom("prov")
    sx, sy, sw, sh = geom("search")
    ex, *_ = geom("eval")
    mid = (py + sy + sh) / 2
    line(ax, [px + 1.4, px + 1.4, sx + sw / 2], [py, mid, mid])
    arrow(ax, (sx + sw / 2, mid), (sx + sw / 2, sy + sh))
    arrow(ax, (ex + 2.8, mid), (ex + 2.8, sy + sh))
    ax.text(px + 1.55, mid + 0.22, "copy graph, g(d)", fontsize=5.8, color=ps.GREY)
    fig.savefig(ps.FIG_DIR / "pipeline.pdf", bbox_inches=None, pad_inches=0)
    plt.close(fig)
    print(ps.FIG_DIR / "pipeline.pdf")


if __name__ == "__main__":
    main()
