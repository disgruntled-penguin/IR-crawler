"""Palette and fonts shared by the report figures."""
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

INK = "#1E1C1A"
OXBLOOD = "#7A2E2E"
OCHRE = "#B5832A"
OLIVE = "#5F6B3A"
GREY = "#8A837A"
PAPER = "#F3EFE7"

FIG_DIR = Path(__file__).resolve().parent / "figures"
TEXT_WIDTH_IN = 16.6 / 2.54

# Matplotlib only sees the first face (Bold) of a .ttc, so the faces we want are written out as single fonts.
AVENIR_TTC = "/System/Library/Fonts/Avenir Next.ttc"
AVENIR_FACES = {"Avenir Next Regular": 7, "Avenir Next Demi Bold": 2}


def _register_avenir():
    try:
        from fontTools.ttLib import TTCollection
        coll = TTCollection(AVENIR_TTC)
    except Exception:
        return None
    out = Path(tempfile.mkdtemp(prefix="spintrace-fonts-"))
    for name, i in AVENIR_FACES.items():
        path = out / (name.replace(" ", "") + ".ttf")
        coll.fonts[i].save(str(path))
        font_manager.fontManager.addfont(str(path))
    return font_manager.FontProperties(fname=str(out / "AvenirNextRegular.ttf")).get_name()


def setup():
    family = _register_avenir() or "DejaVu Sans"
    plt.rcParams.update({
        "font.family": [family, "DejaVu Sans"],
        "font.size": 8,
        "text.color": INK,
        "axes.labelcolor": INK,
        "axes.edgecolor": GREY,
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "xtick.color": INK,
        "ytick.color": INK,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })
    FIG_DIR.mkdir(exist_ok=True)


# Methods: SpinTrace in oxblood, baselines paired by family in ochre (term weighting), olive (shingles), grey.
METHODS = [
    ("spintrace", "SpinTrace", dict(color=OXBLOOD, ls="-", marker="o", lw=2.2, ms=4.5, zorder=6)),
    ("bm25", "BM25", dict(color=OCHRE, ls="-", marker="s", lw=1.3, ms=3.5)),
    ("tfidf_cosine", "tf-idf cosine", dict(color=OCHRE, ls=(0, (4, 2)), marker="D", lw=1.3, ms=3, mfc="white")),
    ("shingle_containment", "Shingle containment (all pairs)", dict(color=OLIVE, ls="-", marker="^", lw=1.3, ms=3.8)),
    ("minhash_jaccard", "MinHash Jaccard", dict(color=OLIVE, ls=(0, (4, 2)), marker="v", lw=1.3, ms=3.8, mfc="white")),
    ("dense_cosine", "Dense cosine", dict(color=GREY, ls="-", marker="o", lw=1.3, ms=3.5, mfc="white")),
    ("exact_hash", "Exact hash", dict(color=GREY, ls=(0, (1.5, 1.5)), marker="x", lw=1.3, ms=3.8)),
]
