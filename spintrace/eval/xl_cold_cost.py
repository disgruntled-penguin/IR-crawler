"""Cold per-suspect cost of SpinTrace-XL: anchor retrieval plus verification with an empty embedding cache.

The evaluation reuses cached sentence embeddings, so its timings leave out the encoder. Here every
suspect starts from an empty cache, as a first sight of a new article would. Run on an idle machine:
python -m spintrace.eval.xl_cold_cost
"""
import json
import time

import numpy as np

from .. import config, embed_xl
from ..detect import crosslingual as xl
from . import xl_eval

N = 30


def main(n=N):
    rows = xl_eval.load_rows()
    emb = embed_xl.Embedder(cache=config.DATA / "eval" / "xl_cold_unused.pkl")
    ctx = xl.Context(emb=emb)
    items = [it for it in xl_eval.build_items(ctx, rows) if it["level"] == "hi_translation" and it["split"] == "test"]
    items = items[:n]
    emb.encode(["warm up the model"])
    cold, warm = [], []
    for it in items:
        s, ex = xl_eval.suspect(it), xl_eval.exclude_mask(ctx.idx, it)
        emb.cache = {}
        t = time.perf_counter()
        xl.analyse_text(ctx, s, exclude=ex, opts=xl_eval.VARIANTS["full"])
        cold.append(1000 * (time.perf_counter() - t))
        t = time.perf_counter()
        xl.analyse_text(ctx, s, exclude=ex, opts=xl_eval.VARIANTS["full"])
        warm.append(1000 * (time.perf_counter() - t))
    out = {"n": len(cold), "ms_per_suspect_cold": float(np.mean(cold)), "ms_per_suspect_cold_median": float(np.median(cold)),
           "ms_per_suspect_warm": float(np.mean(warm)), "note": "empty embedding cache per suspect, encoder warm, CPU"}
    (config.RESULTS / "crosslingual_cold_cost.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
