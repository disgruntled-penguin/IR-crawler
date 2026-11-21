"""Does sentence alignment work in each language? Mean alignment signals on the true source.

Only items whose true source is among the five verified candidates count, so the numbers describe the
verifier's input, not retrieval. A model that does not cover a language aligns a sentence with its
translation no better than with other sentences. Run: python -m spintrace.eval.xl_align_check
"""
import json
import pickle

import numpy as np

from .. import config
from . import xl_eval

LEVELS = ["hi_translation", "ta_translation", "te_translation", "bn_translation"]


def main():
    with xl_eval.RAW.open("rb") as f:
        items = pickle.load(f)
    items = items if isinstance(items, list) else items.get("items", items)
    out = {}
    for lv in LEVELS:
        mean, cov = [], []
        for it in items:
            if it["level"] != lv:
                continue
            hit = [v for v in it["runs"]["full"]["verified"] if v["doc"] == it["source"]]
            if hit:
                mean.append(hit[0]["signals"]["align_mean"])
                cov.append(hit[0]["signals"]["align_coverage"])
        out[lv] = {"n": len(mean), "align_mean": float(np.mean(mean)), "align_coverage": float(np.mean(cov))}
    (config.RESULTS / "crosslingual_align_check.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
