"""Does text length alone separate Hindi derivatives from same-event negatives?

AUC of word count with the derivative as the positive that is shorter than the negative (0.5 means
length says nothing, 1.0 means a length cutoff classifies perfectly, below 0.5 the derivative is longer).
Run: python -m spintrace.eval.xl_length_check
"""
import json
from collections import defaultdict

import numpy as np

from .. import config, store
from .run import split_of
from . import xl_rewrites

LEVELS = ["hi_translation", "hi_seo", "hi_summary"]


def auc(pos, neg):
    p, n = np.array(pos), np.array(neg)
    return float(((p[:, None] < n[None, :]).sum() + 0.5 * (p[:, None] == n[None, :]).sum()) / (len(p) * len(n)))


def main():
    words = defaultdict(list)
    for line in xl_rewrites.OUT.open():
        r = json.loads(line)
        if r["level"] in LEVELS + ["hi_facts_only", "hi_facts_short"] and r.get("text"):
            words[(split_of(store.doc_id_for(r["orig_url"])), r["level"])].append(len(r["text"].split()))
    out = {"median_words": {f"{s}:{lv}": float(np.median(v)) for (s, lv), v in sorted(words.items())}, "auc": {}}
    for sp in ("dev", "test"):
        for lv in LEVELS:
            out["auc"][f"{sp}:{lv}:vs_long"] = auc(words[(sp, lv)], words[(sp, "hi_facts_only")])
    for lv in LEVELS:
        out["auc"][f"test:{lv}:vs_short"] = auc(words[("test", lv)], words[("test", "hi_facts_short")])
    (config.RESULTS / "crosslingual_length_check.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
