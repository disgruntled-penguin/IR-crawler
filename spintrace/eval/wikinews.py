"""Published Wikinews articles (CC BY 2.5) as originals for synthetic rewrites.

Read from the official Wikimedia dump, because Wikinews robots.txt disallows /w/ (the API) for bots.
"""
import bz2
import json
import re
import xml.etree.ElementTree as ET

import mwparserfromhell
import requests

from .. import config
from ..crawl.extract import to_epoch

DUMP_URL = "https://dumps.wikimedia.org/enwikinews/latest/enwikinews-latest-pages-articles.xml.bz2"
DUMP = config.DATA / "raw" / "enwikinews-latest-pages-articles.xml.bz2"
OUT = config.ROOT / "evaldata" / "wikinews.jsonl"
LINK_TEMPLATES = {"w", "wikipedia", "wp"}


def download():
    if DUMP.exists():
        return
    DUMP.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(DUMP_URL, headers={"User-Agent": config.USER_AGENT}, stream=True, timeout=60) as r:
        r.raise_for_status()
        with DUMP.open("wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)


def wikitext_to_plain(text):
    code = mwparserfromhell.parse(text)
    for t in code.filter_templates(recursive=False):
        name = str(t.name).strip().lower()
        if name in LINK_TEMPLATES and t.params:
            label = t.params[-1].value if len(t.params) > 1 else t.params[0].value
            try:
                code.replace(t, str(label))
            except ValueError:
                pass
    plain = code.strip_code(normalize=True, collapse=True)
    plain = re.split(r"\n==\s*(Sources|External links|Related news|See also|Interviews?)\s*==", plain)[0]
    lines = []
    for line in plain.split("\n"):
        line = re.sub(r"\s+", " ", line).strip()
        if not line or line.startswith("==") or line.lower().startswith(("thumb|", "file:", "image:")):
            continue
        line = re.sub(r"^(thumb|left|right|\d+px)(\|[^|]*)*\|", "", line)
        lines.append(line)
    return "\n".join(lines)


def iter_published():
    ns = "{http://www.mediawiki.org/xml/export-0.11/}"
    with bz2.open(DUMP, "rb") as f:
        for _, el in ET.iterparse(f):
            if not el.tag.endswith("page"):
                continue
            tag = el.tag[: -len("page")]
            if el.findtext(f"{tag}ns") != "0":
                el.clear()
                continue
            title = el.findtext(f"{tag}title")
            text = el.findtext(f"{tag}revision/{tag}text") or ""
            el.clear()
            if "{{publish" not in text.lower() and "{{archived" not in text.lower():
                continue
            m = re.search(r"\{\{date\|([^}]+)\}\}", text)
            if not m:
                continue
            yield title, to_epoch(m.group(1)), text


def build(n=400, min_words=150, max_words=900):
    """Keep the n most recent published articles with a usable body."""
    download()
    rows = []
    for title, date, text in iter_published():
        if not date:
            continue
        rows.append((date, title, text))
    rows.sort(reverse=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    with OUT.open("w") as out:
        for date, title, text in rows:
            body = wikitext_to_plain(text)
            words = len(body.split())
            if words < min_words or words > max_words:
                continue
            url = "https://en.wikinews.org/wiki/" + title.replace(" ", "_")
            out.write(json.dumps({"title": title, "url": url, "published": date, "text": body,
                                  "license": "CC BY 2.5, Wikinews contributors"}) + "\n")
            kept += 1
            if kept >= n:
                break
    return kept
