"""Main-text, title, date and link extraction from fetched pages, plus feed and sitemap parsing.

trafilatura does boilerplate removal (drops navigation, ads and comments); publish timestamps are
read from page metadata first because provenance needs times finer than a day.
"""
import json
import re
from datetime import datetime, timezone

import trafilatura
from dateutil import parser as dateparser
from lxml import etree, html as lhtml

DATE_META = [
    ("property", "article:published_time"),
    ("name", "article:published_time"),
    ("itemprop", "datePublished"),
    ("name", "pubdate"),
    ("name", "publishdate"),
    ("name", "publish-date"),
    ("name", "date"),
    ("name", "dc.date.issued"),
    ("name", "DC.date.issued"),
    ("property", "og:published_time"),
    ("name", "parsely-pub-date"),
    ("name", "sailthru.date"),
]


def to_epoch(value):
    if not value:
        return None
    try:
        dt = dateparser.parse(str(value))
    except (ValueError, OverflowError, TypeError):
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    year = dt.year
    if year < 1995 or year > datetime.now(timezone.utc).year + 1:
        return None
    return dt.timestamp()


def _jsonld_dates(tree):
    for node in tree.xpath('//script[@type="application/ld+json"]/text()'):
        try:
            data = json.loads(node)
        except (ValueError, TypeError):
            continue
        stack = [data]
        while stack:
            d = stack.pop()
            if isinstance(d, list):
                stack.extend(d)
            elif isinstance(d, dict):
                if "datePublished" in d:
                    yield d["datePublished"]
                stack.extend(v for v in d.values() if isinstance(v, (dict, list)))


def page_date(tree):
    """Publish time from JSON-LD or meta tags; returns (epoch, source) or (None, None)."""
    for v in _jsonld_dates(tree):
        t = to_epoch(v)
        if t:
            return t, "jsonld"
    for attr, name in DATE_META:
        for v in tree.xpath(f'//meta[@{attr}="{name}"]/@content'):
            t = to_epoch(v)
            if t:
                return t, "meta"
    for v in tree.xpath("//time/@datetime")[:1]:
        t = to_epoch(v)
        if t:
            return t, "time_tag"
    return None, None


def is_article_page(tree):
    types = " ".join(tree.xpath('//meta[@property="og:type"]/@content')).lower()
    if "article" in types:
        return True
    return bool(re.search(r'"@type"\s*:\s*"(News)?Article|"@type"\s*:\s*"(Report|BlogPosting)', " ".join(
        tree.xpath('//script[@type="application/ld+json"]/text()'))))


def extract_page(raw_html, url):
    """Return dict(title, text, published, date_source, is_article, links) or None if unparsable."""
    try:
        tree = lhtml.fromstring(raw_html)
    except (etree.ParserError, ValueError):
        return None
    tree.make_links_absolute(url, resolve_base_href=True)
    links = [a for a in tree.xpath("//a/@href")]
    published, date_source = page_date(tree)
    doc = trafilatura.bare_extraction(
        raw_html, url=url, include_comments=False, include_tables=False, favor_precision=True, with_metadata=True,
    )
    text, title = "", None
    if doc is not None:
        d = doc.as_dict() if hasattr(doc, "as_dict") else doc
        text = d.get("text") or ""
        title = d.get("title")
        if not published and d.get("date"):
            published, date_source = to_epoch(d["date"]), "trafilatura_day"
    if not title:
        t = tree.xpath("//title/text()")
        title = t[0].strip() if t else None
    return {
        "title": title,
        "text": text,
        "published": published,
        "date_source": date_source,
        "is_article": is_article_page(tree),
        "links": links,
    }


def parse_feed_or_sitemap(raw):
    """Return (entries, child_sitemaps). Entries are (url, published_epoch or None)."""
    try:
        root = etree.fromstring(raw, parser=etree.XMLParser(recover=True, resolve_entities=False, huge_tree=False))
    except (etree.XMLSyntaxError, ValueError):
        return [], []
    if root is None:
        return [], []
    entries, children = [], []

    def local(el):
        return etree.QName(el).localname if isinstance(el.tag, str) else ""

    for el in root.iter():
        name = local(el)
        if name == "item":
            link = date = None
            for c in el:
                n = local(c)
                if n == "link" and c.text:
                    link = c.text.strip()
                elif n in ("pubDate", "date") and c.text:
                    date = to_epoch(c.text.strip())
            if link:
                entries.append((link, date))
        elif name == "entry":
            link = date = None
            for c in el:
                n = local(c)
                if n == "link" and c.get("href") and c.get("rel", "alternate") == "alternate":
                    link = c.get("href")
                elif n in ("published", "updated") and c.text and not date:
                    date = to_epoch(c.text.strip())
            if link:
                entries.append((link, date))
        elif name in ("url", "sitemap"):
            loc = date = None
            for c in el.iter():
                n = local(c)
                if n == "loc" and c.text:
                    loc = c.text.strip()
                elif n in ("publication_date", "lastmod") and c.text and not date:
                    date = to_epoch(c.text.strip())
            if loc:
                (entries if name == "url" else children).append((loc, date))
    return entries, children
