"""URL normalisation and the URL filters that keep the crawl focused and out of traps."""
import re
from urllib.parse import urlsplit, urlunsplit, parse_qsl, urlencode, urljoin

from .. import config

TRACKING_PARAMS = re.compile(r"^(utm_|fbclid|gclid|mc_|ocid|cmpid|ref$|ref_|share|at_|ito|xtor|CMP|src$|guccounter)", re.I)

# Comments, user profiles, accounts and non-article sections are never fetched.
SKIP_PATH = re.compile(
    r"/(comments?|profile|profiles|user|users|author|authors|people|login|signin|sign-in|register|account|"
    r"subscribe|subscription|newsletter|newsletters|cart|checkout|search|tag|tags|topic|topics|"
    r"video|videos|live|gallery|galleries|podcast|podcasts|audio|sounds|iplayer|weather|"
    r"wp-admin|wp-login|feed|amp|print|share|cdn-cgi|programmes|sport/av|crossword|puzzles|"
    r"horoscope|horoscopes|obituaries|jobs|careers|advertis\w*|contact|about|privacy|terms)(/|$)",
    re.I,
)
SKIP_EXT = re.compile(r"\.(jpg|jpeg|png|gif|webp|svg|mp4|mp3|pdf|zip|css|js|json|ico|woff2?|xml|rss)$", re.I)
SKIP_QUERY = re.compile(r"(replytocom|comment|share=|print=|page=\d{2,}|sort=|filter=)", re.I)


def normalise(url, base=None):
    """Canonical form: absolute, lowercase scheme and host, no fragment, default port or tracking params."""
    if base:
        url = urljoin(base, url)
    try:
        parts = urlsplit(url.strip())
    except ValueError:
        return None
    if parts.scheme not in ("http", "https") or not parts.hostname:
        return None
    host = parts.hostname.lower()
    if parts.port and parts.port not in (80, 443):
        host = f"{host}:{parts.port}"
    path = re.sub(r"/{2,}", "/", parts.path or "/")
    # Trailing slashes are kept: many sites 301 between the two forms, and stripping one loops.
    path = re.sub(r"/index\.html?$", "/", path)
    query = [(k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True) if not TRACKING_PARAMS.match(k)]
    query.sort()
    return urlunsplit(("https" if parts.scheme in ("http", "https") else parts.scheme, host, path, urlencode(query), ""))


def host_of(url):
    return urlsplit(url).netloc.lower()


def site_of(host):
    """Registrable-ish site key (last two labels, three for co.uk style) used to group hosts of one outlet."""
    labels = host.split(":")[0].split(".")
    if len(labels) >= 3 and labels[-2] in ("co", "com", "org", "net", "ac", "gov") and len(labels[-1]) == 2:
        return ".".join(labels[-3:])
    return ".".join(labels[-2:])


def trap_reason(url):
    """Return why a URL looks like a spider trap or an out-of-scope page, or None if it is fine."""
    if len(url) > config.MAX_URL_LEN:
        return "url_too_long"
    parts = urlsplit(url)
    segs = [s for s in parts.path.split("/") if s]
    if len(segs) > config.MAX_PATH_DEPTH:
        return "path_too_deep"
    if segs and max(segs.count(s) for s in segs) > config.MAX_REPEATED_SEGMENT:
        return "repeated_segment"
    if parts.query and len(parse_qsl(parts.query)) > config.MAX_QUERY_PARAMS:
        return "too_many_params"
    if re.search(r"/(19|20)\d\d/\d\d?/\d\d?/(19|20)\d\d/", parts.path):
        return "calendar_loop"
    return None


def skip_reason(url):
    """Out-of-scope pages: comments, profiles, media, utility pages."""
    parts = urlsplit(url)
    if SKIP_EXT.search(parts.path):
        return "non_html"
    if SKIP_PATH.search(parts.path):
        return "non_article_section"
    if parts.query and SKIP_QUERY.search(parts.query):
        return "comment_or_utility_query"
    return None
