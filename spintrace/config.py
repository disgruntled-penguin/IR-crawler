import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = Path(os.environ.get("SPINTRACE_DATA", ROOT / "data"))
DB_PATH = DATA / "spintrace.db"
INDEX_DIR = DATA / "index"
LOG_DIR = DATA / "logs"
RESULTS = ROOT / "results"
SEEDS = ROOT / "seeds"

CONTACT = os.environ.get("SPINTRACE_CONTACT", "nishitakadali13@gmail.com")
USER_AGENT = f"SpinTraceBot/0.1 (CSD358 IR course project; contact: {CONTACT})"
ROBOTS_AGENT = "SpinTraceBot"

# Politeness: wait at least this long between requests to one host, more if Crawl-delay says so.
DEFAULT_DELAY = 5.0
MAX_DELAY = 600.0
MAX_CONSECUTIVE_ERRORS = 5
REQUEST_TIMEOUT = 20
MAX_BYTES = 3_000_000
ROBOTS_TTL = 24 * 3600
FEED_REPOLL = 45 * 60

# Spider-trap guards.
MAX_URL_LEN = 300
MAX_PATH_DEPTH = 8
MAX_QUERY_PARAMS = 3
MAX_REPEATED_SEGMENT = 2
MAX_PAGES_PER_HOST = 1500

MIN_ARTICLE_WORDS = 120


def ensure_dirs():
    for d in (DATA, INDEX_DIR, LOG_DIR, RESULTS):
        d.mkdir(parents=True, exist_ok=True)
