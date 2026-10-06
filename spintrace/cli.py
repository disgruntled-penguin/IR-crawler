"""SpinTrace command line: python -m spintrace <command>."""
import argparse

from . import config


def cmd_crawl(args):
    from .crawl.crawler import Crawler, setup_logging
    setup_logging(config.LOG_DIR / "crawl.log")
    Crawler(args.seeds, max_pages=args.max_pages).run(hours=args.hours)


def main(argv=None):
    p = argparse.ArgumentParser(prog="spintrace")
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("crawl", help="polite focused crawl from the seed list")
    c.add_argument("--seeds", default=str(config.SEEDS / "seeds.csv"))
    c.add_argument("--hours", type=float)
    c.add_argument("--max-pages", type=int)
    c.set_defaults(fn=cmd_crawl)
    args = p.parse_args(argv)
    args.fn(args)
