#!/bin/sh
# Every experiment, in order, from the current store. Tables and charts land in results/.
set -e
python3 -m spintrace build --stem
python3 -m spintrace eval --mine
python3 -m spintrace scan --where "origin IN ('crawl','synthetic','newsguard')"
python3 -m spintrace search-eval
python3 -m spintrace crawl-report
python3 -m spintrace export
python3 scripts/pipeline_diagram.py
