#!/bin/bash
# Final-phase evaluation, ordered by value so that an interrupted run still leaves the most important results.
# Every step caches its work per item, so rerunning after more rewrites are generated only adds the new items.
# Needs the crawl database, the indexes (python -m spintrace build --stem, build-xl) and evaldata/rewrites_xl.jsonl.
set -u
cd "$(dirname "$0")/.."
PY=${PYTHON:-python3}
LOG=data/logs/final_run.log
mkdir -p data/logs

run() {
  echo "$(date +%H:%M:%S) START $*" | tee -a "$LOG"
  "$@" >> "$LOG" 2>&1
  rc=$?
  echo "$(date +%H:%M:%S) END $* exit=$rc" | tee -a "$LOG"
}

run $PY -m spintrace eval-multi                 # mashups, controls, baselines, ablations, LLM mashups
run $PY -m spintrace eval-multi --live          # multi-source pass over flagged live articles
run $PY -m spintrace eval-xl --fresh            # Hindi levels, negatives, baselines, ablations, other scripts
run $PY -m spintrace live-xl --rebuild          # Indian-language crawl against the English crawl
run $PY -m spintrace live-xl --same-site        # diagnostic: bilingual outlets against their own English pages
run $PY -m spintrace.eval.dataset_stats         # results/dataset.md
run $PY report/make_tables.py                   # LaTeX tables and number macros from results/
run $PY report/make_figures.py                  # report charts
