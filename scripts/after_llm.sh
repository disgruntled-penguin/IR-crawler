#!/bin/sh
# Waits for the running seo/summary generator, then cleans rule-based levels and generates facts-only negatives.
while kill -0 "$1" 2>/dev/null; do sleep 30; done
python3 -c "from spintrace.eval import rewrites; print('rebuilt', rewrites.rebuild_rule_based())"
python3 -c "from spintrace.eval import rewrites; rewrites.build(n_llm=200, levels=['facts_only'])"
echo AFTER_LLM_DONE
