"""Harness-1 original closed-loop eval hyperparameters.

Table-2 / evaluate_harness1.py uses MAX_TURNS=40 (CLI), max_tokens=2048,
temperature=1.0, and SEARCH_DISPLAY_LIMIT=10 against a real corpus index.
SCAPE training keeps a short horizon; eval must not inherit those smoke defaults.
"""

from __future__ import annotations

# evaluate_harness1.py --max-turns default is MAX_TURNS (env, 35);
# the published Table-2 command uses 40.
HARNESS1_EVAL_MAX_TURNS = 40
HARNESS1_EVAL_MAX_NEW_TOKENS = 2048
HARNESS1_EVAL_TEMPERATURE = 1.0
HARNESS1_EVAL_MAX_MODEL_LEN = 32768
HARNESS1_EVAL_SEARCH_K = 10
# Seed the local WM from BM25 when live per-turn search is unavailable.
HARNESS1_EVAL_DOC_STORE_K = 100
