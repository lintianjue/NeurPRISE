"""Fixtures local to `tests/scripts/`.

`producer_resolve_out_dir`, `producer_module` and `producer_eval_dir` live in the repo-level
`tests/conftest.py`, so any test can anchor a control on the real producer in one line;
pytest injects them here unchanged.  This file is kept so `tests/scripts/` retains a place
for fixtures that are genuinely local to it.
"""
from __future__ import annotations
