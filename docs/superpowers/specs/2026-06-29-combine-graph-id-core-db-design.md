# Design: Combine graph-id-core and graph-id-db

**Date:** 2026-06-29
**Issue:** kmu/graph-id-core#100

## Problem

Users must install two separate packages to generate IDs and look up structures in materials databases:

```bash
pip install graph-id-core   # ID generation
pip install graph-id-db     # database lookup
```

This causes confusion. The goal is a single install that covers both.

## Approach

Merge `graph-id-db` into `graph-id-core` (Option A: full merge). Add `orjson` and `huggingface-hub` as required dependencies. Deprecate `graph-id-db` with a stub that re-exports from `graph-id-core`.

## Code Structure

**Files to add to graph-id-core:**

- `graph_id/db/finder.py` — copied from `graph_id_db/finder.py` in graph-id-db; update `DB_PATH` to point to the new `raw/` location
- `graph_id/db/__init__.py` — empty init
- `raw/id_jsons/` — copied verbatim from graph-id-db; included in package data

**Files to update in graph-id-core:**

- `graph_id/__init__.py` — add `from graph_id.db.finder import Finder`
- `pyproject.toml` — add `orjson >= 3.10` and `huggingface-hub >= 0.35.3` to required deps; include `raw/**/*` in package data

## Dependencies

Add to `[tool.poetry.dependencies]`:

```toml
orjson = ">=3.10"
huggingface-hub = ">=0.35.3"
```

## Deprecation Stub (graph-id-db)

Publish a final version of `graph-id-db` with `graph_id_db/finder.py` replaced by:

```python
import warnings
warnings.warn(
    "graph-id-db is deprecated. Use graph-id-core instead: from graph_id import Finder",
    DeprecationWarning,
    stacklevel=2,
)
from graph_id.db.finder import Finder
```

Existing users' code continues to work. They see a deprecation warning pointing to the new import path.

## Testing

Add `tests/py/test_finder.py`:

- Instantiate `Finder` and call `find_fast` with a known Graph ID (e.g. `NaCl-3D-88c8e156db1b0fd9`)
- Assert the result is a non-empty list of dicts
- No network-dependent tests (AFLOW, OQMD, PCOD lookups) — those require HuggingFace Hub access and are covered by graph-id-db's existing tests

## Result

After this change:

```bash
pip install graph-id-core   # gets everything

from graph_id import GraphIDMaker, Finder
```
