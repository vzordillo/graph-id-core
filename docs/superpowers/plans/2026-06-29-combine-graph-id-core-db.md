# Combine graph-id-core and graph-id-db Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Merge `graph-id-db`'s `Finder` class into `graph-id-core` so users only need `pip install graph-id-core` to get both ID generation and database lookup.

**Architecture:** Add `graph_id/db/finder.py` (copied from graph-id-db with updated `DB_PATH`), add `orjson` and `huggingface-hub` as required deps, and re-export `Finder` from `graph_id/__init__.py`. Without the `raw/` data bundle, `find_fast` degrades gracefully to `[]` — the maintainer will be asked to publish MP+IZA data to HuggingFace Hub separately.

**Tech Stack:** Python, orjson ≥ 3.10, huggingface-hub ≥ 0.35.3, pytest, conda (env: `graph_id`)

## Global Constraints

- All commands run as: `conda run -n graph_id <command>`
- No Co-Authored-By lines in commits
- Do not bundle `raw/` data — `find_fast` returns `[]` gracefully when data is absent
- Follow existing code style (black, line-length 119)

---

### Task 1: Add `graph_id/db/finder.py`

**Files:**
- Create: `graph_id/db/__init__.py`
- Create: `graph_id/db/finder.py`

**Interfaces:**
- Produces: `Finder` class with `find(graph_id: str, is_fast: bool = False) -> list[dict[str, str]]`

- [ ] **Step 1: Create the `db` subpackage**

```bash
mkdir -p graph_id/db
touch graph_id/db/__init__.py
```

- [ ] **Step 2: Write the failing test**

Create `tests/py/test_finder.py`:

```python
"""Tests for Finder class."""
from graph_id.db.finder import Finder


def test_finder_import():
    finder = Finder()
    assert finder is not None


def test_find_fast_returns_list_without_raw_data():
    finder = Finder()
    result = finder.find_fast("NaCl-3D-88c8e156db1b0fd9")
    assert isinstance(result, list)


def test_find_returns_list():
    finder = Finder()
    result = finder.find("NaCl-3D-88c8e156db1b0fd9")
    assert isinstance(result, list)
```

- [ ] **Step 3: Run test to verify it fails**

```bash
conda run -n graph_id python -m pytest tests/py/test_finder.py -v
```

Expected: `ModuleNotFoundError: No module named 'graph_id.db'`

- [ ] **Step 4: Create `graph_id/db/finder.py`**

```python
from pathlib import Path

import orjson
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import EntryNotFoundError

DB_PATH = Path(__file__).parent.parent.parent / "raw/id_jsons"


class Finder:
    def find(self, graph_id: str, is_fast: bool = False) -> list[dict[str, str]]:
        """Find structures in materials databases matching the given Graph ID.

        Args:
            graph_id: GraphID calculated using graph-id-core
            is_fast: If True, find only on-memory entries (from IZA and Materials Project)
        """
        if is_fast:
            return self.find_fast(graph_id)

        ret_dict = []
        fast_graph_ids = self.find_fast(graph_id)
        if fast_graph_ids:
            ret_dict += fast_graph_ids

        aflow_graph_ids = self.find_aflow_entries(graph_id)
        if aflow_graph_ids:
            ret_dict += aflow_graph_ids

        oqmd_graph_ids = self.find_oqmd_entries(graph_id)
        if oqmd_graph_ids:
            ret_dict += oqmd_graph_ids

        pcod_graph_ids = self.find_pcod_entries(graph_id)
        if pcod_graph_ids:
            ret_dict += pcod_graph_ids

        return ret_dict

    def find_fast(self, graph_id: str) -> list[dict[str, str]]:
        """Find only on-memory entries (MP + IZA). Returns [] if data not bundled."""
        ret_dict = []
        dir_name = graph_id[:2]
        file_name = graph_id[:4]
        db_path = DB_PATH / dir_name / f"{file_name}.json"
        if db_path.exists():
            with open(db_path) as f:
                docs = orjson.loads(f.read())
                if docs.get(graph_id):
                    ret_dict = docs.get(graph_id)
        return ret_dict

    def find_aflow_entries(self, graph_id: str) -> list[dict[str, str]]:
        """Find only AFLOW entries."""
        ret_dict = []
        dir_name = graph_id[:2]
        file_name = graph_id[:4]
        try:
            local_path = hf_hub_download(
                repo_id="kamabata/aflow_graph_ids",
                filename=f"id_jsons/{dir_name}/{file_name}.json",
                repo_type="dataset",
            )
            with open(local_path) as f:
                docs = orjson.loads(f.read())
                if docs.get(graph_id):
                    ret_dict = docs.get(graph_id)
            return ret_dict
        except EntryNotFoundError:
            return []

    def find_oqmd_entries(self, graph_id: str) -> list[dict[str, str]]:
        """Find only OQMD entries."""
        ret_dict = []
        dir_name = graph_id[:2]
        file_name = graph_id[:4]
        try:
            local_path = hf_hub_download(
                repo_id="kamabata/oqmd_graph_ids",
                filename=f"id_jsons/{dir_name}/{file_name}.json",
                repo_type="dataset",
            )
            with open(local_path) as f:
                docs = orjson.loads(f.read())
                if docs.get(graph_id):
                    ret_dict = docs.get(graph_id)
            return ret_dict
        except EntryNotFoundError:
            return []

    def find_pcod_entries(self, graph_id: str) -> list[dict[str, str]]:
        """Find only PCOD entries."""
        ret_dict = []
        dir_name = graph_id[:2]
        file_name = graph_id[:4]
        try:
            local_path = hf_hub_download(
                repo_id="kamabata/pcod_graph_ids",
                filename=f"id_jsons/{dir_name}/{file_name}.json",
                repo_type="dataset",
            )
            with open(local_path) as f:
                docs = orjson.loads(f.read())
                if docs.get(graph_id):
                    ret_dict = docs.get(graph_id)
            return ret_dict
        except EntryNotFoundError:
            return []
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
conda run -n graph_id python -m pytest tests/py/test_finder.py -v
```

Expected: 3 PASSED

- [ ] **Step 6: Commit**

```bash
git add graph_id/db/__init__.py graph_id/db/finder.py tests/py/test_finder.py
git commit -m "Add Finder class to graph_id/db"
```

---

### Task 2: Add dependencies to `pyproject.toml`

**Files:**
- Modify: `pyproject.toml`

**Interfaces:**
- Consumes: nothing
- Produces: `orjson` and `huggingface-hub` available as installed deps

- [ ] **Step 1: Add deps to `pyproject.toml`**

In the `[tool.poetry.dependencies]` section, add after the existing deps:

```toml
orjson = ">=3.10"
huggingface-hub = ">=0.35.3"
```

- [ ] **Step 2: Install updated deps**

```bash
conda run -n graph_id pip install orjson "huggingface-hub>=0.35.3"
```

Expected: `Successfully installed ...` or `Requirement already satisfied`

- [ ] **Step 3: Run full test suite to verify no regressions**

```bash
conda run -n graph_id python -m pytest -n auto -v
```

Expected: all previously passing tests still pass, plus the 3 new finder tests

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml
git commit -m "Add orjson and huggingface-hub as required dependencies"
```

---

### Task 3: Re-export `Finder` from `graph_id`

**Files:**
- Modify: `graph_id/__init__.py`

**Interfaces:**
- Consumes: `Finder` from `graph_id.db.finder`
- Produces: `from graph_id import Finder` works

- [ ] **Step 1: Write the failing test**

Add to `tests/py/test_finder.py`:

```python
def test_finder_importable_from_graph_id():
    from graph_id import Finder
    assert Finder is not None
```

- [ ] **Step 2: Run test to verify it fails**

```bash
conda run -n graph_id python -m pytest tests/py/test_finder.py::test_finder_importable_from_graph_id -v
```

Expected: `ImportError: cannot import name 'Finder' from 'graph_id'`

- [ ] **Step 3: Update `graph_id/__init__.py`**

```python
"""Graph ID: A library for generating unique identifiers for crystal structures."""

from graph_id.app.maker import GraphIDMaker
from graph_id.core.graph_id import GraphIDGenerator
from graph_id.db.finder import Finder
```

- [ ] **Step 4: Run all tests to verify**

```bash
conda run -n graph_id python -m pytest -n auto -v
```

Expected: all tests pass

- [ ] **Step 5: Commit**

```bash
git add graph_id/__init__.py tests/py/test_finder.py
git commit -m "Re-export Finder from graph_id"
```
