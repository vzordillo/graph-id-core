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


def test_finder_importable_from_graph_id():
    from graph_id import Finder
    assert Finder is not None
