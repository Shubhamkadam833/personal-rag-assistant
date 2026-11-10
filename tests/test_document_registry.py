"""Tests for rag/document_registry.py — document identity and privacy records."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from rag import document_registry


@pytest.fixture(autouse=True)
def isolated_registry(tmp_path, monkeypatch):
    """Point the registry at a throwaway file so tests never touch real data."""
    monkeypatch.setattr(document_registry, "STORE_DIR", tmp_path)
    monkeypatch.setattr(document_registry, "REGISTRY_FILE", tmp_path / "documents.json")
    yield


def test_add_and_get():
    document_registry.add("abc123", "report.pdf", "public", "random1.pdf")
    record = document_registry.get("abc123")
    assert record["display_name"] == "report.pdf"
    assert record["privacy"] == "public"
    assert record["stored_path"] == "random1.pdf"
    assert record["flagged"] is False


def test_get_missing_returns_none():
    assert document_registry.get("does-not-exist") is None


def test_find_by_hash_detects_duplicate():
    document_registry.add("abc123", "report.pdf", "personal", "random1.pdf")
    assert document_registry.find_by_hash("abc123") == "abc123"
    assert document_registry.find_by_hash("different-hash") is None


def test_rejects_unknown_privacy_level():
    with pytest.raises(ValueError):
        document_registry.add("abc123", "report.pdf", "top-secret", "random1.pdf")


def test_remove_deletes_and_returns_record():
    document_registry.add("abc123", "report.pdf", "restricted", "random1.pdf")
    removed = document_registry.remove("abc123")
    assert removed["display_name"] == "report.pdf"
    assert document_registry.get("abc123") is None


def test_remove_missing_returns_none():
    assert document_registry.remove("does-not-exist") is None


def test_list_all_returns_every_record():
    document_registry.add("a", "one.pdf", "public", "p1.pdf")
    document_registry.add("b", "two.pdf", "personal", "p2.pdf")
    all_docs = document_registry.list_all()
    assert set(all_docs.keys()) == {"a", "b"}


def test_clear_all_empties_registry_and_reports_count():
    document_registry.add("a", "one.pdf", "public", "p1.pdf")
    document_registry.add("b", "two.pdf", "personal", "p2.pdf")
    assert document_registry.clear_all() == 2
    assert document_registry.list_all() == {}


def test_flagged_defaults_false_but_can_be_set():
    document_registry.add("a", "clean.pdf", "public", "p1.pdf")
    document_registry.add("b", "sus.pdf", "public", "p2.pdf", flagged=True)
    assert document_registry.get("a")["flagged"] is False
    assert document_registry.get("b")["flagged"] is True


def test_persists_across_reload(tmp_path, monkeypatch):
    document_registry.add("a", "one.pdf", "public", "p1.pdf")
    # A second "process" reading the same file sees the same data.
    assert document_registry.list_all()["a"]["display_name"] == "one.pdf"
