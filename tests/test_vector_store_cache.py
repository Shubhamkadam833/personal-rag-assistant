"""
The index is parsed once and reused until its files change, instead of being
re-read about three times on every click. Temp folders and fixed vectors only.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from rag import vector_store

VEC = [[0.1] * 384]


def _use(folder: Path, monkeypatch) -> None:
    monkeypatch.setattr(vector_store, "STORE_DIR", folder)
    monkeypatch.setattr(vector_store, "EMBEDDINGS_FILE", folder / "embeddings.npy")
    monkeypatch.setattr(vector_store, "METADATA_FILE", folder / "metadata.json")


def _add(doc_id: str, text: str) -> None:
    vector_store.add_chunks([{"doc_name": f"{doc_id}.pdf", "page": 1, "text": text, "doc_id": doc_id}], VEC)


@pytest.fixture(autouse=True)
def temp_index(tmp_path, monkeypatch):
    _use(tmp_path, monkeypatch)


def test_repeated_loads_reuse_the_parsed_index():
    _add("a", "alpha")
    assert vector_store._load()[1] is vector_store._load()[1]


def test_every_save_is_seen_immediately():
    _add("a", "alpha")
    assert vector_store.total_chunks() == 1
    _add("b", "beta")
    assert vector_store.total_chunks() == 2
    vector_store.remove_document("a")
    assert [d["doc_id"] for d in vector_store.list_documents()] == ["b"]
    vector_store.clear_all()
    assert vector_store.total_chunks() == 0


def test_two_folders_never_share_a_cached_index(tmp_path, monkeypatch):
    first, second = tmp_path / "one", tmp_path / "two"
    _use(first, monkeypatch); _add("a", "alpha")
    _use(second, monkeypatch); _add("b", "beta")
    assert [m["text"] for m in vector_store._load()[1]] == ["beta"]
    _use(first, monkeypatch)
    assert [m["text"] for m in vector_store._load()[1]] == ["alpha"]
