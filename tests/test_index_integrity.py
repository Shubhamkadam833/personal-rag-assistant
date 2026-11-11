"""
Index integrity: new versions of a file stay separate, a crash mid-save can't
lock the app, Rebuild works on a damaged index, and a leftover registry record
can't block re-uploading a file. Temp folders and fixed vectors only.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from rag import document_registry, vector_store

VEC = [[0.1] * 384]
HEAD = "Summary of results for the year. " * 4  # identical first 100+ characters


@pytest.fixture(autouse=True)
def temp_index(tmp_path, monkeypatch):
    monkeypatch.setattr(vector_store, "STORE_DIR", tmp_path)
    monkeypatch.setattr(vector_store, "EMBEDDINGS_FILE", tmp_path / "embeddings.npy")
    monkeypatch.setattr(vector_store, "METADATA_FILE", tmp_path / "metadata.json")
    monkeypatch.setattr(document_registry, "STORE_DIR", tmp_path)
    monkeypatch.setattr(document_registry, "REGISTRY_FILE", tmp_path / "documents.json")
    return tmp_path


def _chunk(doc_id, text, page=2):
    return {"doc_name": "report.pdf", "page": page, "doc_id": doc_id, "text": text}


# ── Versions ─────────────────────────────────────────────────────────────────

def test_new_version_with_the_same_name_keeps_its_own_numbers():
    assert vector_store.add_chunks([_chunk("v1", HEAD + "Total revenue 412.6")], VEC) == 1
    assert vector_store.add_chunks([_chunk("v2", HEAD + "Total revenue 413.1")], VEC) == 1
    vector_store.remove_document("v1")
    _, meta = vector_store._load()
    assert [m["text"][-5:] for m in meta] == ["413.1"], "deleting v1 must not take v2's text"


def test_adding_the_same_document_twice_is_still_deduplicated():
    chunks = [_chunk("v1", "alpha", page=1), _chunk("v1", "beta", page=1)]
    assert vector_store.add_chunks(chunks, VEC * 2) == 2
    assert vector_store.add_chunks(chunks, VEC * 2) == 0


# ── Damaged files ────────────────────────────────────────────────────────────

def test_half_written_metadata_reports_damage_instead_of_crashing(temp_index):
    vector_store.add_chunks([_chunk("v1", "alpha")], VEC)
    (temp_index / "metadata.json").write_text('[{"hash": "abc", "doc_na')
    for call in (vector_store.list_documents, vector_store.total_chunks):
        with pytest.raises(vector_store.IndexDamagedError):
            call()


def test_vectors_saved_without_matching_metadata_reports_damage(temp_index):
    vector_store.add_chunks([_chunk("v1", "alpha")], VEC)
    emb, _ = vector_store._load()
    np.save(str(temp_index / "embeddings.npy"), np.vstack([emb, emb]))  # crash between the two saves
    with pytest.raises(vector_store.IndexDamagedError):
        vector_store.total_chunks()


def test_rebuild_works_on_a_damaged_index(temp_index):
    vector_store.add_chunks([_chunk("v1", "alpha")], VEC)
    (temp_index / "metadata.json").write_text("not json")
    vector_store.clear_all()
    assert vector_store.total_chunks() == 0


def test_saving_leaves_no_temp_files(temp_index):
    vector_store.add_chunks([_chunk("v1", "alpha")], VEC)
    document_registry.add("v1", "report.pdf", "public", "x.pdf")
    assert sorted(p.name for p in temp_index.iterdir()) == ["documents.json", "embeddings.npy", "metadata.json"]


def test_damaged_registry_reports_damage_and_can_be_cleared(temp_index):
    (temp_index / "documents.json").write_text('{"v1": {"display_')
    with pytest.raises(vector_store.IndexDamagedError):
        document_registry.list_all()
    document_registry.clear_all()
    assert document_registry.list_all() == {}


# ── Re-upload after the index files were deleted ─────────────────────────────

def test_record_without_chunks_does_not_block_reupload(temp_index):
    vector_store.add_chunks([_chunk("v1", "alpha")], VEC)
    document_registry.add("v1", "report.pdf", "public", "x.pdf")
    assert document_registry.is_indexed("v1") is True

    (temp_index / "embeddings.npy").unlink()
    (temp_index / "metadata.json").unlink()  # e.g. deleted by hand to recover
    assert document_registry.is_indexed("v1") is False
    assert document_registry.get("v1") is None, "the stale record is dropped"


def test_unknown_document_is_not_indexed():
    assert document_registry.is_indexed("nope") is False
