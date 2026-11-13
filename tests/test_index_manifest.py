"""
Embedding/index compatibility (spec items 8-11): duplicates, vector-size
mismatches, model or chunk-setting changes, and safe re-indexing from the
stored PDFs. Temp folders, fake embeddings and real (tiny) PDFs only.
"""
import hashlib
import sys
from pathlib import Path

import fitz
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from rag import document_registry, reindex, settings, vector_store
from rag.text_chunker import chunk_document


@pytest.fixture(autouse=True)
def temp_index(tmp_path, monkeypatch):
    monkeypatch.setattr(vector_store, "STORE_DIR", tmp_path / "index")
    monkeypatch.setattr(vector_store, "EMBEDDINGS_FILE", tmp_path / "index" / "embeddings.npy")
    monkeypatch.setattr(vector_store, "METADATA_FILE", tmp_path / "index" / "metadata.json")
    monkeypatch.setattr(document_registry, "STORE_DIR", tmp_path / "index")
    monkeypatch.setattr(document_registry, "REGISTRY_FILE", tmp_path / "index" / "documents.json")
    return tmp_path


def fake_embed(texts, dim=384):
    """Deterministic vectors from the text, so no model is loaded."""
    out = []
    for t in texts:
        seed = int(hashlib.sha256(t.encode()).hexdigest()[:8], 16)
        out.append(np.random.default_rng(seed).random(dim).tolist())
    return out


def _chunks(doc_id="d1", text="Revenue grew 12.1% in 2024. " * 5):
    pages = [{"doc_name": "r.pdf", "page": 1, "text": text, "blocks": [{"lines": [text], "size": 10, "bold": False}]}]
    chunks = chunk_document(pages, settings.CHUNK_STRATEGY, settings.CHUNK_SIZE, settings.CHUNK_OVERLAP)
    for c in chunks:
        c["doc_id"] = doc_id
    return chunks


def _index_files(tmp):
    return {p.name: p.read_bytes() for p in (tmp / "index").iterdir()}


# 8. Duplicates
def test_same_document_added_twice_is_stored_once():
    chunks = _chunks()
    assert vector_store.add_chunks(chunks, fake_embed([c["text"] for c in chunks])) == len(chunks)
    assert vector_store.add_chunks(chunks, fake_embed([c["text"] for c in chunks])) == 0


# 9. Vector-size mismatches
def test_vectors_of_a_different_size_are_refused():
    chunks = _chunks()
    vector_store.add_chunks(chunks, fake_embed([c["text"] for c in chunks]))
    other = _chunks("d2", "Different text entirely. " * 5)
    with pytest.raises(vector_store.IndexOutdatedError):
        vector_store.add_chunks(other, fake_embed([c["text"] for c in other], dim=512))


def test_vectors_disagreeing_with_their_manifest_count_as_damaged(temp_index):
    chunks = _chunks()
    vector_store.add_chunks(chunks, fake_embed([c["text"] for c in chunks]))
    manifest = temp_index / "index" / "manifest.json"
    manifest.write_text(manifest.read_text().replace('"dimension": 384', '"dimension": 768'))
    with pytest.raises(vector_store.IndexDamagedError):
        vector_store.check_compatible()


# 10. Model and settings changes
def test_changed_embedding_model_makes_the_index_outdated_and_search_refuses(monkeypatch):
    chunks = _chunks()
    vector_store.add_chunks(chunks, fake_embed([c["text"] for c in chunks]))
    vector_store.check_compatible()  # fine with today's settings
    monkeypatch.setattr(settings, "EMBEDDING_MODEL", "some-other-model")
    with pytest.raises(vector_store.IndexOutdatedError, match="embedding_model"):
        vector_store.check_compatible()
    with pytest.raises(vector_store.IndexOutdatedError):
        vector_store.search(fake_embed(["q"])[0], allowed_doc_ids={"d1"})


def test_changed_chunk_settings_make_the_index_outdated(monkeypatch):
    chunks = _chunks()
    vector_store.add_chunks(chunks, fake_embed([c["text"] for c in chunks]))
    monkeypatch.setattr(settings, "CHUNK_SIZE", settings.CHUNK_SIZE + 50)
    with pytest.raises(vector_store.IndexOutdatedError, match="chunking"):
        vector_store.check_compatible()


def test_index_without_a_manifest_is_outdated(temp_index):
    chunks = _chunks()
    vector_store.add_chunks(chunks, fake_embed([c["text"] for c in chunks]))
    (temp_index / "index" / "manifest.json").unlink()  # an index built before this change
    with pytest.raises(vector_store.IndexOutdatedError):
        vector_store.check_compatible()


# 11. Safe re-indexing from the stored PDFs
def _stored_pdf(upload_dir: Path, name: str, text: str) -> str:
    upload_dir.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(); doc.new_page().insert_text((50, 100), text)
    stored = upload_dir / f"{hashlib.sha256(name.encode()).hexdigest()[:16]}.pdf"
    doc.save(str(stored)); doc.close()
    doc_id = hashlib.sha256(stored.read_bytes()).hexdigest()
    document_registry.add(doc_id, name, "public", stored.name)
    return doc_id


def test_reindex_rebuilds_from_stored_files_and_reports_missing_ones(temp_index, monkeypatch):
    uploads = temp_index / "uploads"
    _stored_pdf(uploads, "kept.pdf", "Revenue grew 12.1 percent in 2024.")
    gone = _stored_pdf(uploads, "gone.pdf", "This file will disappear.")
    (uploads / Path(document_registry.get(gone)["stored_path"]).name).unlink()

    monkeypatch.setattr(settings, "CHUNK_STRATEGY", "paragraph")  # settings changed since upload
    report = reindex.reindex_all(uploads, embed=fake_embed)

    assert report["documents"] == 1 and report["missing"] == ["gone.pdf"]
    assert document_registry.get(gone) is not None, "a missing file is reported, never silently dropped"
    vector_store.check_compatible()  # the new index matches today's settings
    assert vector_store.read_manifest()["chunking"]["strategy"] == "paragraph"


def test_reindex_reuses_vectors_for_unchanged_text(temp_index):
    uploads = temp_index / "uploads"
    _stored_pdf(uploads, "a.pdf", "Revenue grew 12.1 percent in 2024.")
    first = reindex.reindex_all(uploads, embed=fake_embed)
    second = reindex.reindex_all(uploads, embed=fake_embed)
    assert first["embedded"] == first["chunks"] > 0
    assert second["embedded"] == 0 and second["reused"] == second["chunks"]


def test_failed_reindex_leaves_the_old_index_untouched(temp_index):
    uploads = temp_index / "uploads"
    _stored_pdf(uploads, "a.pdf", "Revenue grew 12.1 percent in 2024.")
    reindex.reindex_all(uploads, embed=fake_embed)
    _stored_pdf(uploads, "b.pdf", "A second document with new text.")  # its text needs embedding
    before = _index_files(temp_index)

    def broken_embed(texts):
        raise RuntimeError("model crashed mid-way")

    with pytest.raises(RuntimeError):
        reindex.reindex_all(uploads, embed=broken_embed)
    assert _index_files(temp_index) == before
