"""Shared fixtures."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from rag import document_registry, vector_store


@pytest.fixture
def index_doc(tmp_path, monkeypatch):
    """
    A temp index, and a function that adds a document to it and returns its
    chunks exactly as retrieval hands them to generate_answer (with chunk_id
    and privacy). Needed since generate_answer re-checks every excerpt against
    the live index before sending it.
    """
    monkeypatch.setattr(vector_store, "STORE_DIR", tmp_path)
    monkeypatch.setattr(vector_store, "EMBEDDINGS_FILE", tmp_path / "embeddings.npy")
    monkeypatch.setattr(vector_store, "METADATA_FILE", tmp_path / "metadata.json")
    monkeypatch.setattr(document_registry, "STORE_DIR", tmp_path)
    monkeypatch.setattr(document_registry, "REGISTRY_FILE", tmp_path / "documents.json")

    def add(doc_id: str, texts: list[str], privacy: str = "public") -> list[dict]:
        chunks = [{"doc_name": f"{doc_id}.pdf", "page": i + 1, "text": t, "doc_id": doc_id} for i, t in enumerate(texts)]
        vector_store.add_chunks(chunks, [np.ones(384, dtype=np.float32).tolist()] * len(chunks))
        document_registry.add(doc_id, f"{doc_id}.pdf", privacy, f"{doc_id}.pdf")
        _, metadata = vector_store._load()
        return [{**vector_store.result_fields(m), "privacy": privacy} for m in metadata if m["doc_id"] == doc_id]

    return add
