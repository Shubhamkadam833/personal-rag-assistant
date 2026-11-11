"""
retrieve() must never hand Restricted or unclassified (legacy) chunks to the
answer model, and Restricted chunks must not push allowed ones out of the
results. Fixed vectors stand in for the embedding model, so no model loads.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from rag import document_registry, retriever, vector_store

QUERY = np.ones(384, dtype=np.float32)
NEAR = QUERY.copy()                    # identical to the query: best possible match
FAR = QUERY.copy(); FAR[:192] = 0.0    # still related, clearly ranked lower


@pytest.fixture(autouse=True)
def temp_index(tmp_path, monkeypatch):
    monkeypatch.setattr(vector_store, "STORE_DIR", tmp_path)
    monkeypatch.setattr(vector_store, "EMBEDDINGS_FILE", tmp_path / "embeddings.npy")
    monkeypatch.setattr(vector_store, "METADATA_FILE", tmp_path / "metadata.json")
    monkeypatch.setattr(document_registry, "STORE_DIR", tmp_path)
    monkeypatch.setattr(document_registry, "REGISTRY_FILE", tmp_path / "documents.json")
    monkeypatch.setattr(retriever, "embed_query", lambda q: QUERY.tolist())


def _add(doc_id, privacy, vector, n=1):
    chunks = [{"doc_name": f"{doc_id}.pdf", "page": i + 1, "text": f"{doc_id} chunk {i}", "doc_id": doc_id}
              for i in range(n)]
    vector_store.add_chunks(chunks, [vector.tolist()] * n)
    if privacy:
        document_registry.add(doc_id, f"{doc_id}.pdf", privacy, f"{doc_id}.pdf")


def test_restricted_chunks_are_never_returned():
    _add("secret", "restricted", NEAR)
    _add("open", "public", FAR)
    results = retriever.retrieve("anything", top_k=5)
    assert [c["doc_id"] for c in results] == ["open"]


def test_restricted_document_cannot_crowd_out_public_answers():
    # The audit's case: 20 Restricted chunks rank above the only Public one.
    _add("secret", "restricted", NEAR, n=20)
    _add("open", "public", FAR)
    results = retriever.retrieve("anything", top_k=5)
    assert [c["doc_id"] for c in results] == ["open"]


def test_unclassified_legacy_chunks_are_never_returned():
    _add("legacy", None, NEAR)            # chunk exists, no registry record
    vector_store.add_chunks([{"doc_name": "old.pdf", "page": 1, "text": "pre-ID chunk"}], [NEAR.tolist()])
    _add("open", "public", FAR)
    assert [c["doc_id"] for c in retriever.retrieve("anything", top_k=5)] == ["open"]


def test_personal_chunks_come_back_tagged_for_the_approval_prompt():
    _add("mine", "personal", NEAR)
    results = retriever.retrieve("anything", top_k=5)
    assert [(c["doc_id"], c["privacy"]) for c in results] == [("mine", "personal")]


def test_nothing_allowed_returns_nothing():
    _add("secret", "restricted", NEAR)
    assert retriever.retrieve("anything", top_k=5) == []
