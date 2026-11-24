"""
Phase 3: BM25 keyword search, Reciprocal Rank Fusion, and the three search
modes. Fixed vectors stand in for the embedding model, so no model loads.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from rag import bm25, document_registry, retriever, vector_store

QUERY = np.ones(384, dtype=np.float32)


@pytest.fixture(autouse=True)
def temp_index(tmp_path, monkeypatch):
    monkeypatch.setattr(vector_store, "STORE_DIR", tmp_path)
    monkeypatch.setattr(vector_store, "EMBEDDINGS_FILE", tmp_path / "embeddings.npy")
    monkeypatch.setattr(vector_store, "METADATA_FILE", tmp_path / "metadata.json")
    monkeypatch.setattr(document_registry, "STORE_DIR", tmp_path)
    monkeypatch.setattr(document_registry, "REGISTRY_FILE", tmp_path / "documents.json")
    monkeypatch.setattr(retriever, "embed_query", lambda q: QUERY.tolist())
    monkeypatch.setattr(bm25, "_cache", None)
    return tmp_path


def _add(doc_id, texts, privacy="public", vectors=None):
    chunks = [{"doc_name": f"{doc_id}.pdf", "page": i + 1, "text": t, "doc_id": doc_id} for i, t in enumerate(texts)]
    vectors = vectors or [QUERY.tolist()] * len(texts)
    vector_store.add_chunks(chunks, vectors)
    document_registry.add(doc_id, f"{doc_id}.pdf", privacy, f"{doc_id}.pdf")


# --- BM25 -------------------------------------------------------------------

def test_tokenize_keeps_codes_and_decimals_and_drops_stopwords():
    assert bm25.tokenize("What was FY2024 revenue? It was 412.6 million.") == ["fy2024", "revenue", "412.6", "million"]


def test_exact_term_match_ranks_first():
    _add("a", ["The weather was mild and sunny all year.",
               "Northwind Robotics reported revenue of $412.6 million in FY2024.",
               "Revenue recognition policies are described in note 2."])
    results = bm25.search("Northwind FY2024 revenue", top_k=3, allowed_doc_ids={"a"})
    assert results[0]["page"] == 2
    assert all(r["bm25_score"] > 0 for r in results)
    assert 1 not in [r["page"] for r in results], "a chunk sharing no word is never returned"


def test_rare_words_outweigh_common_ones():
    _add("a", ["revenue revenue revenue grew", "revenue at Cobalt Harbor", "revenue fell", "revenue rose"])
    assert bm25.search("Cobalt revenue", top_k=1, allowed_doc_ids={"a"})[0]["page"] == 2


def test_empty_or_stopword_only_question_returns_nothing():
    _add("a", ["Some text about revenue."])
    assert bm25.search("", allowed_doc_ids={"a"}) == []
    assert bm25.search("what is the", allowed_doc_ids={"a"}) == []


def test_bm25_never_returns_or_counts_restricted_documents():
    _add("open", ["quarterly revenue summary"])
    _add("secret", ["secret merger codename Falcon revenue"] * 3, privacy="restricted")
    assert bm25.search("Falcon", allowed_doc_ids={"open"}) == []
    assert [r["doc_id"] for r in bm25.search("revenue Falcon", allowed_doc_ids={"open"})] == ["open"]


def test_bm25_follows_document_removal():
    _add("a", ["Northwind revenue"])
    _add("b", ["Northwind revenue again"])
    assert len(bm25.search("Northwind", allowed_doc_ids={"a", "b"})) == 2
    vector_store.remove_document("b")
    assert [r["doc_id"] for r in bm25.search("Northwind", allowed_doc_ids={"a", "b"})] == ["a"]


def test_bm25_refuses_an_outdated_index(temp_index):
    _add("a", ["Northwind revenue"])
    (temp_index / "manifest.json").unlink()
    with pytest.raises(vector_store.IndexOutdatedError):
        bm25.search("Northwind", allowed_doc_ids={"a"})


# --- RRF --------------------------------------------------------------------

def _c(cid):
    return {"chunk_id": cid, "text": cid}


def test_rrf_scores_match_hand_worked_numbers():
    fused = retriever.rrf_fuse([[_c("x"), _c("y")], [_c("y"), _c("z")]], k=60)
    assert [c["chunk_id"] for c in fused] == ["y", "x", "z"]
    assert fused[0]["rrf_score"] == round(1 / 62 + 1 / 61, 6)  # y: rank 2 in list 1, rank 1 in list 2
    assert fused[1]["rrf_score"] == round(1 / 61, 6)


def test_rrf_dedupes_and_keeps_fields_from_both_lists():
    fused = retriever.rrf_fuse([[{**_c("x"), "distance": 0.1}], [{**_c("x"), "bm25_score": 3.2}]])
    assert len(fused) == 1 and fused[0]["distance"] == 0.1 and fused[0]["bm25_score"] == 3.2


def test_rrf_ties_are_deterministic():
    assert [c["chunk_id"] for c in retriever.rrf_fuse([[_c("a")], [_c("b")]])] == ["a", "b"]
    assert retriever.rrf_fuse([]) == []


# --- search modes -----------------------------------------------------------

def test_all_three_modes_return_results_with_privacy():
    _add("a", ["Northwind revenue FY2024", "Lumen Grid capital expenditure"])
    for mode in ("semantic", "bm25", "hybrid"):
        results = retriever.retrieve("Northwind revenue", top_k=2, mode=mode)
        assert results, mode
        assert all(r["privacy"] == "public" for r in results)


def test_hybrid_finds_a_keyword_match_that_semantic_ranks_low():
    far = QUERY.copy(); far[:300] = 0.0
    _add("a", ["general overview text"] * 5 + ["invoice code ZX-4471 approved"],
         vectors=[QUERY.tolist()] * 5 + [far.tolist()])
    semantic = retriever.retrieve("ZX-4471", top_k=3, mode="semantic")
    hybrid = retriever.retrieve("ZX-4471", top_k=3, mode="hybrid")
    assert "ZX-4471" not in " ".join(r["text"] for r in semantic)
    assert "ZX-4471" in hybrid[0]["text"]


@pytest.mark.parametrize("mode", ["semantic", "bm25", "hybrid"])
def test_no_mode_returns_restricted_chunks(mode):
    _add("secret", ["Northwind revenue secret"] * 4, privacy="restricted")
    _add("open", ["Northwind revenue public"])
    assert {r["doc_id"] for r in retriever.retrieve("Northwind revenue", top_k=5, mode=mode)} == {"open"}


def test_unknown_mode_is_rejected():
    _add("a", ["text"])
    with pytest.raises(ValueError):
        retriever.retrieve("text", mode="magic")
