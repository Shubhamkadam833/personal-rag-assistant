"""
Phase 4: the reranker and its place in retrieval. A fake model stands in for
the cross-encoder so these run in milliseconds; one test at the end uses the
real model and is skipped when it isn't downloaded.
"""
import logging
import math
import sys
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from rag import bm25, document_registry, reranker, retriever, settings, vector_store

QUERY = np.ones(384, dtype=np.float32)


class FakeModel:
    """Scores a pair by how many question words the chunk contains (or by `score_fn`)."""

    def __init__(self, score_fn=None):
        self.score_fn = score_fn or (lambda q, t: len(set(q.lower().split()) & set(t.lower().split())))
        self.seen_texts = []

    def predict(self, pairs, show_progress_bar=False):
        self.seen_texts += [t for _, t in pairs]
        return np.array([self.score_fn(q, t) for q, t in pairs], dtype=np.float32)


@pytest.fixture(autouse=True)
def temp_index(tmp_path, monkeypatch):
    monkeypatch.setattr(vector_store, "STORE_DIR", tmp_path)
    monkeypatch.setattr(vector_store, "EMBEDDINGS_FILE", tmp_path / "embeddings.npy")
    monkeypatch.setattr(vector_store, "METADATA_FILE", tmp_path / "metadata.json")
    monkeypatch.setattr(document_registry, "STORE_DIR", tmp_path)
    monkeypatch.setattr(document_registry, "REGISTRY_FILE", tmp_path / "documents.json")
    monkeypatch.setattr(retriever, "embed_query", lambda q: QUERY.tolist())
    monkeypatch.setattr(bm25, "_cache", None)
    monkeypatch.setattr(reranker, "_load_error", None)
    return tmp_path


@pytest.fixture
def fake(monkeypatch):
    model = FakeModel()
    monkeypatch.setattr(reranker, "_model", model)
    return model


def _cands(*texts, **extra):
    return [{"chunk_id": f"c{i}", "doc_id": "d", "doc_name": "d.pdf", "page": i, "pages": [i],
             "section": "", "text": t, **extra} for i, t in enumerate(texts, start=1)]


def _add(doc_id, texts, privacy="public"):
    chunks = [{"doc_name": f"{doc_id}.pdf", "page": i + 1, "text": t, "doc_id": doc_id} for i, t in enumerate(texts)]
    vector_store.add_chunks(chunks, [QUERY.tolist()] * len(texts))
    document_registry.add(doc_id, f"{doc_id}.pdf", privacy, f"{doc_id}.pdf")


# 1. Ranking ------------------------------------------------------------------

def test_most_relevant_chunk_moves_to_the_top(fake):
    out, info = reranker.rerank("northwind revenue 2024", _cands("unrelated text", "northwind revenue 2024 was high"))
    assert [c["chunk_id"] for c in out] == ["c2", "c1"]
    assert (out[0]["original_rank"], out[0]["final_rank"]) == (2, 1)
    assert info == {"reranked": True, "fallback": None, "seconds": info["seconds"], "candidates": 2}


# 2. Wrong company / wrong year (fake model that understands both) -------------

def test_right_company_and_year_beat_look_alikes(monkeypatch):
    def score(q, t):
        return 2 * ("Northwind" in t) + ("2024" in t)
    monkeypatch.setattr(reranker, "_model", FakeModel(score))
    out, _ = reranker.rerank("Northwind operating income 2024", _cands(
        "Northwind operating income 2023", "Cobalt operating income 2024", "Northwind operating income 2024"))
    assert out[0]["text"] == "Northwind operating income 2024"


# 3-5. Duplicates, empty, fewer than requested ----------------------------------

def test_duplicate_candidates_are_scored_once(fake):
    cands = _cands("a b", "c d")
    out, info = reranker.rerank("a", cands + [dict(cands[0])])
    assert info["candidates"] == 2
    assert sorted(c["original_rank"] for c in out) == [1, 2]


def test_empty_candidates_return_nothing_without_fallback(fake):
    assert reranker.rerank("q", []) == ([], {"reranked": False, "fallback": None, "seconds": 0.0, "candidates": 0})


def test_fewer_candidates_than_top_n(fake):
    out, _ = reranker.rerank("x", _cands("x", "y", "z"), top_n=5)
    assert len(out) == 3


# 6. Missing metadata --------------------------------------------------------

def test_missing_optional_fields_pass_through_untouched(fake):
    out, info = reranker.rerank("x", [{"chunk_id": "only", "text": "x"}])
    assert info["reranked"] and out[0]["chunk_id"] == "only" and "page" not in out[0]


def test_candidate_without_text_falls_back(fake):
    cands = _cands("x", "y")
    del cands[1]["text"]
    out, info = reranker.rerank("x", cands)
    assert info["fallback"] == "a candidate has no text" and [c["chunk_id"] for c in out] == ["c1", "c2"]


# 7-8. Model failures ----------------------------------------------------------

def test_model_load_failure_falls_back_and_is_not_retried(monkeypatch):
    calls = []
    def broken():
        calls.append(1)
        raise OSError("offline")
    monkeypatch.setattr(reranker, "_model", None)
    monkeypatch.setattr(reranker, "_load_model", broken)
    for _ in range(2):
        out, info = reranker.rerank("x", _cands("a", "b"))
        assert info["fallback"] == "model unavailable"
        assert [c["chunk_id"] for c in out] == ["c1", "c2"] and out[0]["rerank_score"] is None
    assert len(calls) == 1, "a failed load is remembered, not retried on every question"


@pytest.mark.parametrize("bad", [[1.0], [1.0, math.nan], [1.0, math.inf], ["high", "low"], None])
def test_invalid_scores_fall_back(monkeypatch, bad):
    class Bad:
        def predict(self, pairs, show_progress_bar=False):
            return bad
    monkeypatch.setattr(reranker, "_model", Bad())
    out, info = reranker.rerank("x", _cands("a", "b"))
    assert info["fallback"] == "invalid scores" and [c["chunk_id"] for c in out] == ["c1", "c2"]


def test_model_error_falls_back(monkeypatch):
    class Crashes:
        def predict(self, pairs, show_progress_bar=False):
            raise MemoryError
    monkeypatch.setattr(reranker, "_model", Crashes())
    assert reranker.rerank("x", _cands("a"))[1]["fallback"] == "model error (MemoryError)"


def test_slow_model_times_out_and_falls_back(monkeypatch):
    class Slow:
        def predict(self, pairs, show_progress_bar=False):
            time.sleep(0.5)
            return [1.0] * len(pairs)
    monkeypatch.setattr(reranker, "_model", Slow())
    monkeypatch.setattr(settings, "RERANK_TIMEOUT_S", 0.05)
    start = time.perf_counter()
    out, info = reranker.rerank("x", _cands("a", "b"))
    assert info["fallback"].startswith("took longer than") and len(out) == 2
    assert time.perf_counter() - start < 0.4, "the app doesn't wait for a stuck model"


def test_fallback_logs_never_contain_document_text(monkeypatch, caplog):
    monkeypatch.setattr(reranker, "_model", None)
    monkeypatch.setattr(reranker, "_load_model", lambda: (_ for _ in ()).throw(OSError("offline")))
    with caplog.at_level(logging.DEBUG, logger="rag.reranker"):
        reranker.rerank("secret question", _cands("confidential merger terms"))
    assert caplog.records
    assert not any("confidential" in r.getMessage() or "secret" in r.getMessage() for r in caplog.records)


# 13-14. Determinism and citations --------------------------------------------

def test_tied_scores_keep_search_order(monkeypatch):
    monkeypatch.setattr(reranker, "_model", FakeModel(lambda q, t: 1.0))
    out, _ = reranker.rerank("x", _cands("a", "b", "c"))
    assert [c["chunk_id"] for c in out] == ["c1", "c2", "c3"]


def test_citation_fields_are_never_changed(fake):
    cands = _cands("x y", "x", section="Income")
    out, _ = reranker.rerank("x y", cands)
    for c in out:
        original = next(o for o in cands if o["chunk_id"] == c["chunk_id"])
        assert {k: c[k] for k in original} == original
    assert cands[0].keys() == _cands("x")[0].keys() | {"section"}, "the input list isn't modified"


# 9-12. Inside retrieval: privacy, deletion, on/off ----------------------------

@pytest.mark.parametrize("mode", ["semantic", "bm25", "hybrid"])
def test_reranker_never_sees_restricted_text(monkeypatch, mode):
    model = FakeModel(lambda q, t: 100.0 if "secret" in t else 1.0)  # would rank restricted first
    monkeypatch.setattr(reranker, "_model", model)
    _add("vault", ["northwind revenue secret plan"] * 3, privacy="restricted")
    _add("open", ["northwind revenue public summary"])
    chunks, info = retriever.retrieve_with_info("northwind revenue", mode=mode, rerank=True)
    assert info["reranked"]
    assert {c["doc_id"] for c in chunks} == {"open"}
    assert not any("secret" in t for t in model.seen_texts)


def test_fallback_keeps_privacy_filtering(monkeypatch):
    monkeypatch.setattr(reranker, "_model", None)
    monkeypatch.setattr(reranker, "_load_model", lambda: (_ for _ in ()).throw(OSError("offline")))
    _add("vault", ["northwind revenue secret"] * 3, privacy="restricted")
    _add("open", ["northwind revenue public"])
    chunks, info = retriever.retrieve_with_info("northwind revenue", rerank=True)
    assert info["rerank_fallback"] == "model unavailable"
    assert {c["doc_id"] for c in chunks} == {"open"}


def test_deleted_document_is_never_reranked_or_returned(fake):
    _add("old", ["northwind revenue old version"])
    _add("new", ["northwind revenue new version"])
    vector_store.remove_document("old")
    chunks, _ = retriever.retrieve_with_info("northwind revenue old", rerank=True)
    assert {c["doc_id"] for c in chunks} == {"new"}
    assert not any("old version" in t for t in fake.seen_texts)


def test_rerank_on_and_off_need_no_reindex(fake, temp_index):
    _add("d", [f"filler text {i}" for i in range(8)] + ["northwind revenue 2024 answer"])
    before = {p.name: p.stat().st_mtime_ns for p in temp_index.iterdir()}

    off, off_info = retriever.retrieve_with_info("northwind revenue 2024", top_k=3, mode="semantic", rerank=False)
    on, on_info = retriever.retrieve_with_info("northwind revenue 2024", top_k=3, mode="semantic", rerank=True)

    assert not off_info["reranked"] and "rerank_score" not in off[0]
    assert [c["final_rank"] for c in off] == [c["original_rank"] for c in off] == [1, 2, 3]
    assert on_info["reranked"] and on_info["candidates"] == 9
    assert on[0]["text"] == "northwind revenue 2024 answer", "found among 9 candidates, though not in the top 3 by search"
    assert {p.name: p.stat().st_mtime_ns for p in temp_index.iterdir()} == before


def test_rerank_default_follows_settings(fake, monkeypatch):
    _add("d", ["northwind revenue"])
    monkeypatch.setattr(settings, "RERANK_ENABLED", True)
    assert retriever.retrieve_with_info("northwind")[1]["reranked"]
    monkeypatch.setattr(settings, "RERANK_ENABLED", False)
    assert not retriever.retrieve_with_info("northwind")[1]["reranked"]


# Real model (skipped when not downloaded) --------------------------------------

@pytest.fixture(scope="module")
def real_model():
    from sentence_transformers import CrossEncoder
    try:
        return CrossEncoder(settings.RERANK_MODEL, revision=settings.RERANK_MODEL_REVISION,
                            max_length=settings.RERANK_MAX_LENGTH, local_files_only=True)
    except OSError:
        pytest.skip("reranker model not downloaded")


def test_real_model_prefers_right_company_and_year(real_model, monkeypatch):
    monkeypatch.setattr(reranker, "_model", real_model)
    out, info = reranker.rerank(
        "What was Northwind Robotics' research and development expense in fiscal year 2024?",
        _cands("Northwind Robotics recorded research and development expense of $38.2 million in fiscal year 2023.",
               "Cobalt Harbor Logistics recorded research and development expense of $12.4 million in fiscal year 2024.",
               "Northwind Robotics recorded research and development expense of $45.9 million in fiscal year 2024.",
               "Northwind Robotics reported gross profit of $171.4 million in fiscal year 2024."))
    assert info["reranked"]
    assert out[0]["text"].endswith("$45.9 million in fiscal year 2024.")
    assert "Cobalt" not in out[0]["text"]
