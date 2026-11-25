"""
retriever.py — turns a question into the most relevant document chunks.

SEARCH MODES (rag/settings.py SEARCH_MODE, or the `mode` argument):
  semantic — embeddings + cosine similarity: finds chunks with the same
             MEANING, even in different words.
  bm25     — keyword scoring (rag/bm25.py): finds chunks with the same
             WORDS, good for names, codes and numbers.
  hybrid   — runs both and merges them with Reciprocal Rank Fusion (RRF).

RERANKING (optional, rag/reranker.py):
  Search fetches RERANK_CANDIDATES results instead of top_k; a local
  cross-encoder re-reads each next to the question and keeps the best top_k.
  If it fails, the search order is used and the reason is reported.

RECIPROCAL RANK FUSION:
  Each method's list gives a chunk 1 / (RRF_K + rank) points; points are
  summed. A chunk ranked well by both methods beats one ranked first by only
  one. It uses positions, not raw scores, because a cosine similarity and a
  BM25 score are on different scales and can't be compared directly.

PRIVACY ENFORCEMENT (not just requested in the system prompt — enforced here):
  Chunks from "restricted" documents are never ranked, in any mode: every
  search only looks at registered Public and Personal documents.
  generate_answer() never receives a Restricted chunk, so it structurally
  cannot be sent to the Claude API, regardless of what any prompt says.
  Filtering before ranking (not after) also means a large Restricted document
  can't push allowed answers out of the results. A chunk with no registry
  record (indexed before privacy classification existed) is excluded the same
  way — fail closed rather than assume it is safe to send.

  "Personal" documents are NOT filtered here: that privacy level requires a
  user approval step, which only makes sense in the UI (app.py), not inside a
  pure search function.
"""

import time

from rag import bm25, document_registry, reranker, settings, vector_store
from rag.embeddings import embed_query
from rag.query_rewriter import condense_question


def retrieve(
    question: str,
    top_k: int = 5,
    history: list[dict] | None = None,
    mode: str | None = None,
    rerank: bool | None = None,
    candidates: int | None = None,
) -> list[dict]:
    """The chunks from retrieve_with_info(), without the info."""
    return retrieve_with_info(question, top_k, history, mode, rerank, candidates)[0]


def retrieve_with_info(
    question: str,
    top_k: int = 5,
    history: list[dict] | None = None,
    mode: str | None = None,
    rerank: bool | None = None,
    candidates: int | None = None,
) -> tuple[list[dict], dict]:
    """
    Given a question, return the top_k most relevant document chunks, and
    info about how they were found.

    Each chunk has: text, doc_id, doc_name, page, pages, section, chunk_id,
    privacy, original_rank, final_rank, the score of the method that found
    it, and rerank_score when reranked.
    info = {mode, reranked, rerank_fallback, candidates, retrieval_ms, rerank_ms}

    `mode`, `rerank` and `candidates` default to rag/settings.py. When
    `history` is given, the question is first rewritten to stand alone so that
    follow-ups like "what about the prior year?" search for the right thing.

    Restricted and unclassified (legacy) chunks are never returned or
    reranked — see the module docstring.
    """
    mode = mode or settings.SEARCH_MODE
    if mode not in settings.SEARCH_MODES:
        raise ValueError(f"unknown search mode {mode!r}; use one of {settings.SEARCH_MODES}")
    rerank = settings.RERANK_ENABLED if rerank is None else rerank
    candidates = settings.RERANK_CANDIDATES if candidates is None else candidates
    info = {"mode": mode, "reranked": False, "rerank_fallback": None, "candidates": 0,
            "retrieval_ms": 0.0, "rerank_ms": 0.0}

    # Only registered Public and Personal documents may be ranked at all.
    privacy = {doc_id: rec["privacy"] for doc_id, rec in document_registry.list_all().items()}
    allowed_ids = {doc_id for doc_id, level in privacy.items() if level in ("public", "personal")}
    if not allowed_ids:
        return [], info

    search_question = condense_question(question, history)
    start = time.perf_counter()
    results = search(search_question, max(top_k, candidates) if rerank else top_k, mode, allowed_ids)
    info.update(candidates=len(results), retrieval_ms=round((time.perf_counter() - start) * 1000, 1))

    if rerank:
        results, rinfo = reranker.rerank(search_question, results, top_k)
        info.update(reranked=rinfo["reranked"], rerank_fallback=rinfo["fallback"],
                    rerank_ms=round(rinfo["seconds"] * 1000, 1))
    else:
        results = [{**c, "original_rank": i, "final_rank": i} for i, c in enumerate(results[:top_k], start=1)]

    for chunk in results:
        chunk["privacy"] = privacy[chunk["doc_id"]]
    return results, info


def search(question: str, top_k: int, mode: str, allowed_ids: set[str]) -> list[dict]:
    """The top_k chunks of one search mode, among allowed documents only. No reranking."""
    if mode == "semantic":
        return vector_store.search(embed_query(question), top_k=top_k, allowed_doc_ids=allowed_ids)
    if mode == "bm25":
        return bm25.search(question, top_k=top_k, allowed_doc_ids=allowed_ids)
    depth = max(top_k, settings.HYBRID_DEPTH)
    return rrf_fuse([
        vector_store.search(embed_query(question), top_k=depth, allowed_doc_ids=allowed_ids),
        bm25.search(question, top_k=depth, allowed_doc_ids=allowed_ids),
    ])[:top_k]


def rrf_fuse(ranked_lists: list[list[dict]], k: int | None = None) -> list[dict]:
    """
    Merge ranked result lists into one, by Reciprocal Rank Fusion.
    A chunk found by several lists appears once, keeping every list's fields
    plus `rrf_score`. Ties keep first-seen order, so results are repeatable.
    """
    k = settings.RRF_K if k is None else k
    scores: dict[str, float] = {}
    merged: dict[str, dict] = {}
    for ranked in ranked_lists:
        for rank, chunk in enumerate(ranked, start=1):
            cid = chunk["chunk_id"]
            scores[cid] = scores.get(cid, 0.0) + 1 / (k + rank)
            merged[cid] = {**chunk, **merged.get(cid, {})}
    order = sorted(merged, key=lambda cid: -scores[cid])  # stable sort: ties stay in first-seen order
    return [{**merged[cid], "rrf_score": round(scores[cid], 6)} for cid in order]
