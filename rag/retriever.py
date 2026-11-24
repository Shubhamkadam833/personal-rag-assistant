"""
retriever.py — turns a question into the most relevant document chunks.

SEARCH MODES (rag/settings.py SEARCH_MODE, or the `mode` argument):
  semantic — embeddings + cosine similarity: finds chunks with the same
             MEANING, even in different words.
  bm25     — keyword scoring (rag/bm25.py): finds chunks with the same
             WORDS, good for names, codes and numbers.
  hybrid   — runs both and merges them with Reciprocal Rank Fusion (RRF).

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

from rag import bm25, document_registry, settings, vector_store
from rag.embeddings import embed_query
from rag.query_rewriter import condense_question


def retrieve(
    question: str,
    top_k: int = 5,
    history: list[dict] | None = None,
    mode: str | None = None,
) -> list[dict]:
    """
    Given a question, return the top_k most relevant document chunks.
    Each result has: text, doc_id, doc_name, page, pages, section, chunk_id,
    privacy, plus the score of the method that found it.

    When `history` is given, the question is first rewritten to stand alone so
    that follow-ups like "what about the prior year?" search for the right thing.

    Restricted and unclassified (legacy) chunks are never returned — see the
    module docstring.
    """
    mode = mode or settings.SEARCH_MODE
    if mode not in settings.SEARCH_MODES:
        raise ValueError(f"unknown search mode {mode!r}; use one of {settings.SEARCH_MODES}")

    # Only registered Public and Personal documents may be ranked at all.
    privacy = {doc_id: rec["privacy"] for doc_id, rec in document_registry.list_all().items()}
    allowed_ids = {doc_id for doc_id, level in privacy.items() if level in ("public", "personal")}
    if not allowed_ids:
        return []

    search_question = condense_question(question, history)
    if mode == "semantic":
        results = vector_store.search(embed_query(search_question), top_k=top_k, allowed_doc_ids=allowed_ids)
    elif mode == "bm25":
        results = bm25.search(search_question, top_k=top_k, allowed_doc_ids=allowed_ids)
    else:
        depth = max(top_k, settings.HYBRID_DEPTH)
        results = rrf_fuse([
            vector_store.search(embed_query(search_question), top_k=depth, allowed_doc_ids=allowed_ids),
            bm25.search(search_question, top_k=depth, allowed_doc_ids=allowed_ids),
        ])[:top_k]

    for chunk in results:
        chunk["privacy"] = privacy[chunk["doc_id"]]
    return results


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
