"""
retriever.py — combines embeddings + vector store into one search function.

This is the bridge between the user's question and the stored documents.
It converts the question into an embedding, then finds matching chunks.

PRIVACY ENFORCEMENT (not just requested in the system prompt — enforced here):
  Chunks from "restricted" documents are never ranked: the search only looks
  at registered Public and Personal documents. generate_answer() never
  receives a Restricted chunk, so it structurally cannot be sent to the Claude
  API, regardless of what any prompt says. Filtering before ranking (not after)
  also means a large Restricted document can't push allowed answers out of the
  results. A chunk with no registry record (indexed before privacy
  classification existed) is excluded the same way — fail closed rather than
  assume it is safe to send.

  "Personal" documents are NOT filtered here: that privacy level requires a
  user approval step, which only makes sense in the UI (app.py), not inside a
  pure search function.
"""

from rag.embeddings import embed_query
from rag.query_rewriter import condense_question
from rag import vector_store
from rag import document_registry


def retrieve(
    question: str,
    top_k: int = 5,
    history: list[dict] | None = None,
) -> list[dict]:
    """
    Given a question, return the top_k most relevant document chunks.
    Each result has: text, doc_id, doc_name, page, distance, privacy.

    When `history` is given, the question is first rewritten to stand alone so
    that follow-ups like "what about the prior year?" search for the right thing.

    Restricted and unclassified (legacy) chunks are never returned — see the
    module docstring.
    """
    # Only registered Public and Personal documents may be ranked at all.
    # Restricted ones and chunks with no registry record (legacy) are left out
    # BEFORE ranking, so they can't push allowed results out of the top_k.
    privacy = {doc_id: rec["privacy"] for doc_id, rec in document_registry.list_all().items()}
    allowed_ids = {doc_id for doc_id, level in privacy.items() if level in ("public", "personal")}
    if not allowed_ids:
        return []

    search_question = condense_question(question, history)
    results = vector_store.search(embed_query(search_question), top_k=top_k, allowed_doc_ids=allowed_ids)
    for chunk in results:
        chunk["privacy"] = privacy[chunk["doc_id"]]
    return results
