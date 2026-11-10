"""
retriever.py — combines embeddings + vector store into one search function.

This is the bridge between the user's question and the stored documents.
It converts the question into an embedding, then finds matching chunks.

PRIVACY ENFORCEMENT (not just requested in the system prompt — enforced here):
  A chunk whose document is classified "restricted" is removed from the
  results before this function returns. generate_answer() never receives it,
  so it structurally cannot be sent to the Claude API, regardless of what any
  prompt says. A chunk with no doc_id at all (indexed before document privacy
  classification existed) is treated as unclassified and excluded the same
  way — fail closed rather than assume it is safe to send.

  "Personal" documents are NOT filtered here: that privacy level requires a
  user approval step, which only makes sense in the UI (app.py), not inside a
  pure search function.
"""

from rag.embeddings import embed_query
from rag.query_rewriter import condense_question
from rag import vector_store
from rag import document_registry


def _privacy_of(doc_id: str | None) -> str | None:
    """Looks up a document's privacy level. None for a missing/legacy doc_id."""
    if not doc_id:
        return None
    record = document_registry.get(doc_id)
    return record["privacy"] if record else None


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
    search_question = condense_question(question, history)
    query_vec = embed_query(search_question)

    # Overfetch: filtering after search can drop results below top_k, so ask
    # the vector store for more than we need before narrowing down.
    raw_results = vector_store.search(query_vec, top_k=max(top_k * 3, top_k))

    allowed = []
    for chunk in raw_results:
        privacy = _privacy_of(chunk.get("doc_id"))
        if privacy == "restricted" or privacy is None:
            continue
        chunk["privacy"] = privacy
        allowed.append(chunk)
        if len(allowed) == top_k:
            break

    return allowed
