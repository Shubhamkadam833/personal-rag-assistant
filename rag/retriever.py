"""
retriever.py — combines embeddings + vector store into one search function.

This is the bridge between the user's question and the stored documents.
It converts the question into an embedding, then finds matching chunks.
"""

from rag.embeddings import embed_query
from rag.query_rewriter import condense_question
from rag import vector_store


def retrieve(
    question: str,
    top_k: int = 5,
    history: list[dict] | None = None,
) -> list[dict]:
    """
    Given a question, return the top_k most relevant document chunks.
    Each result has: text, doc_name, page, distance.

    When `history` is given, the question is first rewritten to stand alone so
    that follow-ups like "what about the prior year?" search for the right thing.
    """
    search_question = condense_question(question, history)
    query_vec = embed_query(search_question)
    return vector_store.search(query_vec, top_k=top_k)
