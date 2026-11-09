"""
retriever.py — combines embeddings + vector store into one search function.

This is the bridge between the user's question and the stored documents.
It converts the question into an embedding, then finds matching chunks.
"""

from rag.embeddings import embed_query
from rag import vector_store


def retrieve(question: str, top_k: int = 5) -> list[dict]:
    """
    Given a question, return the top_k most relevant document chunks.
    Each result has: text, doc_name, page, distance.
    """
    query_vec = embed_query(question)
    return vector_store.search(query_vec, top_k=top_k)
