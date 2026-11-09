"""
vector_store.py — stores and retrieves document chunks using ChromaDB.

WHAT IS A VECTOR DATABASE?
  It stores text chunks alongside their embeddings (numerical representations).
  When you search, it finds chunks whose embeddings are closest to your question's
  embedding — this is called "similarity search".

DUPLICATE PREVENTION:
  Each chunk gets a unique ID built from doc_name + page + position.
  If you upload the same PDF twice, ChromaDB skips the existing IDs.

PERSISTENCE:
  Data is saved to data/chroma_db/ on disk, so it survives app restarts.
"""

from pathlib import Path
import hashlib
import chromadb
from chromadb.config import Settings


CHROMA_PATH = Path(__file__).parent.parent / "data" / "chroma_db"
COLLECTION_NAME = "documents"


def _get_collection():
    client = chromadb.Client(
        Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=str(CHROMA_PATH),
            anonymized_telemetry=False,
        )
    )
    return client.get_or_create_collection(name=COLLECTION_NAME)


def _chunk_id(doc_name: str, page: int, position: int) -> str:
    raw = f"{doc_name}::page{page}::pos{position}"
    return hashlib.md5(raw.encode()).hexdigest()


def add_chunks(chunks: list[dict], embeddings: list[list[float]]) -> int:
    """
    Store chunks and their embeddings in ChromaDB.
    Returns the number of NEW chunks added (duplicates are skipped).
    """
    collection = _get_collection()

    ids = []
    docs = []
    metas = []
    vecs = []

    for i, (chunk, vec) in enumerate(zip(chunks, embeddings)):
        cid = _chunk_id(chunk["doc_name"], chunk["page"], i)
        ids.append(cid)
        docs.append(chunk["text"])
        metas.append({"doc_name": chunk["doc_name"], "page": chunk["page"]})
        vecs.append(vec)

    # Find which IDs already exist to avoid duplicates
    existing = set(collection.get(ids=ids)["ids"])
    new_ids, new_docs, new_metas, new_vecs = [], [], [], []

    for cid, doc, meta, vec in zip(ids, docs, metas, vecs):
        if cid not in existing:
            new_ids.append(cid)
            new_docs.append(doc)
            new_metas.append(meta)
            new_vecs.append(vec)

    if new_ids:
        collection.add(ids=new_ids, documents=new_docs, metadatas=new_metas, embeddings=new_vecs)
        collection.persist()

    return len(new_ids)


def search(query_embedding: list[float], top_k: int = 5) -> list[dict]:
    """
    Find the top_k most relevant chunks for a query.
    Returns list of dicts with text, doc_name, page, and similarity distance.
    """
    collection = _get_collection()

    if collection.count() == 0:
        return []

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    hits = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        hits.append({
            "text": doc,
            "doc_name": meta["doc_name"],
            "page": meta["page"],
            "distance": dist,
        })

    return hits


def list_documents() -> list[str]:
    """Return unique document names currently stored."""
    collection = _get_collection()
    if collection.count() == 0:
        return []
    all_metas = collection.get(include=["metadatas"])["metadatas"]
    return sorted(set(m["doc_name"] for m in all_metas))


def remove_document(doc_name: str) -> int:
    """Delete all chunks belonging to a document. Returns count deleted."""
    collection = _get_collection()
    result = collection.get(where={"doc_name": doc_name}, include=["metadatas"])
    ids_to_delete = result["ids"]
    if ids_to_delete:
        collection.delete(ids=ids_to_delete)
        collection.persist()
    return len(ids_to_delete)


def total_chunks() -> int:
    return _get_collection().count()
