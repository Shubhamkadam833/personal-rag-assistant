"""
vector_store.py — stores and searches document chunks using numpy cosine similarity.

WHAT IS COSINE SIMILARITY?
  It measures how similar two vectors are by the angle between them (0 = identical, 1 = opposite).
  We use it to find chunks whose meaning is closest to the user's question.

HOW DATA IS STORED (on disk):
  data/chroma_db/embeddings.npy  — numpy matrix of all vectors (shape: N x 384)
  data/chroma_db/metadata.json   — list of {doc_name, page, text} for each vector

DUPLICATE PREVENTION:
  Each chunk gets a hash from doc_name + page + text. Existing hashes are skipped.

PERSISTENCE:
  Data is saved to disk after every add/remove, so it survives app restarts.
"""

import json
import hashlib
from pathlib import Path
import numpy as np

STORE_DIR = Path(__file__).parent.parent / "data" / "chroma_db"
EMBEDDINGS_FILE = STORE_DIR / "embeddings.npy"
METADATA_FILE = STORE_DIR / "metadata.json"


def _load() -> tuple[np.ndarray, list[dict]]:
    """Load embeddings matrix and metadata list from disk."""
    if EMBEDDINGS_FILE.exists() and METADATA_FILE.exists():
        embeddings = np.load(str(EMBEDDINGS_FILE))
        metadata = json.loads(METADATA_FILE.read_text(encoding="utf-8"))
        return embeddings, metadata
    return np.empty((0, 384), dtype=np.float32), []


def _save(embeddings: np.ndarray, metadata: list[dict]) -> None:
    STORE_DIR.mkdir(parents=True, exist_ok=True)
    np.save(str(EMBEDDINGS_FILE), embeddings)
    METADATA_FILE.write_text(json.dumps(metadata, ensure_ascii=False), encoding="utf-8")


def _chunk_hash(doc_name: str, page: int, text: str) -> str:
    raw = f"{doc_name}::page{page}::{text[:100]}"
    return hashlib.md5(raw.encode()).hexdigest()


def add_chunks(chunks: list[dict], embeddings: list[list[float]]) -> int:
    """
    Store chunks and their embeddings.
    Returns the number of NEW chunks added (duplicates are skipped).
    """
    existing_vecs, existing_meta = _load()
    existing_hashes = {m["hash"] for m in existing_meta}

    new_vecs, new_meta = [], []
    for chunk, vec in zip(chunks, embeddings):
        h = _chunk_hash(chunk["doc_name"], chunk["page"], chunk["text"])
        if h not in existing_hashes:
            existing_hashes.add(h)
            new_vecs.append(vec)
            new_meta.append({
                "hash": h,
                "doc_name": chunk["doc_name"],
                "page": chunk["page"],
                "text": chunk["text"],
            })

    if new_vecs:
        new_matrix = np.array(new_vecs, dtype=np.float32)
        combined = np.vstack([existing_vecs, new_matrix]) if existing_vecs.shape[0] > 0 else new_matrix
        _save(combined, existing_meta + new_meta)

    return len(new_vecs)


def search(query_embedding: list[float], top_k: int = 5) -> list[dict]:
    """
    Find the top_k most relevant chunks using cosine similarity.
    Returns list of dicts with text, doc_name, page, distance.
    """
    embeddings, metadata = _load()
    if embeddings.shape[0] == 0:
        return []

    query = np.array(query_embedding, dtype=np.float32)

    # Cosine similarity: dot product of normalized vectors
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    normed = embeddings / norms

    query_norm = query / (np.linalg.norm(query) or 1)
    similarities = normed @ query_norm  # shape: (N,)

    top_indices = np.argsort(similarities)[::-1][:top_k]

    results = []
    for idx in top_indices:
        m = metadata[idx]
        results.append({
            "text": m["text"],
            "doc_name": m["doc_name"],
            "page": m["page"],
            "distance": float(1 - similarities[idx]),
        })
    return results


def list_documents() -> list[str]:
    """Return unique document names currently stored."""
    _, metadata = _load()
    return sorted(set(m["doc_name"] for m in metadata))


def remove_document(doc_name: str) -> int:
    """Delete all chunks belonging to a document. Returns count deleted."""
    embeddings, metadata = _load()
    keep_indices = [i for i, m in enumerate(metadata) if m["doc_name"] != doc_name]
    removed = len(metadata) - len(keep_indices)

    if removed > 0:
        new_embeddings = embeddings[keep_indices] if keep_indices else np.empty((0, 384), dtype=np.float32)
        new_metadata = [metadata[i] for i in keep_indices]
        _save(new_embeddings, new_metadata)

    return removed


def total_chunks() -> int:
    _, metadata = _load()
    return len(metadata)
