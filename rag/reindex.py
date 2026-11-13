"""
reindex.py — rebuilds the search index from the stored PDFs with today's
settings (rag/settings.py).

WHEN:
  After the embedding model or chunk settings change, the old index is
  "outdated" (vector_store.check_compatible) and search refuses it, so vectors
  from different setups are never mixed.

SAFE BY DESIGN:
  - Nothing is uploaded again and no file is deleted: it reads the PDFs the
    registry already tracks in data/uploads.
  - Everything is extracted (in the time-limited worker), chunked and
    embedded in memory first. The index files are replaced only at the end,
    so a failure part-way leaves the old index exactly as it was.
  - A document whose stored file is missing is reported and stays in the
    registry, so nothing disappears silently.
  - A chunk whose text is unchanged keeps its old vector when the model is the
    same, so only new or changed text is embedded again.
"""

import hashlib
from pathlib import Path

from rag import document_registry, settings, vector_store
from rag.embeddings import embed_texts
from rag.text_chunker import chunk_document
from security.pdf_sandbox import extract_pages_with_timeout
from security.prompt_guard import sanitize_chunk_text


def _key(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _reusable_vectors() -> dict[str, list[float]]:
    """Old vectors by text, only if they came from the model in use now."""
    try:
        manifest = vector_store.read_manifest()
        embeddings, metadata = vector_store._load()
    except vector_store.IndexDamagedError:
        return {}
    if not manifest or manifest.get("embedding_model") != settings.EMBEDDING_MODEL:
        return {}
    return {_key(m["text"]): vec.tolist() for m, vec in zip(metadata, embeddings)}


def reindex_all(upload_dir: Path, embed=embed_texts) -> dict:
    """
    Rebuild the index from the stored PDFs. Returns a report:
    {"documents": n, "chunks": n, "reused": n, "embedded": n,
     "missing": [names whose stored file is gone], "dropped_legacy": n}
    """
    reusable = _reusable_vectors()
    try:
        legacy = sum(1 for m in vector_store._load()[1] if not m.get("doc_id"))
    except vector_store.IndexDamagedError:
        legacy = 0

    chunks, missing, documents = [], [], 0
    for doc_id, record in document_registry.list_all().items():
        path = upload_dir / Path(record["stored_path"]).name
        if not path.exists():
            missing.append(record["display_name"])
            continue
        pages = extract_pages_with_timeout(path, record["display_name"])
        doc_chunks = chunk_document(pages, settings.CHUNK_STRATEGY, settings.CHUNK_SIZE, settings.CHUNK_OVERLAP)
        for chunk in doc_chunks:
            chunk["text"] = sanitize_chunk_text(chunk["text"])
            chunk["doc_id"] = doc_id
        chunks += doc_chunks
        documents += 1

    todo = [c["text"] for c in chunks if _key(c["text"]) not in reusable]
    fresh = dict(zip(map(_key, todo), embed(todo))) if todo else {}
    vectors = [reusable.get(_key(c["text"])) or fresh[_key(c["text"])] for c in chunks]

    # Only now are the index files touched.
    vector_store.replace_all(chunks, vectors)
    return {"documents": documents, "chunks": len(chunks), "reused": len(chunks) - len(todo),
            "embedded": len(todo), "missing": missing, "dropped_legacy": legacy}
