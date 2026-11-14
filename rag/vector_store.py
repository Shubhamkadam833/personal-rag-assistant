"""
vector_store.py — stores and searches document chunks using numpy cosine similarity.

WHAT IS COSINE SIMILARITY?
  It measures how similar two vectors are by the angle between them (0 = identical, 1 = opposite).
  We use it to find chunks whose meaning is closest to the user's question.

HOW DATA IS STORED (on disk):
  data/chroma_db/embeddings.npy  — numpy matrix of all vectors (shape: N x 384)
  data/chroma_db/metadata.json   — list of {doc_name, page, text} for each vector

DUPLICATE PREVENTION:
  Each chunk gets a SHA-256 hash of its document ID + page + position + full
  text; existing hashes are skipped, so adding the same document twice stores
  it once. Using the document ID (not the file name) keeps a new version of a
  same-named file separate from the old one. Whole-file duplicates are caught
  earlier, by the file hash in rag/document_registry.py.

PERSISTENCE:
  Data is saved to disk after every add/remove, so it survives app restarts.
  Each file is written to a temp file and swapped in whole. Unreadable or
  out-of-sync files raise IndexDamagedError instead of crashing the caller.
"""

import json
import hashlib
import os
from pathlib import Path
from typing import BinaryIO, Callable

import numpy as np

from rag import settings

STORE_DIR = Path(__file__).parent.parent / "data" / "chroma_db"
EMBEDDINGS_FILE = STORE_DIR / "embeddings.npy"
METADATA_FILE = STORE_DIR / "metadata.json"


def _manifest_file() -> Path:
    # Next to the metadata file, so pointing METADATA_FILE elsewhere (tests,
    # the benchmark) moves the manifest with it.
    return METADATA_FILE.with_name("manifest.json")


class IndexDamagedError(RuntimeError):
    """Index files exist but can't be trusted (unreadable, or out of sync)."""


class IndexOutdatedError(RuntimeError):
    """The index was built with a different model or chunk settings; re-index before using it."""


def expected_manifest(dimension: int | None) -> dict:
    """What the manifest of an index built with today's settings looks like."""
    return {
        "format": settings.INDEX_FORMAT,
        "embedding_model": settings.EMBEDDING_MODEL,
        "dimension": dimension,
        "chunking": {"strategy": settings.CHUNK_STRATEGY, "size": settings.CHUNK_SIZE,
                     "overlap": settings.CHUNK_OVERLAP},
    }


def read_manifest() -> dict | None:
    if not _manifest_file().exists():
        return None
    try:
        return json.loads(_manifest_file().read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise IndexDamagedError("the index manifest can't be read") from e


def check_compatible() -> None:
    """
    Raise IndexOutdatedError if the stored index was built with other settings
    (or before settings were recorded), IndexDamagedError if its vectors don't
    match its own manifest. An empty index is always compatible.
    """
    embeddings, metadata = _load()
    if not metadata:
        return
    manifest = read_manifest()
    if manifest is None:
        raise IndexOutdatedError("the index was built before its settings were recorded")
    if embeddings.shape[1] != manifest.get("dimension"):
        raise IndexDamagedError(f"vectors have size {embeddings.shape[1]}, the manifest says {manifest.get('dimension')}")
    want = expected_manifest(manifest["dimension"])
    changed = [k for k in want if manifest.get(k) != want[k]]
    if changed:
        raise IndexOutdatedError(f"index settings changed: {', '.join(changed)}")


def write_atomic(path: Path, write: Callable[[BinaryIO], object]) -> None:
    """Write to a temp file, then swap it in, so a crash leaves the old or the new file, never half of one."""
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        write(f)
    os.replace(tmp, path)


_cache: tuple | None = None  # (file stamp, embeddings, metadata)


def _stamp() -> tuple:
    """Identifies the current index files; changes whenever either file is replaced."""
    out = []
    for path in (EMBEDDINGS_FILE, METADATA_FILE):
        try:
            st = path.stat()
            out.append((str(path), st.st_mtime_ns, st.st_size, st.st_ino))
        except FileNotFoundError:
            out.append((str(path), None))
    return tuple(out)


def _load() -> tuple[np.ndarray, list[dict]]:
    """
    Load embeddings matrix and metadata list. Raises IndexDamagedError.

    The parsed index is reused until either file changes, so a page redraw
    doesn't re-read it from disk several times. Callers must not modify the
    returned objects (none do; every writer builds new ones).
    """
    global _cache
    stamp = _stamp()
    if _cache is not None and _cache[0] == stamp:
        return _cache[1], _cache[2]
    embeddings, metadata = _read()
    _cache = (stamp, embeddings, metadata)
    return embeddings, metadata


def _read() -> tuple[np.ndarray, list[dict]]:
    if not EMBEDDINGS_FILE.exists() and not METADATA_FILE.exists():
        return np.empty((0, 0), dtype=np.float32), []
    try:
        embeddings = np.load(str(EMBEDDINGS_FILE))
        metadata = json.loads(METADATA_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError, EOFError) as e:  # missing half, cut short, not JSON
        raise IndexDamagedError("the index files can't be read") from e
    if embeddings.ndim != 2 or embeddings.shape[0] != len(metadata):
        raise IndexDamagedError(f"{len(embeddings)} vectors but {len(metadata)} metadata entries")
    return embeddings, metadata


def _save(embeddings: np.ndarray, metadata: list[dict], manifest: dict | None = None) -> None:
    """Save the index; write `manifest` too if given (otherwise the existing one is kept)."""
    STORE_DIR.mkdir(parents=True, exist_ok=True)
    # Each file is swapped in whole. The two can't change in one step, so a crash
    # between them is still possible; _load's count check reports it as damaged.
    write_atomic(EMBEDDINGS_FILE, lambda f: np.save(f, embeddings))
    write_atomic(METADATA_FILE, lambda f: f.write(json.dumps(metadata, ensure_ascii=False).encode("utf-8")))
    if manifest is not None:  # last, so it never describes vectors that weren't written
        write_atomic(_manifest_file(), lambda f: f.write(json.dumps(manifest, indent=2).encode("utf-8")))


def _chunk_hash(doc_id: str | None, page: int, position: int, text: str) -> str:
    raw = f"{doc_id}::page{page}::pos{position}::{text}"
    return hashlib.sha256(raw.encode()).hexdigest()


def add_chunks(chunks: list[dict], embeddings: list[list[float]]) -> int:
    """
    Store chunks and their embeddings.
    Returns the number of NEW chunks added (duplicates are skipped).

    Each chunk dict may carry a "doc_id" key (the owning document's registry
    ID — see rag/document_registry.py). It is persisted as-is so retrieval can
    later look up that document's privacy level. Chunks from before doc_id
    existed simply don't have the key; that is the legacy case retriever.py
    treats as unclassified.
    """
    check_compatible()  # never add vectors to an index built with other settings
    existing_vecs, existing_meta = _load()
    new_vecs, new_meta = _entries(chunks, embeddings, {m["hash"] for m in existing_meta})

    if new_vecs:
        new_matrix = np.array(new_vecs, dtype=np.float32)
        if existing_meta and new_matrix.shape[1] != existing_vecs.shape[1]:
            raise IndexOutdatedError(f"new vectors have size {new_matrix.shape[1]}, the index uses {existing_vecs.shape[1]}")
        combined = np.vstack([existing_vecs, new_matrix]) if existing_meta else new_matrix
        _save(combined, existing_meta + new_meta, expected_manifest(new_matrix.shape[1]))

    return len(new_vecs)


def replace_all(chunks: list[dict], embeddings: list[list[float]]) -> int:
    """Replace the whole index with these chunks, stamped with today's settings. Used by re-indexing."""
    vecs, meta = _entries(chunks, embeddings, set())
    matrix = np.array(vecs, dtype=np.float32) if vecs else np.empty((0, 0), dtype=np.float32)
    _save(matrix, meta, expected_manifest(matrix.shape[1] if vecs else None))
    return len(meta)


def _entries(chunks: list[dict], embeddings: list[list[float]], known: set[str]) -> tuple[list, list[dict]]:
    """Metadata entries for chunks whose fingerprint isn't in `known` (duplicates skipped)."""
    vecs, meta = [], []
    for position, (chunk, vec) in enumerate(zip(chunks, embeddings)):
        position = chunk.get("position", position)
        h = _chunk_hash(chunk.get("doc_id"), chunk["page"], position, chunk["text"])
        if h in known:
            continue
        known.add(h)
        vecs.append(vec)
        meta.append({
            "hash": h,  # stable chunk ID
            "doc_id": chunk.get("doc_id"),
            "doc_name": chunk["doc_name"],
            "page": chunk["page"],
            "pages": chunk.get("pages", [chunk["page"]]),
            "section": chunk.get("section", ""),
            "text": chunk["text"],
        })
    return vecs, meta


def search(query_embedding: list[float], top_k: int = 5, *, allowed_doc_ids: set[str]) -> list[dict]:
    """
    Find the top_k most relevant chunks, among documents in `allowed_doc_ids`
    only, using cosine similarity. Returns list of dicts with text, doc_id,
    doc_name, page, distance.

    Filtering happens before ranking, so excluded documents can neither be
    returned nor push allowed ones out of the top_k. The argument is required
    so no caller can search everything by accident.
    """
    check_compatible()  # never compare a query against vectors from another setup
    embeddings, metadata = _load()
    candidates = np.flatnonzero([m.get("doc_id") in allowed_doc_ids for m in metadata])
    if candidates.size == 0:
        return []

    query = np.array(query_embedding, dtype=np.float32)

    # Cosine similarity: dot product of normalized vectors
    vectors = embeddings[candidates]
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    normed = vectors / norms

    query_norm = query / (np.linalg.norm(query) or 1)
    sims = normed @ query_norm  # shape: (len(candidates),)
    similarities = dict(zip(candidates, sims))

    top_indices = candidates[np.argsort(sims)[::-1][:top_k]]

    results = []
    for idx in top_indices:
        m = metadata[idx]
        results.append({
            "text": m["text"],
            "doc_id": m.get("doc_id"),
            "doc_name": m["doc_name"],
            "page": m["page"],
            "pages": m.get("pages", [m["page"]]),
            "section": m.get("section", ""),
            "chunk_id": m["hash"],
            "distance": float(1 - similarities[idx]),
        })
    return results


def list_documents() -> list[dict]:
    """
    Return one entry per unique document currently stored, as
    {"doc_id": ..., "doc_name": ...}. doc_id is None for legacy chunks stored
    before document IDs existed.
    """
    _, metadata = _load()
    seen = {}
    for m in metadata:
        key = (m.get("doc_id"), m["doc_name"])
        seen.setdefault(key, {"doc_id": m.get("doc_id"), "doc_name": m["doc_name"]})
    return sorted(seen.values(), key=lambda d: d["doc_name"])


def remove_document(doc_id: str) -> int:
    """
    Delete all chunks belonging to the document with this doc_id.
    Returns the count of chunks deleted.

    Keyed by doc_id, not doc_name: two different documents could share a
    display name, and before doc_id existed, deletion-by-name had no way to
    find the matching raw file on disk at all (see document_registry.py).
    """
    embeddings, metadata = _load()
    keep_indices = [i for i, m in enumerate(metadata) if m.get("doc_id") != doc_id]
    removed = len(metadata) - len(keep_indices)

    if removed > 0:
        # Keeps the existing manifest: removing chunks doesn't change how the rest were built.
        _save(embeddings[keep_indices], [metadata[i] for i in keep_indices])

    return removed


def total_chunks() -> int:
    _, metadata = _load()
    return len(metadata)


def clear_all() -> int:
    """
    Delete the entire index, including a damaged one. Returns the number of
    chunks removed (0 if the files couldn't be read).

    Needed when chunking settings change (a document re-indexed at a new
    chunk size would otherwise be stored twice) and to recover from damage.
    """
    try:
        removed = len(_load()[1])
    except IndexDamagedError:
        removed = 0

    EMBEDDINGS_FILE.unlink(missing_ok=True)
    METADATA_FILE.unlink(missing_ok=True)
    _manifest_file().unlink(missing_ok=True)

    return removed
