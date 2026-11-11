"""
document_registry.py — tracks one record per uploaded document.

WHY THIS EXISTS:
  vector_store.py knows about CHUNKS — pieces of text with embeddings. It has
  no concept of "the document" as a whole: no stable ID, no link back to the
  file on disk, no privacy level. That gap is what let document deletion break
  silently (removing a document's chunks left its raw PDF orphaned on disk —
  stored files are random filenames with no link back to the document name
  recorded anywhere) and let "should this be sent to Claude?" have no answer.

  This module is that missing layer: one JSON record per document, keyed by
  a SHA-256 of its full raw bytes.

WHY SHA-256 OF THE RAW BYTES (not the extracted text):
  It is the strongest, simplest statement of "this is the same file" — two
  uploads produce the same ID only if they are byte-for-byte identical. This
  catches true re-uploads reliably and never confuses two different files
  that happen to extract to similar text.

PRIVACY LEVELS:
  "public"     — no restriction.
  "personal"   — requires explicit per-question approval before its excerpts
                  are sent to the Claude API (enforced in app.py).
  "restricted" — never sent to the Claude API, under any circumstance
                  (enforced in retriever.py, before generate_answer() is ever
                  called — not merely requested via the system prompt).

PERSISTENCE:
  One JSON file, sibling to vector_store.py's on-disk files, saved after
  every change so it survives app restarts.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

from rag import vector_store

STORE_DIR = Path(__file__).parent.parent / "data" / "chroma_db"
REGISTRY_FILE = STORE_DIR / "documents.json"

PRIVACY_LEVELS = ("public", "personal", "restricted")
DEFAULT_PRIVACY = "personal"


def _load() -> dict:
    if not REGISTRY_FILE.exists():
        return {}
    try:
        return json.loads(REGISTRY_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise vector_store.IndexDamagedError("the document registry can't be read") from e


def _save(registry: dict) -> None:
    STORE_DIR.mkdir(parents=True, exist_ok=True)
    data = json.dumps(registry, ensure_ascii=False, indent=2).encode("utf-8")
    vector_store.write_atomic(REGISTRY_FILE, lambda f: f.write(data))


def add(
    doc_id: str,
    display_name: str,
    privacy: str,
    stored_path: str,
    flagged: bool = False,
) -> None:
    """
    Register a newly uploaded document.

    `flagged` records whether security.prompt_guard found injection-like
    phrasing anywhere in the document, so the UI can keep showing a warning
    badge after the upload moment has passed.
    """
    if privacy not in PRIVACY_LEVELS:
        raise ValueError(f"Unknown privacy level: {privacy!r}")

    registry = _load()
    registry[doc_id] = {
        "display_name": display_name,
        "privacy": privacy,
        "stored_path": stored_path,
        "sha256": doc_id,
        "flagged": flagged,
        "added_at": datetime.now(timezone.utc).isoformat(),
    }
    _save(registry)


def get(doc_id: str) -> dict | None:
    return _load().get(doc_id)


def is_indexed(doc_id: str) -> bool:
    """
    True if this exact file is registered AND still has chunks in the index.

    A record with no chunks (e.g. the index files were deleted by hand to
    recover from damage) is dropped, so it can't block re-uploading the file.
    """
    if doc_id not in _load():
        return False
    if any(d["doc_id"] == doc_id for d in vector_store.list_documents()):
        return True
    remove(doc_id)
    return False


def remove(doc_id: str) -> dict | None:
    """Remove a document's registry entry. Returns the removed record, if any."""
    registry = _load()
    record = registry.pop(doc_id, None)
    if record is not None:
        _save(registry)
    return record


def list_all() -> dict:
    """Return the full registry: {doc_id: record}."""
    return _load()


def clear_all() -> int:
    """
    Delete every registry entry. Returns the number removed.

    Called together with vector_store.clear_all(), and works on a damaged
    registry (returns 0 if it couldn't be read).
    """
    try:
        count = len(_load())
    except vector_store.IndexDamagedError:
        count = 0
    REGISTRY_FILE.unlink(missing_ok=True)
    return count
