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

STORE_DIR = Path(__file__).parent.parent / "data" / "chroma_db"
REGISTRY_FILE = STORE_DIR / "documents.json"

PRIVACY_LEVELS = ("public", "personal", "restricted")
DEFAULT_PRIVACY = "personal"


def _load() -> dict:
    if REGISTRY_FILE.exists():
        return json.loads(REGISTRY_FILE.read_text(encoding="utf-8"))
    return {}


def _save(registry: dict) -> None:
    STORE_DIR.mkdir(parents=True, exist_ok=True)
    REGISTRY_FILE.write_text(json.dumps(registry, ensure_ascii=False, indent=2), encoding="utf-8")


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


def find_by_hash(sha256: str) -> str | None:
    """Return the doc_id of an already-registered document with this hash, if any.

    Since doc_id IS the sha256, this is just an existence check — kept as its
    own function so callers read as "is this a duplicate?" rather than poking
    at the ID scheme directly.
    """
    return sha256 if sha256 in _load() else None


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

    Must be called together with vector_store.clear_all() — if the registry
    survives a chunk wipe, re-uploading the exact same file afterwards would
    be rejected by find_by_hash() as "already loaded" even though its chunks
    are gone, permanently orphaning that document.
    """
    registry = _load()
    count = len(registry)
    REGISTRY_FILE.unlink(missing_ok=True)
    return count
