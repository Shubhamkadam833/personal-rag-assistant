"""
settings.py — one place for the chunking and embedding settings.

Changing any of these makes the stored index "outdated": search refuses it,
and the app offers a one-click re-index from the stored PDFs (rag/reindex.py).
That way vectors from different setups are never mixed.

The chunk settings were chosen with the Phase 2 benchmark; see
docs/phase2_chunking_report.md before changing them.
"""

import os

EMBEDDING_MODEL = os.getenv("RAG_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
EMBEDDING_BATCH_SIZE = 32

CHUNK_STRATEGY = "paragraph"  # "fixed" | "paragraph" | "section" — see docs/chunking.md
CHUNK_SIZE = 150          # words
CHUNK_OVERLAP = 30        # words, must be smaller than CHUNK_SIZE

# Bump when the stored index layout changes, so old indexes are re-built.
INDEX_FORMAT = 2
