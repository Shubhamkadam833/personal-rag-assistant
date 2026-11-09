"""
embeddings.py — converts text into numerical vectors (embeddings).

WHAT IS AN EMBEDDING?
  An embedding is a list of ~384 numbers that captures the *meaning* of a sentence.
  Two sentences about the same topic will have similar numbers, so we can search
  by meaning rather than keyword matching.

MODEL USED:
  "all-MiniLM-L6-v2" — small, fast, and runs entirely on your computer.
  It downloads once (~90 MB) on first use.
"""

from sentence_transformers import SentenceTransformer

# Module-level singleton so we only load the model once
_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Convert a list of strings into a list of embedding vectors."""
    model = get_model()
    vectors = model.encode(texts, show_progress_bar=False)
    return vectors.tolist()


def embed_query(query: str) -> list[float]:
    """Convert a single query string into an embedding vector."""
    return embed_texts([query])[0]
