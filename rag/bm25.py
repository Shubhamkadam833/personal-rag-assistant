"""
bm25.py — keyword search over the chunks already in the vector store.

WHAT IS BM25?
  A classic scoring formula that ranks a chunk higher when it contains the
  question's words, especially rare ones ("FY2024", "Northwind", "$5,000").
  Semantic search matches meaning and can miss exact names and numbers;
  BM25 matches exact words and misses paraphrases. Together they cover both.

  For each question word found in a chunk:
      idf   = how rare the word is across chunks (rare words count more)
      tf    = how often it appears in this chunk, with diminishing returns (k1)
      b     = a penalty for long chunks, so they don't win just by being long
  A chunk's score is the sum over the question's words.

NO SECOND INDEX:
  Word counts are built from the chunk texts in vector_store's metadata and
  rebuilt whenever those files change, so keyword search can never drift out
  of sync with semantic search (a deleted document disappears from both).

PRIVACY:
  Like vector_store.search, `allowed_doc_ids` is required and applied before
  scoring. Word rarity (idf) is computed over allowed chunks only, so a
  Restricted document can't even influence the scores of allowed ones.
"""

import math
import re
from collections import Counter

from rag import vector_store

K1 = 1.5   # standard BM25 defaults
B = 0.75

# Words too common to say anything about relevance. A question made only of
# these returns no keyword results instead of every chunk.
STOPWORDS = frozenset(
    "a an and are as at be by did do does for from had has have how i in is it its "
    "me my of on or that the their this to was were what when where which who why "
    "will with you your".split()
)

_cache: tuple | None = None  # (index file stamp, per-chunk word counts, per-chunk lengths)


def tokenize(text: str) -> list[str]:
    """Lower-case words and numbers, minus stopwords. '412.6' and 'fy2024' stay whole."""
    return [t for t in re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", text.lower()) if t not in STOPWORDS]


def _word_counts(metadata: list[dict]) -> tuple[list[Counter], list[int]]:
    global _cache
    stamp = vector_store._stamp()
    if _cache is None or _cache[0] != stamp:
        counts = [Counter(tokenize(m["text"])) for m in metadata]
        _cache = (stamp, counts, [sum(c.values()) for c in counts])
    return _cache[1], _cache[2]


def search(question: str, top_k: int = 5, *, allowed_doc_ids: set[str]) -> list[dict]:
    """
    Top_k chunks by BM25 score among documents in `allowed_doc_ids`. Same
    result fields as vector_store.search, with `bm25_score` instead of
    `distance`. Chunks sharing no word with the question are never returned.
    """
    vector_store.check_compatible()  # same rule as semantic search: never use an outdated index
    _, metadata = vector_store._load()
    terms = set(tokenize(question))
    allowed = [i for i, m in enumerate(metadata) if m.get("doc_id") in allowed_doc_ids]
    if not terms or not allowed:
        return []

    counts, lengths = _word_counts(metadata)
    n = len(allowed)
    avg_len = sum(lengths[i] for i in allowed) / n or 1
    # shortcut: pure-Python loop over allowed chunks; fine for thousands of
    # chunks, move to a sparse matrix if the index grows to ~100k.
    idf = {}
    for term in terms:
        df = sum(1 for i in allowed if term in counts[i])
        idf[term] = math.log(1 + (n - df + 0.5) / (df + 0.5))

    scored = []
    for i in allowed:
        score = 0.0
        for term in terms:
            tf = counts[i].get(term, 0)
            if tf:
                score += idf[term] * tf * (K1 + 1) / (tf + K1 * (1 - B + B * lengths[i] / avg_len))
        if score > 0:
            scored.append((score, i))

    # Ties keep index order, so the same question always gives the same ranking.
    scored.sort(key=lambda s: (-s[0], s[1]))
    return [{**vector_store.result_fields(metadata[i]), "bm25_score": round(score, 4)}
            for score, i in scored[:top_k]]
