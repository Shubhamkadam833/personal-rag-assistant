"""
retrieval_metrics.py — scores how well search found the right evidence.

RELEVANCE RULE:
  A retrieved chunk is "relevant" to an evidence item if it comes from the
  evidence document AND contains the evidence phrase. Matching by text, not by
  chunk ID, keeps the benchmark valid when chunking changes, because chunk
  IDs change every time chunk boundaries move.

METRICS:
  recall@k       — share of a question's evidence items found in the top k.
  reciprocal rank — 1 / position of the first relevant chunk (0 if none in top k).
                    Averaged over questions, this is MRR: rewards ranking the
                    right chunk first, not just somewhere in the list.
  intactness     — does ANY chunk in the index hold the whole phrase? If not,
                    chunking split a label from its value, and no search method
                    can retrieve it in one piece.
"""

import math
import re


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def is_relevant(chunk: dict, evidence: dict) -> bool:
    return chunk["doc_name"] == evidence["doc"] and normalize(evidence["phrase"]) in normalize(chunk["text"])


def recall_at_k(results: list[dict], evidence: list[dict], k: int) -> float:
    if not evidence:
        raise ValueError("recall is undefined for a question with no evidence")
    top = results[:k]
    found = sum(1 for ev in evidence if any(is_relevant(c, ev) for c in top))
    return found / len(evidence)


def reciprocal_rank(results: list[dict], evidence: list[dict], k: int = 10) -> float:
    for rank, chunk in enumerate(results[:k], start=1):
        if any(is_relevant(chunk, ev) for ev in evidence):
            return 1 / rank
    return 0.0


def evidence_intact(all_chunks: list[dict], evidence: dict) -> bool:
    return any(is_relevant(c, evidence) for c in all_chunks)


def percentile(values: list[float], p: float) -> float:
    """Nearest-rank percentile, e.g. p=95. Simple and exact for small samples."""
    if not values:
        raise ValueError("no values")
    ordered = sorted(values)
    index = max(0, math.ceil(p / 100 * len(ordered)) - 1)
    return ordered[index]
