"""
reranker.py — a second, more careful relevance check on the search results.

WHY:
  Search (semantic, BM25 or hybrid) is fast because it never reads the
  question and a chunk *together*: each is turned into numbers or word counts
  separately. That's how "FY2023 operating income" can rank above "FY2024
  operating income" for a FY2024 question. A cross-encoder reads the question
  and one chunk side by side, word by word, and scores how well that chunk
  answers it. It's too slow to run over every chunk, so search picks ~20
  candidates and the cross-encoder re-orders just those.

MODEL:
  cross-encoder/ms-marco-MiniLM-L6-v2 (Apache-2.0, ~90 MB), runs locally:
  no API call, no cost, document text never leaves the machine. The download
  is pinned to one commit (settings.RERANK_MODEL_REVISION), loads the
  safetensors weights, and never runs code from the model repo.

SCORES ARE NOT PROBABILITIES:
  A higher score means "more relevant than the others in this list". It is not
  a chance of being correct and can be negative.

FAIL-SAFE:
  If the model can't load, returns bad scores, or takes longer than
  settings.RERANK_TIMEOUT_S, rerank() returns the search results in their
  original order with a `fallback` reason. The input list was already
  privacy-filtered, so falling back never widens what can be shown or sent.
  Logs carry counts, timings and reasons only, never document text.
"""

import concurrent.futures
import logging
import math
import time

from rag import settings

log = logging.getLogger(__name__)

_model = None
_load_error: str | None = None
# One worker: predictions run one at a time, and a timed-out one can finish in
# the background without blocking the app.
# shortcut: a timed-out prediction keeps its CPU until it finishes (threads
# can't be killed); move to a subprocess if timeouts become common.
_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="reranker")


def _load_model():
    from sentence_transformers import CrossEncoder  # imported here so tests with a fake model never load torch

    kwargs = dict(revision=settings.RERANK_MODEL_REVISION, max_length=settings.RERANK_MAX_LENGTH,
                  trust_remote_code=False)
    try:
        return CrossEncoder(settings.RERANK_MODEL, local_files_only=True, **kwargs)
    except OSError:
        return CrossEncoder(settings.RERANK_MODEL, **kwargs)  # first run: download once


def get_model():
    """The loaded cross-encoder. A failed load is remembered (until restart) so
    every question doesn't retry a slow download; raises RuntimeError."""
    global _model, _load_error
    if _model is None and _load_error is None:
        try:
            _model = _load_model()
        except Exception as e:  # noqa: BLE001 - network, disk, memory, bad files: all mean "fall back"
            _load_error = type(e).__name__
            log.warning("reranker model unavailable (%s); using search order", _load_error)
    if _model is None:
        raise RuntimeError(f"reranker model unavailable ({_load_error})")
    return _model


def _key(chunk: dict):
    return chunk.get("chunk_id") or (chunk.get("doc_id"), chunk.get("page"), chunk.get("text"))


def unique(candidates: list[dict]) -> list[dict]:
    """Drop repeated chunks (same chunk ID), keeping the first, best-ranked copy."""
    seen, out = set(), []
    for c in candidates:
        if _key(c) not in seen:
            seen.add(_key(c))
            out.append(c)
    return out


def rerank(question: str, candidates: list[dict], top_n: int = settings.FINAL_K) -> tuple[list[dict], dict]:
    """
    Re-order `candidates` (already privacy-filtered search results) by
    cross-encoder relevance to `question`; return the best `top_n` and info.

    Each returned chunk is a copy with every original field (document, page,
    chunk ID: citations are never rebuilt) plus `original_rank`, `final_rank`
    and `rerank_score` (None when falling back). Equal scores keep search order.

    info = {"reranked": bool, "fallback": reason or None, "seconds": float, "candidates": int}
    """
    start = time.perf_counter()
    pool = [{**c, "original_rank": i} for i, c in enumerate(unique(candidates), start=1)]
    info = {"reranked": False, "fallback": None, "seconds": 0.0, "candidates": len(pool)}

    def fallback(reason: str):
        log.warning("reranking skipped: %s (%d candidates)", reason, len(pool))
        info.update(fallback=reason, seconds=round(time.perf_counter() - start, 3))
        return [{**c, "final_rank": c["original_rank"], "rerank_score": None} for c in pool[:top_n]], info

    if not pool:
        return [], info
    if not all(isinstance(c.get("text"), str) for c in pool):
        return fallback("a candidate has no text")
    try:
        model = get_model()
    except RuntimeError:
        return fallback("model unavailable")

    pairs = [(question, c["text"]) for c in pool]
    future = _executor.submit(model.predict, pairs, show_progress_bar=False)
    try:
        raw = future.result(timeout=settings.RERANK_TIMEOUT_S)
    except concurrent.futures.TimeoutError:
        return fallback(f"took longer than {settings.RERANK_TIMEOUT_S:g}s")
    except Exception as e:  # noqa: BLE001 - any model failure must fall back, never crash the chat
        return fallback(f"model error ({type(e).__name__})")

    try:
        scores = [float(s) for s in raw]
    except (TypeError, ValueError):
        return fallback("invalid scores")
    if len(scores) != len(pool) or not all(math.isfinite(s) for s in scores):
        return fallback("invalid scores")

    order = sorted(range(len(pool)), key=lambda i: (-scores[i], i))  # tie -> earlier search rank wins
    out = [{**pool[i], "final_rank": rank, "rerank_score": round(scores[i], 4)}
           for rank, i in enumerate(order[:top_n], start=1)]
    info.update(reranked=True, seconds=round(time.perf_counter() - start, 3))
    return out, info
