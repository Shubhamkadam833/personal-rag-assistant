# Phase 1 — Repository audit and baseline

Date: 2026-11-10 · Branch: `phase-1/baseline` · Results: [`evaluation/results/baseline.json`](../evaluation/results/baseline.json)

## Summary

- The app has a solid security base (PRs #1–#4) and one critical privacy gap, fixed separately in `fix/history-privacy`.
- A repeatable retrieval benchmark now exists: 31 hand-verified questions over a synthetic 4-document corpus, scored without API calls.
- **Baseline semantic search: Recall@5 0.962, MRR@10 0.863, p95 latency 27 ms.**
- **These numbers are flattering and should not be read as "search is already good."** The corpus has only 30 chunks, so top-5 returns a sixth of everything. The spec targets 1,000+ chunks. Growing the corpus comes before any "hybrid beats semantic" claim can mean much.

## Test and scan results

| Check | Result |
|---|---|
| Existing test suite (on `main`) | 67 passed |
| After this branch | 109 passed (42 new: metric math, benchmark ground truth) |
| Bandit (`rag security evaluation app.py`, 1,370 lines) | 2 Low, 0 Medium, 0 High. Both are the runner's fixed `git rev-parse` call (B603, B607): no user input, intentional, left visible rather than suppressed. |
| pip-audit | **Not run.** It needs network access to PyPI, which the dev sandbox blocks. Run locally: `.venv\Scripts\pip-audit` |
| Real `data/` folder | Checksummed before and after the benchmark run: unchanged |

## Baseline measurements

Config: semantic search, fixed 150-word chunks with 30-word overlap, `all-MiniLM-L6-v2`, no reranking. Corpus v1, dataset v1. Two runs gave identical metrics, so the benchmark is deterministic.

| Metric | Value |
|---|---|
| Recall@5 (26 answerable questions) | 0.962 |
| MRR@10 | 0.863 |
| Evidence intactness (label and value in one chunk) | 1.000 |
| Retrieval latency p50 / p95 (warm, local Windows) | 21.7 / 27.0 ms |
| Chunks indexed | 30 |

Recall@5 by category: direct 0.875 · numerical 1.0 · multi-document 1.0 · multi-section 1.0 · adversarial 1.0

**Failures and weak spots:**
- **D01 "Who is the CEO of Northwind Robotics?"** The right chunk is ranked 6th, outside the top 5. The question says "CEO" and the document says "Chief Executive Officer", and the overview sentence shares its chunk with generic filler. BM25 alone won't fix this, because it matches words, not abbreviations. Reranking is the likelier fix.
- **Ranked low but still in the top 5:** D02 (rank 5), M03 (rank 4), N05 (rank 3), S01 (rank 2). The pattern: facts in short overview sentences lose to boilerplate-heavy chunks from the same document.
- **Intactness is 1.0** only because the synthetic tables are short, one-line rows. Real annual-report tables will split. This metric matters more once real-world-like layouts are added.

**Not measured yet (by design):** answer correctness, faithfulness, citation accuracy, abstention, and API cost. Each needs paid Claude calls, and they're scheduled for Phases 5–6.

## Requirements matrix

Done = implemented and tested · Partial = some of it exists · Missing = not started

| Req | Status | Where / gap |
|---|---|---|
| FR-001 PDF processing | Partial | Multi-upload, page metadata, doc IDs, SHA-256 dedup, deletion, privacy, validation, sandboxed parsing: done (`app.py`, `security/`, `rag/document_registry.py`). Gaps: no table detection, no section metadata, and the scanned-page warning is document-level only (a doc with *some* image pages gets no warning). |
| FR-002 Chunking | Partial | Fixed-size only (`rag/text_chunker.py`). No paragraph or section strategy, no heading or table-row protection. The rebuild exists only as a manual button. |
| FR-003 Embeddings | Partial | One model, batched. The model name, version, and dimension are not stored with the index, so mixing models goes undetected. The dimension 384 is hardcoded in `vector_store.py`. |
| FR-004 Hybrid search | Missing | Semantic only. No BM25 or RRF, and no document-scope filter. Privacy filtering exists and is enforced in `retriever.py`. |
| FR-005 Reranking | Missing | |
| FR-006 Grounded answers | Partial | Evidence-only prompt, injection guards, output redaction, follow-ups: done. Privacy approval on rewriting and history: fixed in `fix/history-privacy`. Gaps: the fallback text differs from the spec's required sentence; no handling of figures from different years or units. |
| FR-007 Citations | Partial | Only the excerpts the model reports using are cited, and none on a refusal (`answer_generator.py`). Gaps: no chunk IDs or excerpts in citations, no validation that cited text supports the claim. |
| FR-008 Evaluation | Partial | Retrieval benchmark (31 of the 100 required questions) and metrics: this branch. Answer-level metrics: missing. |
| NFR-001 Performance | Partial | Measured at 30 chunks; the 1,000-chunk target isn't tested yet. |
| NFR-002 Security | Done (pending PR A merge) | See below. |
| NFR-003 Reliability | Partial | Failed uploads clean up their file. Index writes are not atomic, and corrupted index files aren't detected. |
| NFR-004 Maintainability | Partial | Modular code with tests. No central config; type hints are incomplete. |
| UI additions / dashboard | Missing | |

## Issues found (ranked)

1. **[Critical, fixed in PR A] Personal data could reach Claude without approval** via chat history and the query rewriter. See `fix/history-privacy`.
2. **[High] Non-atomic index writes.** `vector_store._save` writes `embeddings.npy` and then `metadata.json`. A crash between the two leaves vectors and text mismatched, and search would then return the wrong text for a vector, silently. Fix: write to temp files and atomically replace, plus a row-count check on load.
3. **[High] No embedding-model record in the index** (FR-003). Changing models would silently mix incompatible vectors.
4. **[Medium] Chunk IDs aren't guaranteed unique.** `_chunk_hash` uses doc name + page + the first 100 characters, so two chunks on one page that start the same way collide, and the second is silently dropped as a "duplicate." Fix: include the chunk's position or its full text in the hash.
5. **[Medium] Every search reloads and re-normalises the whole index from disk.** Fine at 30 chunks (27 ms); it will grow linearly. Cache in memory and invalidate on write.
6. **[Low] Scanned-page warning only fires when the whole document has no text.**
7. **[Low] Storage paths are module constants.** The benchmark runner works around this by overriding them (recorded as a shortcut). A central config should replace both.

## Recommended next PR (Phase 2, first step)

**"Index integrity": atomic writes, an index manifest, and unique chunk IDs.** This fixes issues 2–4 together:
- A small `index_manifest.json` stored alongside the index: embedding model, dimension, chunking config, chunk count, and format version. Search refuses to run, with a clear message, when the manifest doesn't match the current config.
- Write to temp files, then atomically replace them.
- Chunk ID = hash of doc ID + page + position + text.

It's small, it's a prerequisite for safely trying several chunking configs (the next Phase 2 step), and it can be checked with the benchmark: the metrics should stay identical after the change.

Before adding hybrid search, grow the corpus toward the spec's scale, so the comparison between methods is meaningful.
