# Architecture

How a PDF becomes searchable, and how a question becomes an answer. Every box runs on your computer except the two Claude calls.

## Upload (once per PDF)

```
PDF ──► validate ──► extract text ──► chunk ──► clean ──► embed ──► store
        size, pages,  time-limited     paragraph   prompt-     local      embeddings.npy
        signature,    subprocess       150/30      injection   MiniLM     metadata.json
        encryption    (pdf_sandbox)    (text_      scan        (embed-    manifest.json
        (pdf_validator)                chunker)    (prompt_    dings)     documents.json
                                                   guard)                 (vector_store,
                                                                           document_registry)
```

## Question (every time)

```
question
   │
   ├─ follow-up? ─► rewrite to stand alone ............ Claude Haiku (privacy-gated)
   │
   ▼
privacy filter: only Public + Personal documents ......... retriever.py
   │            (Restricted are removed BEFORE ranking)
   ▼
search, 20 candidates ..................................... retriever.search
   ├─ semantic: cosine similarity on embeddings ........... vector_store.py
   ├─ bm25:     keyword scoring on the same chunks ........ bm25.py
   └─ hybrid:   both, merged by Reciprocal Rank Fusion .... retriever.rrf_fuse   ◄ default
   │
   ▼
rerank: local cross-encoder re-reads each candidate ....... reranker.py         ◄ default on
   │    next to the question; keeps the best 5
   │    (fails safe: on error/timeout, keeps search order and says so)
   ▼
Personal excerpts? ─► ask the user before sending ......... app.py + privacy_gate.py
   │
   ▼
answer from the excerpts only, with citations ............. Claude Sonnet (answer_generator.py)
   │
   ▼
answer + sources + optional Evidence details (method, rank changes, timings)
```

## Where the settings live

`rag/settings.py` holds them all.

**Changing these makes the index outdated** (the app offers a one-click re-index):
- embedding model
- chunk strategy, size and overlap

**These apply to the next question, with no re-index.** Users can also change them in the sidebar under *Search settings*:
- search mode
- RRF constant and depth
- reranking on/off
- reranker model and timeout
- candidate count
- excerpt count

## Why each piece exists

| Piece | Plain English | Measured in |
|---|---|---|
| Paragraph chunking | Cut at paragraph breaks so a table row or heading stays with its text | `docs/phase2_chunking_report.md` |
| Hybrid search | Meaning + exact words: catches paraphrases *and* names, codes and years | `docs/phase3_hybrid_search_report.md` |
| Reranking | A slower, closer read of the top 20 that puts the right year and company first | `docs/phase4_reranking_report.md` |
| Privacy filter before ranking | Restricted text can never be ranked, reranked, shown or sent | `tests/test_retriever_privacy.py`, `tests/test_reranker.py` |
