# How documents are chunked and embedded

Search works on **chunks**, small pieces of a document, not whole files. Each chunk is turned into an **embedding**: a list of numbers capturing its meaning, made by a local Sentence Transformers model. A question is embedded the same way, and the closest chunks are sent to Claude.

All settings live in `rag/settings.py`.

## The three chunking strategies

| Strategy | What it does | Good at | Weak at |
|---|---|---|---|
| `fixed` | Cuts each page into windows of N words, overlapping by M words. Never crosses a page. | Simple and predictable. | Can cut a sentence, a table row, or a heading away from its text. |
| `paragraph` (default) | Packs whole paragraphs into chunks of up to N words. It can cross pages, and records every page it spans. A paragraph that is too long is split at line breaks, never in the middle of a line. A chunk never ends on a heading. | Keeps table rows ("Total revenue: FY2024 412.6") and headings with their text. Fewer, fuller chunks. | Relies on the PDF's paragraph layout. Odd layouts give odd paragraphs. |
| `section` | Like `paragraph`, but never crosses a heading. The heading is repeated at the top of every chunk in its section. | Each chunk clearly belongs to one section. | Short sections give many small chunks. On the benchmark it ranked answers worse (see the report). |

**Headings** are detected from the PDF itself: a block noticeably larger than the document's normal text, or a short line in bold. A PDF with no visible styling has no detectable headings, so `section` then behaves like `paragraph`.

**Overlap** repeats the end of one chunk at the start of the next, so an answer that falls on a boundary isn't lost. It must be smaller than the chunk size. Impossible settings raise an error instead of looping forever.

**Every chunk records** its document, its first page (used in citations), all the pages it spans, its nearest heading (`section`), its position, and a stable ID: a SHA-256 of document ID, page, position and text.

## Why not bigger chunks?

The default model, `all-MiniLM-L6-v2`, reads only the **first 256 tokens** of a chunk and ignores the rest. 150 words fits. At 300 and 500 words, most chunks go past the limit, so the end of each chunk is never searched. On the benchmark this lowered recall. See `docs/phase2_chunking_report.md`.

## Changing settings safely

The index records how it was built, in `data/chroma_db/manifest.json`: the model, the vector size, and the chunk strategy, size and overlap.

If you change the model or chunk settings, the app sees the mismatch and stops search instead of mixing incompatible vectors. It then offers **Re-index documents**, which:
- re-reads the PDFs already stored in `data/uploads`, so nothing is uploaded again or deleted
- builds the whole new index before replacing the old one, so a failure leaves the old index intact
- reuses old vectors for text that didn't change, when the model is the same
- lists any document whose stored file is missing, instead of dropping it silently

To try another embedding model, set `RAG_EMBEDDING_MODEL` (for example in `.env`), restart, and click Re-index.
