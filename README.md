# Personal AI Document Assistant

A local RAG (retrieval-augmented generation) app: upload PDFs, ask questions in a chat interface, get answers grounded in the document with page citations.

## How it works

1. **Upload** — a PDF is validated (size, page count, signature, not encrypted), stored under a random filename, and text is extracted page by page.
2. **Chunk** — text is split into ~150-word chunks along paragraph boundaries, keeping table rows and headings with their text (strategies and settings: `docs/chunking.md`).
3. **Embed** — each chunk is embedded locally with `all-MiniLM-L6-v2` (`sentence-transformers`) — no API call, no cost.
4. **Store** — vectors and metadata are saved to disk as a `.npy` file + JSON (`rag/vector_store.py`), with simple cosine-similarity search. No external vector database.
5. **Ask** — 20 candidate chunks are found by **hybrid search**: semantic search (meaning) and BM25 keyword search (exact words, names, numbers), merged with Reciprocal Rank Fusion (`docs/search_modes.md`). A local **reranker** (cross-encoder) then re-reads each candidate next to the question and keeps the best 5. If it fails, the search order is used and the answer says so. The 5 chunks are sent to Claude (`claude-sonnet-5-5`) along with the question and recent conversation history, each as a document with **citations** turned on.
6. **Check** — every sentence of the answer comes back with the exact passage it relied on. The app checks in code that each quote really comes from an excerpt it sent, and that every figure in a sentence appears in its quote. Answers show `[1]`-style markers, a one-line **answer check**, and the quoted passages under **Sources**. If nothing in the answer can be backed by a quote, a "couldn't verify" message is shown instead (`rag/citation_check.py`, `docs/phase5_grounded_answers_report.md`).
7. **Follow-ups** — a question like "what about the prior year?" is first rewritten into a standalone question (via Claude Haiku) before retrieval, since the original wording shares no words with the document. Once the conversation includes an answer from a Personal document, rewriting is skipped, so no Claude call happens before you approve.

## Setup

```powershell
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
copy .env.example .env   # then add your ANTHROPIC_API_KEY
.venv\Scripts\streamlit run app.py
```

The app binds to `127.0.0.1` only (see `.streamlit/config.toml`) — it is not reachable from other devices on your network.

## Project layout

```
app.py                    Streamlit UI — upload, chat, document management
rag/
  pdf_loader.py            Extract text from a PDF, page by page
  text_chunker.py          Fixed, paragraph and section chunking strategies
  settings.py              Embedding model, chunk and search settings (one place)
  bm25.py                  Keyword search over the stored chunks
  reindex.py               Rebuild the index from stored PDFs after a settings change
  embeddings.py             Local embedding model (sentence-transformers)
  vector_store.py           Disk-backed vector store, cosine similarity search
  retriever.py               Search modes (semantic / bm25 / hybrid) + RRF fusion
  reranker.py                Local cross-encoder reranking, with safe fallback
  query_rewriter.py          Condenses follow-up questions before retrieval
  answer_generator.py       Sends excerpts to Claude with citations on; builds the shown answer and sources
  citation_check.py          Re-checks excerpts before sending; checks every citation and figure after
security/
  pdf_validator.py          Upload validation: size, pages, signature, encryption
tests/                     pytest suite covering the above
data/
  uploads/                  Stored PDFs (gitignored)
  chroma_db/                Vector store files (gitignored, legacy folder name)
```

## Running tests

```powershell
.venv\Scripts\python -m pytest -q
```

Retrieval benchmark (synthetic PDFs, temp index, no API cost):

```powershell
.venv\Scripts\python -m evaluation.evaluation_runner           # current settings
.venv\Scripts\python -m evaluation.evaluation_runner --modes   # compare search modes
.venv\Scripts\python -m evaluation.evaluation_runner --sweep   # compare chunk settings
.venv\Scripts\python -m evaluation.evaluation_runner --rerank-compare  # reranking on/off, corpus v3
```

Answer benchmark (20 hand-checked questions; **calls the Claude API, about $0.30 per run**: Sonnet answers, an Opus judge grades them):

```powershell
.venv\Scripts\python -m evaluation.answer_eval --name my_run
.venv\Scripts\python -m evaluation.answer_eval --name quick --ids F01 M01   # a few questions, a few cents
```

Architecture overview: `docs/architecture.md`.

## Notes

- Document data (`data/`) and secrets (`.env`) never leave your machine except for the question + retrieved excerpts sent to the Claude API per query.
- **Search settings** (sidebar): search method, reranking on/off, how many candidates to rerank and how many excerpts to send to Claude. They apply to the next question, with no re-index. Each answer has an optional **Evidence details** panel showing the method, how ranks moved, and timings.
- The reranker model (`cross-encoder/ms-marco-MiniLM-L6-v2`, Apache-2.0, ~90 MB) downloads once on first use, pinned to one version. After that it runs offline. Reranking adds about 1 second per question; benchmark in `docs/phase4_reranking_report.md`.
- The index records which model and chunk settings built it (`data/chroma_db/manifest.json`). After a settings change, or the first time you run this version, the app shows **Re-index documents**: one click rebuilds the index from the PDFs already stored, without re-uploading or deleting anything. Benchmark behind the default settings: `docs/phase2_chunking_report.md`.
