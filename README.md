# Personal AI Document Assistant

A local RAG (retrieval-augmented generation) app: upload PDFs, ask questions in a chat interface, get answers grounded in the document with page citations.

## How it works

1. **Upload** — a PDF is validated (size, page count, signature, not encrypted), stored under a random filename, and text is extracted page by page.
2. **Chunk** — text is split into ~150-word chunks along paragraph boundaries, keeping table rows and headings with their text (strategies and settings: `docs/chunking.md`).
3. **Embed** — each chunk is embedded locally with `all-MiniLM-L6-v2` (`sentence-transformers`) — no API call, no cost.
4. **Store** — vectors and metadata are saved to disk as a `.npy` file + JSON (`rag/vector_store.py`), with simple cosine-similarity search. No external vector database.
5. **Ask** — a question is embedded, the top-k most similar chunks are retrieved, and sent to Claude (`claude-sonnet-5-5`) along with the question and recent conversation history. Claude is instructed to answer only from the provided excerpts and cite document + page.
6. **Follow-ups** — a question like "what about the prior year?" is first rewritten into a standalone question (via Claude Haiku) before retrieval, since the original wording shares no words with the document.

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
  settings.py              Embedding model and chunk settings (one place)
  reindex.py               Rebuild the index from stored PDFs after a settings change
  embeddings.py             Local embedding model (sentence-transformers)
  vector_store.py           Disk-backed vector store, cosine similarity search
  retriever.py               Combines embeddings + vector store into search
  query_rewriter.py          Condenses follow-up questions before retrieval
  answer_generator.py       Builds the prompt, calls Claude, extracts the answer
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

## Notes

- Document data (`data/`) and secrets (`.env`) never leave your machine except for the question + retrieved excerpts sent to the Claude API per query.
- The index records which model and chunk settings built it (`data/chroma_db/manifest.json`). After a settings change, or the first time you run this version, the app shows **Re-index documents**: one click rebuilds the index from the PDFs already stored, without re-uploading or deleting anything. Benchmark behind the default settings: `docs/phase2_chunking_report.md`.
